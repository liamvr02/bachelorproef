"""
partial_corr.py  (streaming / online)
=========================================
Partial correlations of every registry feature with temperature, after
removing shared variance with temporal (and optionally spatial) confounders.

Algorithm  --  Frisch-Waugh-Lovell theorem, single streaming pass
-----------------------------------------------------------------
For each feature x_i and target y (temperature), controlling for Z:

  Accumulate per feature (rows where x_i is non-NaN):
    ZtZ_i  (q x q)  -- z z' sum
    Ztx_i  (q,)     -- z x_i sum
    Zty_i  (q,)     -- z y sum
    xtx_i  scalar   -- x_i^2 sum
    xty_i  scalar   -- x_i y sum
    yty_i  scalar   -- y^2 sum
    n_i    int      -- valid row count

  Compute:
    beta_x  = (ZtZ_i)^-1  Ztx_i          (OLS of x on Z)
    beta_y  = (ZtZ_i)^-1  Zty_i          (OLS of y on Z)
    x'M_Z y = xty_i - Ztx_i' beta_y      (cross-product of residuals)
    x'M_Z x = xtx_i - Ztx_i' beta_x
    y'M_Z y = yty_i - Zty_i' beta_y
    partial_r = x'M_Z y / sqrt(x'M_Z x * y'M_Z y)
    partial_t = partial_r * sqrt((n_i - q - 1) / (1 - partial_r^2))
    partial_p = 2 * t_cdf(-|partial_t|, df = n_i - q - 1)

Confounders Z (default, q=6):
  1, year_norm, sin(2pi*month/12), cos(2pi*month/12),
  sin(2pi*hour/24), cos(2pi*hour/24)

  With --spatial-confounders (q=8): add lat_norm, lon_norm.

Usage
-----
    python src/ds/partial_corr.py [--max-rows N] [--batch-size N]
                                  [--spatial-confounders] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

_DS_DIR      = Path(__file__).parent
_SRC         = _DS_DIR.parent
_OUT_DEFAULT = _SRC / "ds_reports" / "partial_corr.json"

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_DS_DIR))

from stream_configs.presets import all_rows
from stream_configs.registry import build_registry


_YEAR_MU  = 2012.5
_YEAR_SIG = 6.25
_LAT_MU   = 51.0
_LAT_SIG  = 0.1
_LON_MU   = 3.71
_LON_SIG  = 0.1

# ── Greening / morphology column classification ───────────────────────────────
#
# Greening features (X) : trees_* + ndvi + UA vegetation codes
# Morphology confounders: dhm_* + all non-vegetation UA codes
#
# UA vegetation codes (Urban Atlas 2018) matched by column-name fragment:
#   14100 green urban areas, 31000 forests, 32000 herbaceous vegetation,
#   40000 wetlands, 21000/22000/23000 agricultural / semi-natural land.
# Any ua_* column NOT matching these fragments is treated as built morphology.

_UA_VEG_KW = frozenset({
    "14100", "green_urban", "greenurban",
    "31000", "forest",
    "32000", "herbaceous",
    "40000", "wetland",
    "21000", "22000", "23000", "arable", "pasture", "agricultural",
    "vegetation",
})


def _is_greening(col: str) -> bool:
    if col.startswith("trees_") or col == "ndvi":
        return True
    if col.startswith("ndvi_struct"):          # structural (temporal-mean) NDVI
        return True
    if col.startswith("ua_"):
        cl = col.lower()
        return any(kw in cl for kw in _UA_VEG_KW)
    return False


def _is_morphology(col: str) -> bool:
    if col.startswith("dhm_"):
        return True
    if col.startswith("ua_") and not _is_greening(col):
        return True
    return False


def _build_Z(
    df,
    include_spatial: bool = False,
) -> np.ndarray:
    """
    Build the confounder matrix Z from a batch.

    Columns (always):
      0  intercept
      1  year_norm
      2  sin(2pi * month / 12)
      3  cos(2pi * month / 12)
      4  sin(2pi * hour / 24)
      5  cos(2pi * hour / 24)

    Additional columns when include_spatial=True:
      6  lat_norm
      7  lon_norm
    """
    n = len(df)
    yr  = (df["year"].to_numpy(dtype=np.float64)          - _YEAR_MU) / _YEAR_SIG
    mo  = df["month_of_year"].to_numpy(dtype=np.float64)
    hr  = df["hour_of_day"].to_numpy(dtype=np.float64)

    cols = [
        np.ones(n),
        yr,
        np.sin(2.0 * np.pi * mo / 12.0),
        np.cos(2.0 * np.pi * mo / 12.0),
        np.sin(2.0 * np.pi * hr / 24.0),
        np.cos(2.0 * np.pi * hr / 24.0),
    ]
    if include_spatial:
        lat = (df["latitude"].to_numpy(dtype=np.float64)  - _LAT_MU) / _LAT_SIG
        lon = (df["longitude"].to_numpy(dtype=np.float64) - _LON_MU) / _LON_SIG
        cols += [lat, lon]

    return np.column_stack(cols)   # (n, q)


def _build_morph_Z(df, morph_cols: List[str]) -> np.ndarray:
    """
    Extract morphology columns from *df*, standardise each to mean=0 std=1,
    and fill NaN with 0 (= column mean post-standardisation).
    Returns an (n, len(morph_cols)) float64 array, never NaN.
    """
    if not morph_cols:
        return np.empty((len(df), 0), dtype=np.float64)
    mat = df[morph_cols].to_numpy(dtype=np.float64)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        finite = col[np.isfinite(col)]
        if len(finite) < 2:
            mat[:, j] = 0.0
            continue
        mu = float(finite.mean())
        sd = float(finite.std())
        if sd < 1e-10:
            mat[:, j] = 0.0
        else:
            mat[:, j] = np.where(np.isfinite(col), (col - mu) / sd, 0.0)
    return mat


# ── Per-feature accumulator ────────────────────────────────────────────────────

@dataclass
class _FeatureAcc:
    q:    int
    ZtZ:  np.ndarray = field(default=None)
    Ztx:  np.ndarray = field(default=None)
    Zty:  np.ndarray = field(default=None)
    xtx:  float      = 0.0
    xty:  float      = 0.0
    yty:  float      = 0.0
    n:    int        = 0

    def __post_init__(self):
        self.ZtZ = np.zeros((self.q, self.q))
        self.Ztx = np.zeros(self.q)
        self.Zty = np.zeros(self.q)

    def update(
        self,
        z_full: np.ndarray,   # (n, q)  -- ALL rows in this batch (Z never NaN)
        x: np.ndarray,        # (n,)    -- feature (may have NaN)
        y: np.ndarray,        # (n,)    -- temperature (no NaN)
        # Pre-computed for the full batch (fast correction approach):
        ZtZ_all: np.ndarray,  # (q, q)
        Zty_all: np.ndarray,  # (q,)
        yty_all: float,
    ) -> None:
        mask_valid   = ~np.isnan(x)
        mask_invalid = ~mask_valid
        n_valid = mask_valid.sum()
        if n_valid < 2:
            return

        # Subtract the NaN-row contribution from the full-batch accumulants.
        if mask_invalid.any():
            z_nan = z_full[mask_invalid]
            y_nan = y[mask_invalid]
            ZtZ_i = ZtZ_all - z_nan.T @ z_nan
            Zty_i = Zty_all - z_nan.T @ y_nan
            yty_i = yty_all - float(y_nan @ y_nan)
        else:
            ZtZ_i = ZtZ_all
            Zty_i = Zty_all
            yty_i = yty_all

        x_v = x[mask_valid]
        z_v = z_full[mask_valid]
        y_v = y[mask_valid]

        self.ZtZ += ZtZ_i
        self.Ztx += z_v.T @ x_v
        self.Zty += Zty_i
        self.xtx += float(x_v @ x_v)
        self.xty += float(x_v @ y_v)
        self.yty += yty_i
        self.n   += int(n_valid)


def _partial_r(acc: _FeatureAcc, q: int):
    """
    Return (partial_r, partial_p, n, extras) from accumulated sufficient statistics.

    extras is a dict with:
      effect_slope   -- OLS slope of M_Z·y on M_Z·x in natural units (y-units per x-unit)
      feature_mean   -- mean of x for valid rows (from intercept column of Ztx)
      feature_std    -- std  of x for valid rows
      target_mean    -- mean of y for valid rows (from intercept column of Zty)
      target_std     -- std  of y for valid rows

    Returns (None, None, acc.n, {}) when rank-deficient or n too small.
    """
    _empty = {}
    if acc.n < q + 2:
        return None, None, acc.n, _empty

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beta_x = np.linalg.lstsq(acc.ZtZ, acc.Ztx, rcond=None)[0]
            beta_y = np.linalg.lstsq(acc.ZtZ, acc.Zty, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None, acc.n, _empty

    xMx = acc.xtx - float(acc.Ztx @ beta_x)
    yMy = acc.yty - float(acc.Zty @ beta_y)
    xMy = acc.xty - float(acc.Ztx @ beta_y)

    denom = xMx * yMy
    if denom <= 0.0:
        return None, None, acc.n, _empty

    pr = float(np.clip(xMy / np.sqrt(denom), -1.0, 1.0))

    df = acc.n - q - 1
    if df < 1:
        return pr, None, acc.n, _empty

    t_stat = pr * np.sqrt(df / max(1.0 - pr * pr, 1e-14))
    p_val  = float(2.0 * stats.t.cdf(-abs(t_stat), df=df))

    # --- extra traceable quantities ---
    # slope: OLS coefficient of M_Z·y regressed on M_Z·x  (natural units)
    slope = (xMy / xMx) if abs(xMx) > 1e-14 else None

    # intercept column (col 0) of Z is all-ones, so Ztx[0] = sum(x_valid)
    x_mean = float(acc.Ztx[0]) / acc.n
    x_var  = max(0.0, acc.xtx / acc.n - x_mean ** 2)
    y_mean = float(acc.Zty[0]) / acc.n
    y_var  = max(0.0, acc.yty / acc.n - y_mean ** 2)

    extras = {
        "effect_slope": round(slope, 6)      if slope   is not None else None,
        "feature_mean": round(x_mean, 6),
        "feature_std":  round(x_var ** 0.5, 6),
        "target_mean":  round(y_mean, 6),
        "target_std":   round(y_var  ** 0.5, 6),
    }
    return pr, p_val, acc.n, extras


# ── Driver ─────────────────────────────────────────────────────────────────────

def _source_of(col: str) -> Optional[str]:
    if col in {"temperature", "aster_lst", "modis_lst", "ndvi"}:
        return "LST"
    if col.startswith("dhm_"):   return "DHM"
    if col.startswith("trees_"): return "Trees"
    if col.startswith("ua_"):    return "UA"
    if col.startswith("wis_"):   return "WIS"
    return "other"


def run(
    max_rows:        Optional[int],
    batch_size:      int,
    include_spatial: bool,
    greening_only:   bool = True,
    dataset:         Optional[str] = None,
) -> dict:
    """
    greening_only=True  (default)
        Evaluates only greening features (trees_*, ndvi, vegetation UA codes).
        Computes two partial correlations per feature:
          partial_r     — controlling for temporal confounders only
          partial_r_adj — controlling for temporal + morphology confounders
                          (dhm_* + non-vegetation ua_*)
        Also reports confounding_fraction = (partial_r - partial_r_adj) / partial_r.

    greening_only=False
        Legacy mode: evaluates all floating-point features, temporal confounders only.
    """
    if dataset:
        from utils import (
            load_parquet_sample, dataset_stats as _ds_stats,
            compute_structural_ndvi, run_timestamp, version_meta,
        )
        data_dir = _SRC / "test_data"
        path = data_dir / f"{dataset}.parquet"
        df_full = load_parquet_sample(path, max_rows, seed=42)
        df_full = compute_structural_ndvi(df_full)   # adds ndvi_struct_mean/_std
        _ds_meta   = _ds_stats(df_full)
        _run_ts    = run_timestamp()
        _ver_meta  = version_meta()
        batch_iter   = (df_full.iloc[i:i+batch_size]
                        for i in range(0, len(df_full), batch_size))
        source_label = f"dataset:{dataset} (n={len(df_full):,})"
    else:
        _ds_meta, _run_ts, _ver_meta = {}, None, {}
        reg          = build_registry()
        cfg          = all_rows(batch_size=batch_size)
        batch_iter   = cfg.stream(reg, batch_size=batch_size, max_rows=max_rows)
        source_label = "full stream"

    _NON_FEATURE_BASE = frozenset({
        "temperature", "aster_lst", "modis_lst",
        # ndvi is excluded: it is computed from the same satellite image as LST,
        # making it a co-measurement rather than an independent greening indicator.
        "ndvi",
        "longitude", "latitude", "image_id", "timestamp",
        "partition_key", "tile_id", "year", "month_of_year",
        "day_of_month", "day_of_year", "hour_of_day",
    })
    _NON_FEATURE = _NON_FEATURE_BASE

    q_base = 8 if include_spatial else 6

    # Accumulator dicts — initialised on first batch
    accs_base: Optional[Dict[str, _FeatureAcc]] = None   # Z = temporal
    accs_adj:  Optional[Dict[str, _FeatureAcc]] = None   # Z = temporal + morphology
    greening_cols: List[str] = []
    morph_cols:    List[str] = []
    q_adj = q_base  # updated once morph_cols are known

    total_rows = 0
    t0 = time.perf_counter()

    bar = tqdm(desc=f"partial_corr ({source_label})", unit="rows", dynamic_ncols=True)
    try:
        for batch_df in batch_iter:
            if batch_df is None or len(batch_df) == 0:
                continue

            y = batch_df["temperature"].to_numpy(dtype=np.float64)
            Z_base = _build_Z(batch_df, include_spatial=include_spatial)

            if accs_base is None:
                all_float = [c for c in batch_df.columns
                             if c not in _NON_FEATURE
                             and np.issubdtype(batch_df[c].dtype, np.floating)]
                if greening_only:
                    # Include integer columns (e.g. trees_* counts) as greening features
                    all_numeric = [c for c in batch_df.columns
                                   if c not in _NON_FEATURE
                                   and np.issubdtype(batch_df[c].dtype, np.number)]
                    greening_cols = [c for c in all_numeric if _is_greening(c)]
                    morph_cols    = [c for c in all_float if _is_morphology(c)]
                    q_adj = q_base + len(morph_cols)
                    accs_base = {c: _FeatureAcc(q=q_base) for c in greening_cols}
                    accs_adj  = {c: _FeatureAcc(q=q_adj)  for c in greening_cols}
                    tqdm.write(
                        f"greening features: {len(greening_cols)}  "
                        f"morphology confounders: {len(morph_cols)}"
                    )
                else:
                    greening_cols = all_float
                    accs_base = {c: _FeatureAcc(q=q_base) for c in greening_cols}
                    tqdm.write(f"features discovered: {len(greening_cols)}")

            ZtZ_base = Z_base.T @ Z_base
            Zty_base = Z_base.T @ y
            yty      = float(y @ y)

            X_green = batch_df[greening_cols].to_numpy(dtype=np.float64)

            if greening_only:
                morph_mat = _build_morph_Z(batch_df, morph_cols)
                Z_adj     = np.hstack([Z_base, morph_mat])
                ZtZ_adj   = Z_adj.T @ Z_adj
                Zty_adj   = Z_adj.T @ y

            for i, col in enumerate(greening_cols):
                x = X_green[:, i]
                accs_base[col].update(Z_base, x, y,
                                      ZtZ_all=ZtZ_base, Zty_all=Zty_base, yty_all=yty)
                if greening_only:
                    accs_adj[col].update(Z_adj, x, y,
                                         ZtZ_all=ZtZ_adj, Zty_all=Zty_adj, yty_all=yty)

            total_rows += len(batch_df)
            bar.update(len(batch_df))
    finally:
        bar.close()

    elapsed = time.perf_counter() - t0
    tqdm.write(f"stream done: {total_rows:,} rows in {elapsed/60:.1f} min")

    if accs_base is None:
        print("No data streamed.", file=sys.stderr)
        return {}

    results = []
    skipped = []
    for col in greening_cols:
        r_base, p_base, n, extras_base = _partial_r(accs_base[col], q_base)

        if greening_only:
            r_adj, p_adj, _, extras_adj = _partial_r(accs_adj[col], q_adj)
            cf = None
            if r_base is not None and r_adj is not None and abs(r_base) > 1e-8:
                cf = round(float((r_base - r_adj) / r_base), 4)

            if r_base is None and r_adj is None:
                skipped.append({"feature": col, "source": _source_of(col),
                                "n_valid": n, "reason": "insufficient_data"})
                continue

            results.append({
                "feature":              col,
                "source":               _source_of(col),
                "n_valid":              n,
                # base (temporal controls only)
                "partial_r":            round(r_base, 6) if r_base is not None else None,
                "partial_p":            round(p_base, 6) if p_base is not None else None,
                "effect_slope":         extras_base.get("effect_slope"),
                # morphology-adjusted
                "partial_r_adj":        round(r_adj,  6) if r_adj  is not None else None,
                "partial_p_adj":        round(p_adj,  6) if p_adj  is not None else None,
                "effect_slope_adj":     extras_adj.get("effect_slope"),
                "confounding_fraction": cf,
                # feature / target moments (for unit conversion)
                "feature_mean":         extras_base.get("feature_mean"),
                "feature_std":          extras_base.get("feature_std"),
                "target_mean_K":        extras_base.get("target_mean"),
                "target_std_K":         extras_base.get("target_std"),
            })
        else:
            if r_base is None:
                skipped.append({"feature": col, "source": _source_of(col),
                                "n_valid": n, "reason": "insufficient_data"})
                continue
            results.append({
                "feature":      col,
                "source":       _source_of(col),
                "n_valid":      n,
                "partial_r":    round(r_base, 6),
                "partial_p":    round(p_base, 6) if p_base is not None else None,
                "effect_slope": extras_base.get("effect_slope"),
                "feature_mean": extras_base.get("feature_mean"),
                "feature_std":  extras_base.get("feature_std"),
            })

    sort_key = "partial_r_adj" if greening_only else "partial_r"
    results.sort(key=lambda r: abs(r.get(sort_key) or r.get("partial_r") or 0), reverse=True)

    base_confounders = ["intercept", "year_norm", "month_sin", "month_cos",
                        "hour_sin", "hour_cos"]
    if include_spatial:
        base_confounders += ["lat_norm", "lon_norm"]

    # Classify greening features by physical type for the meta summary
    def _green_type(col):
        if col.startswith("trees_"):         return "trees"
        if col.startswith("ndvi_struct"):    return "ndvi_structural"
        if "water" in col or "wetland" in col.lower(): return "ua_water_wetlands"
        if col.startswith("ua_"):            return "ua_vegetation"
        return "other"

    green_by_type: Dict[str, List[str]] = {}
    for col in greening_cols:
        green_by_type.setdefault(_green_type(col), []).append(col)

    output = {
        "_meta": {
            "algorithm":              "FWL-sufficient-statistics",
            "mode":                   "greening_only" if greening_only else "all_features",
            "run_timestamp":          _run_ts,
            "versions":               _ver_meta,
            "base_confounders":       base_confounders,
            "morphology_confounders": morph_cols if greening_only else [],
            "n_morphology_confounders": len(morph_cols) if greening_only else 0,
            "target":                 "temperature",
            "source":                 source_label,
            "total_rows_streamed":    total_rows,
            "elapsed_s":              round(elapsed, 1),
            # greening feature inventory
            "n_greening_features":    len(greening_cols),
            "greening_by_type":       {k: len(v) for k, v in green_by_type.items()},
            "greening_cols_by_type":  green_by_type,
            # dataset provenance
            "dataset_stats":          _ds_meta,
            # interpretation note for effect_slope
            "effect_slope_note": (
                "OLS slope of (M_Z · y) on (M_Z · x): "
                "y-units (Kelvin) per 1-unit increase in feature x, "
                "after partialling out confounders Z. "
                "Multiply by feature_std to get the per-1-SD effect in Kelvin."
            ),
        },
        "results":          results,
        "skipped_features": skipped,
    }

    return output


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--max-rows",            type=int,  default=None,
                    help="cap total rows / sample size (default: all)")
    ap.add_argument("--batch-size",          type=int,  default=50_000)
    ap.add_argument("--spatial-confounders", action="store_true",
                    help="also partial out latitude and longitude")
    ap.add_argument("--all-features",        action="store_true",
                    help="evaluate all features (legacy mode); default evaluates "
                         "only greening features with morphology adjustment")
    ap.add_argument("--dataset",             type=str,  default=None,
                    help="run on one specific pre-built dataset parquet "
                         "(default: all parquets in test_data/)")
    ap.add_argument("--full-stream",         action="store_true",
                    help="stream all source data instead of using pre-built parquets")
    ap.add_argument("--out",                 type=Path, default=_OUT_DEFAULT)
    args = ap.parse_args()

    if args.full_stream and args.dataset:
        ap.error("--full-stream and --dataset are mutually exclusive")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _call(**kw):
        return run(max_rows=args.max_rows, batch_size=args.batch_size,
                   include_spatial=args.spatial_confounders,
                   greening_only=not args.all_features, **kw)

    if args.full_stream:
        output = _call(dataset=None)
    elif args.dataset:
        output = _call(dataset=args.dataset)
    else:
        from utils import list_datasets
        datasets = list_datasets()
        if not datasets:
            print(f"no parquet files in {_SRC / 'test_data'} -- "
                  "run build_test_datasets.py or pass --full-stream",
                  file=sys.stderr)
            return 1
        output = {}
        for name, _ in datasets:
            print(f"\n--- {name} ---")
            output[name] = _call(dataset=name)

    args.out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
