"""
morans_i.py  (streaming / online)
======================================
Streaming Moran's I on OLS residuals — tests whether spatial
autocorrelation in temperature is absorbed by the feature set.

Algorithm  --  single streaming pass
-------------------------------------
Pass: stream all batches with the full registry.

  Accumulate globally:
    XtX  (p x p)  -- X'X
    Xty  (p,)     -- X'y
    yty  scalar   -- y'y
    sum_y scalar  -- sum(y)
    n    int

  Accumulate per H3-tile (tile_id, resolution 9):
    sum_X_tile  (p,)  -- sum of feature rows in this tile
    sum_y_tile  scalar
    n_tile      int

After the pass:
  1. Solve global OLS:  beta = lstsq(XtX, Xty)
  2. Per tile:  mean_X = sum_X / n_tile,  mean_y = sum_y / n_tile
               e_tile  = mean_y - mean_X @ beta       (tile-level residual)
  3. Build H3 ring-1 neighbor weight matrix W (binary, row-standardised)
  4. Compute Moran's I = (n / sum_W) * z'Wz / z'z
                         where z = e_tile - mean(e_tile)
  5. Permutation p-value (n_perm=999)

Two models are compared:
  "null"  -- feature matrix X = temporal confounders only (no spatial features)
  "full"  -- feature matrix X = all registry features + temporal confounders

Both use the same streaming pass (features already in batch; null model
uses only the confounder columns).

Usage
-----
    python src/ds/morans_i.py [--max-rows N] [--batch-size N]
                              [--n-perm N] [--min-tile-n N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h3
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm

_DS_DIR      = Path(__file__).parent
_SRC         = _DS_DIR.parent
_OUT_DEFAULT = _SRC / "ds_reports" / "morans_i.json"

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_DS_DIR))

from stream_configs.presets import all_rows
from stream_configs.registry import build_registry

_YEAR_MU  = 2012.5
_YEAR_SIG = 6.25

# Reuse the same greening / morphology classification as partial_corr.
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
        return any(kw in col.lower() for kw in _UA_VEG_KW)
    return False


def _is_morphology(col: str) -> bool:
    if col.startswith("dhm_"):
        return True
    if col.startswith("ua_") and not _is_greening(col):
        return True
    return False


# ── Confounder builder ────────────────────────────────────────────────────────

def _build_confounders(df) -> np.ndarray:
    """Return (n, 6) confounder matrix: intercept, year_norm, month/hour cyclic."""
    n  = len(df)
    yr = (df["year"].to_numpy(dtype=np.float64) - _YEAR_MU) / _YEAR_SIG
    mo = df["month_of_year"].to_numpy(dtype=np.float64)
    hr = df["hour_of_day"].to_numpy(dtype=np.float64)
    return np.column_stack([
        np.ones(n),
        yr,
        np.sin(2.0 * np.pi * mo / 12.0),
        np.cos(2.0 * np.pi * mo / 12.0),
        np.sin(2.0 * np.pi * hr / 24.0),
        np.cos(2.0 * np.pi * hr / 24.0),
    ])


# ── Streaming accumulator ─────────────────────────────────────────────────────

class _OLSAccumulator:
    """Accumulate X'X, X'y, y'y, sum_y, n  plus per-tile sums."""

    def __init__(self, p: int) -> None:
        self.p    = p
        self.XtX  = np.zeros((p, p))
        self.Xty  = np.zeros(p)
        self.yty  = 0.0
        self.sum_y = 0.0
        self.n    = 0
        self.tiles: Dict[str, List] = defaultdict(lambda: [np.zeros(p), 0.0, 0])

    def update(
        self,
        X:      np.ndarray,   # (n, p) -- may have NaN
        y:      np.ndarray,   # (n,)   -- no NaN
        tiles:  np.ndarray,   # (n,)   -- H3 tile strings
    ) -> None:
        mask  = ~np.isnan(X).any(axis=1)
        # tile_id can be None/NaN in the parquet; np.unique can't sort mixed types.
        mask &= np.fromiter((t is not None for t in tiles), dtype=bool, count=len(tiles))
        X_cc  = X[mask]
        y_cc  = y[mask]
        t_cc  = tiles[mask]

        if len(X_cc) == 0:
            return

        self.XtX   += X_cc.T @ X_cc
        self.Xty   += X_cc.T @ y_cc
        self.yty   += float(y_cc @ y_cc)
        self.sum_y += float(y_cc.sum())
        self.n     += len(y_cc)

        for tile in np.unique(t_cc):
            idx = t_cc == tile
            self.tiles[tile][0] += X_cc[idx].sum(axis=0)
            self.tiles[tile][1] += float(y_cc[idx].sum())
            self.tiles[tile][2] += int(idx.sum())

    def solve(self) -> np.ndarray:
        return np.linalg.lstsq(self.XtX, self.Xty, rcond=None)[0]

    def tile_residuals(self, beta: np.ndarray, min_n: int) -> Dict[str, float]:
        """Return {tile_id: tile_mean_residual} for tiles with >= min_n rows."""
        out = {}
        for tile, (sum_x, sum_y, n) in self.tiles.items():
            if n < min_n:
                continue
            mean_x = sum_x / n
            mean_y = sum_y / n
            out[tile] = mean_y - float(mean_x @ beta)
        return out


# ── Moran's I ─────────────────────────────────────────────────────────────────

def _morans_i(
    tile_ids:  List[str],
    residuals: np.ndarray,
    n_perm:    int = 999,
    rng:       np.random.Generator = None,
) -> Tuple[float, float, float]:
    """
    Compute Moran's I for the tile residuals, using H3 ring-1 binary weights
    (row-standardised).

    Returns (I, E[I], p_value) where p_value is from a permutation test.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    tile_set = set(tile_ids)
    idx_of   = {t: i for i, t in enumerate(tile_ids)}
    k        = len(tile_ids)

    rows_idx, cols_idx = [], []
    for i, tile in enumerate(tile_ids):
        for nbr in h3.grid_disk(tile, 1):
            if nbr != tile and nbr in tile_set:
                j = idx_of[nbr]
                rows_idx.append(i)
                cols_idx.append(j)

    if not rows_idx:
        return 0.0, -1.0 / (k - 1), 1.0

    W = sparse.csr_matrix(
        (np.ones(len(rows_idx)), (rows_idx, cols_idx)),
        shape=(k, k),
    )
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv  = sparse.diags(1.0 / row_sums)
    W_norm = D_inv @ W
    W_sum  = float(W_norm.sum())

    z = residuals - residuals.mean()

    def _I(v: np.ndarray) -> float:
        return float(k / W_sum * (v @ W_norm @ v) / max(v @ v, 1e-14))

    I_obs = _I(z)
    E_I   = -1.0 / (k - 1)

    perm_Is = np.array([_I(rng.permutation(z)) for _ in range(n_perm)])
    p_val   = float((np.sum(np.abs(perm_Is) >= abs(I_obs)) + 1) / (n_perm + 1))

    return I_obs, E_I, p_val


# ── Driver ─────────────────────────────────────────────────────────────────────

def run(
    max_rows:   Optional[int],
    batch_size: int,
    n_perm:     int,
    min_tile_n: int,
    dataset:    Optional[str] = None,
) -> dict:
    if dataset:
        from utils import (
            load_parquet_sample, dataset_stats as _ds_stats,
            compute_structural_ndvi, run_timestamp, version_meta,
        )
        data_dir = _SRC / "test_data"
        path = data_dir / f"{dataset}.parquet"
        df_full = load_parquet_sample(path, max_rows, seed=42)
        df_full = compute_structural_ndvi(df_full)   # adds ndvi_struct_mean/_std
        _ds_meta  = _ds_stats(df_full)
        _run_ts   = run_timestamp()
        _ver_meta = version_meta()
        batch_iter   = (df_full.iloc[i:i+batch_size]
                        for i in range(0, len(df_full), batch_size))
        source_label = f"dataset:{dataset} (n={len(df_full):,})"
    else:
        _ds_meta, _run_ts, _ver_meta = {}, None, {}
        reg          = build_registry()
        cfg          = all_rows(batch_size=batch_size)
        batch_iter   = cfg.stream(reg, batch_size=batch_size, max_rows=max_rows)
        source_label = "full stream"

    _NON_FEATURE = frozenset({
        "temperature", "aster_lst", "modis_lst", "ndvi",
        "longitude", "latitude", "image_id", "timestamp",
        "partition_key", "tile_id", "year", "month_of_year",
        "day_of_month", "day_of_year", "hour_of_day",
    })

    acc_full: Optional[_OLSAccumulator] = None
    acc_null: Optional[_OLSAccumulator] = None
    acc_morph: Optional[_OLSAccumulator] = None   # morphology features only
    feature_cols: List[str] = []
    morph_cols:   List[str] = []
    greening_cols: List[str] = []

    total_rows = 0
    t0 = time.perf_counter()

    bar = tqdm(desc=f"morans_i ({source_label})", unit="rows", dynamic_ncols=True)
    try:
        for batch_df in batch_iter:
            if batch_df is None or len(batch_df) == 0:
                continue

            y      = batch_df["temperature"].to_numpy(dtype=np.float64)
            tiles  = batch_df["tile_id"].to_numpy()
            Z_conf = _build_confounders(batch_df)

            if acc_full is None:
                feature_cols = [c for c in batch_df.columns
                                if c not in _NON_FEATURE
                                and np.issubdtype(batch_df[c].dtype, np.floating)]
                # Include integer columns (e.g. trees_* counts) in greening
                all_numeric = [c for c in batch_df.columns
                               if c not in _NON_FEATURE
                               and np.issubdtype(batch_df[c].dtype, np.number)]
                greening_cols = [c for c in all_numeric if _is_greening(c)]
                morph_cols    = [c for c in feature_cols if _is_morphology(c)]
                # Extend feature_cols with integer greening cols for full model
                extra_green = [c for c in greening_cols if c not in feature_cols]
                feature_cols = feature_cols + extra_green
                p_full  = len(feature_cols) + Z_conf.shape[1]
                p_null  = Z_conf.shape[1]
                p_morph = len(morph_cols)  + Z_conf.shape[1]
                acc_full  = _OLSAccumulator(p_full)
                acc_null  = _OLSAccumulator(p_null)
                acc_morph = _OLSAccumulator(p_morph)
                tqdm.write(
                    f"features: {len(feature_cols)}  "
                    f"(greening: {len(greening_cols)}, morphology: {len(morph_cols)})  "
                    f"confounders: {Z_conf.shape[1]}"
                )

            X_feat  = batch_df[feature_cols].to_numpy(dtype=np.float64)
            X_morph = batch_df[morph_cols].to_numpy(dtype=np.float64) if morph_cols else np.empty((len(batch_df), 0))

            X_full  = np.hstack([X_feat,  Z_conf])
            X_null  = Z_conf
            X_monly = np.hstack([X_morph, Z_conf])

            acc_full.update(X_full,  y, tiles)
            acc_null.update(X_null,  y, tiles)
            acc_morph.update(X_monly, y, tiles)

            total_rows += len(batch_df)
            bar.update(len(batch_df))
    finally:
        bar.close()

    elapsed = time.perf_counter() - t0
    tqdm.write(f"stream done: {total_rows:,} rows in {elapsed/60:.1f} min")

    rng = np.random.default_rng(42)

    def _model_stats(acc: _OLSAccumulator, label: str) -> dict:
        if acc.n == 0:
            return {"label": label, "error": "no data"}

        beta   = acc.solve()
        e_dict = acc.tile_residuals(beta, min_n=min_tile_n)

        if len(e_dict) < 4:
            return {
                "label": label,
                "n_tiles": len(e_dict),
                "error": f"fewer than 4 tiles with >= {min_tile_n} rows",
            }

        tile_ids   = list(e_dict.keys())
        residuals  = np.array([e_dict[t] for t in tile_ids])

        ybar     = acc.sum_y / acc.n
        ss_tot   = acc.yty - acc.n * ybar ** 2
        ss_res   = acc.yty - float(acc.Xty @ beta)
        r2_global = float(np.clip(1.0 - ss_res / max(ss_tot, 1e-14), 0.0, 1.0))

        tqdm.write(f"[{label}] {len(tile_ids)} tiles, computing Moran's I ...")
        I_obs, E_I, p_val = _morans_i(tile_ids, residuals, n_perm=n_perm, rng=rng)
        tqdm.write(f"  Moran's I = {I_obs:.4f}  E[I] = {E_I:.4f}  p = {p_val:.4f}")

        I_drop_vs_null = None   # filled in by caller after all models are computed
        return {
            "label":        label,
            "n_obs":        acc.n,
            "n_tiles":      len(tile_ids),
            "n_features":   acc.p,          # total predictor columns (features + confounders)
            "global_r2":    round(r2_global, 4),
            "morans_I":     round(I_obs, 6),
            "morans_E_I":   round(E_I, 6),
            "morans_p":     round(p_val, 6),
            "n_perm_used":  n_perm,
            "interpretation": (
                "spatial autocorrelation REMAINS in residuals -- features do not "
                "fully explain the spatial LST structure"
                if I_obs > E_I + 0.02
                else "residuals are spatially random -- features capture spatial "
                "LST variation well"
            ),
        }

    null_stats  = _model_stats(acc_null,  "null")
    morph_stats = _model_stats(acc_morph, "morphology")
    full_stats  = _model_stats(acc_full,  "full")

    # ΔI_greening = I(morphology) − I(full):
    #   positive  → greening reduces tile-level spatial clustering beyond morphology
    #   near-zero → greening effect is sub-tile (local); confirmed by partial r
    #   negative  → full model has more residual clustering than morphology-only
    delta_i_greening = None
    delta_i_morph    = None   # null → morphology drop (morphology's spatial contribution)
    if "morans_I" in null_stats  and "morans_I" in morph_stats:
        delta_i_morph = round(
            float(null_stats["morans_I"]) - float(morph_stats["morans_I"]), 6
        )
    if "morans_I" in morph_stats and "morans_I" in full_stats:
        delta_i_greening = round(
            float(morph_stats["morans_I"]) - float(full_stats["morans_I"]), 6
        )

    def _di_interpretation(di):
        if di is None:   return None
        if di >  0.02:   return "positive: greening reduces tile-level spatial clustering beyond morphology (neighbourhood-scale signal detected)"
        if di > -0.02:   return "near-zero: greening operates below ~200m tile resolution; local effect should be confirmed by partial r"
        if di > -0.05:   return "mildly negative: greening does not reduce tile-level clustering; effect is sub-neighbourhood scale"
        return               "strongly negative: greening increases spatial clustering; check for spatial multicollinearity in greening feature set"

    output = {
        "_meta": {
            "algorithm":           "OLS-tile-residuals + Moran-I",
            "tile_level":          "H3 resolution 9 (~30m LST pixel -> tile grouping)",
            "spatial_weights":     "H3 ring-1 binary (row-standardised)",
            "permutation_test":    f"Phipson-Smyth ({n_perm} permutations, seed=42)",
            "min_tile_n":          min_tile_n,
            "run_timestamp":       _run_ts,
            "versions":            _ver_meta,
            "source":              source_label,
            "total_rows_streamed": total_rows,
            "elapsed_s":           round(elapsed, 1),
            # feature inventory
            "n_features_total":    len(feature_cols),
            "n_greening":          len(greening_cols),
            "n_morphology":        len(morph_cols),
            "n_confounders":       6,
            "greening_cols":       greening_cols,
            "morph_cols":          morph_cols,
            # dataset provenance
            "dataset_stats":       _ds_meta,
        },
        "null_model":        null_stats,
        "morphology_model":  morph_stats,
        "full_model":        full_stats,
        # headline scalars for conclusion
        "delta_i_morph":    delta_i_morph,
        "delta_i_greening": delta_i_greening,
        "delta_i_greening_interpretation": _di_interpretation(delta_i_greening),
    }

    return output


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--max-rows",    type=int,  default=None)
    ap.add_argument("--batch-size",  type=int,  default=50_000)
    ap.add_argument("--n-perm",      type=int,  default=999,
                    help="permutation test iterations (default: 999)")
    ap.add_argument("--min-tile-n",  type=int,  default=5,
                    help="min rows per tile to include in Moran's I (default: 5)")
    ap.add_argument("--dataset",     type=str,  default=None,
                    help="run on one specific pre-built dataset parquet "
                         "(default: all parquets in test_data/)")
    ap.add_argument("--full-stream", action="store_true",
                    help="stream all source data instead of using pre-built parquets")
    ap.add_argument("--out",         type=Path, default=_OUT_DEFAULT)
    args = ap.parse_args()

    if args.full_stream and args.dataset:
        ap.error("--full-stream and --dataset are mutually exclusive")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _call(**kw):
        return run(max_rows=args.max_rows, batch_size=args.batch_size,
                   n_perm=args.n_perm, min_tile_n=args.min_tile_n, **kw)

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
