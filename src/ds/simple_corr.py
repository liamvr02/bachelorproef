"""
simple_corr.py
==============
Pearson r, Spearman rho, mutual information, and OLS slope for every greening
feature vs temperature.  No confounder removal — the simplest interpretable
association.

For each greening feature x and target y (temperature):
  pearson_r, pearson_p    linear correlation + two-sided p-value
  spearman_r, spearman_p  rank-based correlation + p-value
  mi                      mutual information (k-NN, sklearn)
  slope_k_per_unit        Kelvin per 1-unit increase in x  (= r * sigma_y / sigma_x)
  slope_k_per_std         Kelvin per 1-SD increase in x    (= r * sigma_y)

Greening features: trees_*, ndvi_struct_*, ua_*vegetation*, ua_*water*, ua_*wetland*.

Usage
-----
    python src/ds/simple_corr.py [--dataset NAME] [--max-rows N] [--out PATH]
    python src/ds/simple_corr.py          # runs on all parquets in test_data/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import warnings

import numpy as np
from scipy import stats

_DS_DIR      = Path(__file__).parent
_SRC         = _DS_DIR.parent
_OUT_DEFAULT = _SRC / "ds_reports" / "simple_corr.json"

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_DS_DIR))

# ── Feature classification (mirrors partial_corr.py) ──────────────────────────

_UA_VEG_KW = frozenset({
    "14100", "green_urban", "greenurban",
    "31000", "forest",
    "32000", "herbaceous",
    "40000", "wetland",
    "21000", "22000", "23000", "arable", "pasture", "agricultural",
    "vegetation",
})

_NON_FEATURE = frozenset({
    "temperature", "aster_lst", "modis_lst", "ndvi",
    "longitude", "latitude", "image_id", "timestamp",
    "partition_key", "tile_id", "year", "month_of_year",
    "day_of_month", "day_of_year", "hour_of_day",
})


def _is_greening(col: str) -> bool:
    if col.startswith("trees_"):
        return True
    if col.startswith("ndvi_struct"):
        return True
    if col.startswith("ua_"):
        return any(kw in col.lower() for kw in _UA_VEG_KW)
    return False


def _source_of(col: str) -> str:
    if col.startswith("trees_"):
        return "Trees"
    if col.startswith("ua_"):
        return "UA"
    if col.startswith("ndvi"):
        return "LST"
    return "other"


# ── Per-feature computation ────────────────────────────────────────────────────

def _compute_feature(
    x: np.ndarray,
    y: np.ndarray,
    mi_x: Optional[np.ndarray] = None,
    mi_y: Optional[np.ndarray] = None,
) -> dict:
    """
    Pearson r, Spearman rho, MI and derived quantities for one feature.
    x and y must be finite (no NaN).  mi_x/mi_y are the (possibly subsampled)
    arrays used for MI estimation — may be None to skip MI.
    """
    n = len(x)
    if n < 5:
        return {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)

    if not np.isfinite(pr):
        return {}

    x_std = float(np.std(x, ddof=1))
    y_std = float(np.std(y, ddof=1))
    slope_unit = float(pr * y_std / x_std) if x_std > 1e-10 else None
    slope_std  = float(pr * y_std)

    mi = None
    if mi_x is not None and mi_y is not None and len(mi_x) >= 5:
        try:
            from sklearn.feature_selection import mutual_info_regression
            mi_arr = mutual_info_regression(
                mi_x.reshape(-1, 1), mi_y,
                n_neighbors=5, random_state=42,
            )
            mi = float(mi_arr[0])
        except Exception:
            pass

    return {
        "n_valid":           n,
        "pearson_r":         round(float(pr), 6),
        "pearson_p":         round(float(pp), 6),
        "spearman_r":        round(float(sr), 6),
        "spearman_p":        round(float(sp), 6),
        "mi":                round(mi, 6) if mi is not None else None,
        "slope_k_per_unit":  round(slope_unit, 6) if slope_unit is not None else None,
        "slope_k_per_std":   round(slope_std, 6),
        "feature_mean":      round(float(np.mean(x)), 6),
        "feature_std":       round(float(x_std), 6),
        "target_mean":       round(float(np.mean(y)), 6),
        "target_std":        round(float(y_std), 6),
    }


# ── Driver ─────────────────────────────────────────────────────────────────────

def run(
    dataset:   str,
    max_rows:  Optional[int] = None,
    mi_sample: int = 50_000,
    seed:      int = 42,
) -> dict:
    from utils import (
        load_parquet_sample, dataset_stats, compute_structural_ndvi,
        run_timestamp, version_meta,
    )

    path = _SRC / "test_data" / f"{dataset}.parquet"
    df   = load_parquet_sample(path, max_rows, seed=seed)
    df   = compute_structural_ndvi(df)

    ds_meta = dataset_stats(df)
    _run_ts = run_timestamp()
    _ver    = version_meta()

    temp_col = next(
        (c for c in ("temperature", "aster_lst", "modis_lst") if c in df.columns),
        None,
    )
    if temp_col is None:
        return {"_meta": {"error": "no temperature column"}, "results": []}

    y_all = df[temp_col].to_numpy(dtype=np.float64)

    all_numeric: List[str] = [
        c for c in df.columns
        if c not in _NON_FEATURE and np.issubdtype(df[c].dtype, np.number)
    ]
    greening_cols = [c for c in all_numeric if _is_greening(c)]

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    results = []

    for col in greening_cols:
        x_all = df[col].to_numpy(dtype=np.float64)
        mask  = np.isfinite(x_all) & np.isfinite(y_all)
        xv, yv = x_all[mask], y_all[mask]
        if len(xv) < 5:
            continue

        if len(xv) > mi_sample:
            idx     = rng.choice(len(xv), mi_sample, replace=False)
            mi_x, mi_y = xv[idx], yv[idx]
        else:
            mi_x, mi_y = xv, yv

        res = _compute_feature(xv, yv, mi_x, mi_y)
        if not res:
            continue
        res["feature"] = col
        res["source"]  = _source_of(col)
        results.append(res)

    elapsed = time.perf_counter() - t0
    results.sort(key=lambda r: abs(r.get("pearson_r") or 0), reverse=True)

    return {
        "_meta": {
            "algorithm":      "pearson-spearman-mi",
            "run_timestamp":  _run_ts,
            "versions":       _ver,
            "target":         temp_col,
            "dataset":        dataset,
            "n_greening":     len(greening_cols),
            "greening_cols":  greening_cols,
            "mi_sample":      mi_sample,
            "elapsed_s":      round(elapsed, 2),
            "dataset_stats":  ds_meta,
            "slope_note": (
                "slope_k_per_unit = r * (sigma_y / sigma_x): "
                "expected Kelvin change per 1-unit increase in x. "
                "slope_k_per_std = r * sigma_y: "
                "expected Kelvin change per 1-SD increase in x."
            ),
        },
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset",   type=str,  default=None,
                    help="single dataset name (default: all parquets in test_data/)")
    ap.add_argument("--max-rows",  type=int,  default=None,
                    help="row cap (default: all)")
    ap.add_argument("--mi-sample", type=int,  default=50_000,
                    help="rows to subsample for MI estimation (default 50 000)")
    ap.add_argument("--seed",      type=int,  default=42)
    ap.add_argument("--out",       type=Path, default=_OUT_DEFAULT)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _call(name: str) -> dict:
        return run(dataset=name, max_rows=args.max_rows,
                   mi_sample=args.mi_sample, seed=args.seed)

    if args.dataset:
        output = {args.dataset: _call(args.dataset)}
    else:
        from utils import list_datasets
        datasets = list_datasets()
        if not datasets:
            print(f"no parquet files in {_SRC / 'test_data'}", file=sys.stderr)
            return 1
        output = {}
        for name, _ in datasets:
            print(f"  --- {name} ---")
            output[name] = _call(name)

    args.out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
