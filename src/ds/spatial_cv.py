"""
spatial_cv.py  (streaming / online)
========================================
Spatial leave-one-block-out cross-validation via sufficient statistics.

Algorithm  --  single streaming pass
-------------------------------------
Stream all batches with the full registry.  For each batch row, compute its
spatial block assignment (H3 r8 by default) from longitude/latitude.

Accumulate globally AND per block:
  XtX    (p x p)      -- X'X
  Xty    (p,)         -- X'y
  yty    scalar       -- y'y
  sum_y  scalar       -- sum y
  n      int          -- row count
  (raw test rows up to --max-test-rows per block for evaluation)

Leave-one-block-out evaluation (no additional streaming pass):
  For held-out block b:
    XtX_train = XtX - XtX_b
    Xty_train = Xty - Xty_b
    beta = lstsq(XtX_train, Xty_train)   (+ ridge when near-singular)
    Evaluate on stored raw rows: MAE, RMSE, R2

Also computes:
  - Random K-fold CV for comparison (same beta, random block shuffling)
  - Per-block performance spread

Block column choices (--block-col):
  h3r8      H3 resolution 8  (~860 m,   ~100-300 blocks in Ghent)  [default]
  h3r7      H3 resolution 7  (~2.3 km,  ~20-40 blocks)
  rect1km   rectangular 1 km Lambert grid
  rect2km   rectangular 2 km Lambert grid

Usage
-----
    python src/ds/spatial_cv.py [--max-rows N] [--batch-size N]
                                [--block-col h3r8|h3r7|rect1km|rect2km]
                                [--max-test-rows N] [--ridge LAMBDA]
                                [--out PATH]
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
from pyproj import Transformer
from tqdm.auto import tqdm

_DS_DIR      = Path(__file__).parent
_SRC         = _DS_DIR.parent
_OUT_DEFAULT = _SRC / "ds_reports" / "spatial_cv.json"

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_DS_DIR))

from stream_configs.presets import all_rows
from stream_configs.registry import build_registry

_wgs84_to_lambert = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)

_YEAR_MU  = 2012.5
_YEAR_SIG = 6.25


def _compute_blocks(
    lons:      np.ndarray,
    lats:      np.ndarray,
    block_col: str,
) -> np.ndarray:
    """Compute block ID string for each row from lon/lat."""
    if block_col == "h3r8":
        return np.array([
            h3.latlng_to_cell(float(la), float(lo), 8)
            for lo, la in zip(lons, lats)
        ])
    if block_col == "h3r7":
        return np.array([
            h3.latlng_to_cell(float(la), float(lo), 7)
            for lo, la in zip(lons, lats)
        ])
    if block_col in ("rect1km", "rect2km"):
        size_m = 1000.0 if block_col == "rect1km" else 2000.0
        lx, ly = _wgs84_to_lambert.transform(lons, lats)
        ix = np.floor(lx / size_m).astype(int)
        iy = np.floor(ly / size_m).astype(int)
        return np.array([f"{x}_{y}" for x, y in zip(ix, iy)])
    raise ValueError(f"Unknown block_col: {block_col!r}")


def _build_confounders(df) -> np.ndarray:
    """(n, 6) temporal confounders appended to feature matrix."""
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


# ── Sufficient-statistics accumulator ────────────────────────────────────────

class _BlockAccumulator:
    """
    Accumulates OLS sufficient statistics globally and per spatial block.
    Stores raw test rows (capped at max_test_rows) for held-out evaluation.
    """

    def __init__(self, p: int, max_test_rows: int) -> None:
        self.p        = p
        self.max_test = max_test_rows
        self.XtX      = np.zeros((p, p))
        self.Xty      = np.zeros(p)
        self.yty      = 0.0
        self.sum_y    = 0.0
        self.n        = 0
        self.blocks: Dict[str, list] = {}

    def _ensure_block(self, b: str) -> None:
        if b not in self.blocks:
            self.blocks[b] = [
                np.zeros((self.p, self.p)),
                np.zeros(self.p),
                0.0,
                0.0,
                0,
                [],
                [],
            ]

    def update(
        self,
        X:      np.ndarray,
        y:      np.ndarray,
        blocks: np.ndarray,
    ) -> None:
        mask = ~np.isnan(X).any(axis=1)
        X_cc = X[mask]
        y_cc = y[mask]
        b_cc = blocks[mask]

        if len(X_cc) == 0:
            return

        self.XtX   += X_cc.T @ X_cc
        self.Xty   += X_cc.T @ y_cc
        self.yty   += float(y_cc @ y_cc)
        self.sum_y += float(y_cc.sum())
        self.n     += len(y_cc)

        for blk in np.unique(b_cc):
            idx = b_cc == blk
            self._ensure_block(blk)
            s = self.blocks[blk]
            x_b = X_cc[idx]
            y_b = y_cc[idx]
            s[0] += x_b.T @ x_b
            s[1] += x_b.T @ y_b
            s[2] += float(y_b @ y_b)
            s[3] += float(y_b.sum())
            s[4] += len(y_b)
            already = sum(len(a) for a in s[5])
            if already < self.max_test:
                take = min(len(x_b), self.max_test - already)
                s[5].append(x_b[:take])
                s[6].append(y_b[:take])


def _loo_block_eval(
    acc: _BlockAccumulator,
    ridge: float,
    min_block_n: int,
) -> List[dict]:
    """Leave-one-block-out evaluation from sufficient statistics."""
    p         = acc.p
    I         = np.eye(p)
    ybar_glob = acc.sum_y / max(acc.n, 1)
    sst_glob  = acc.yty - acc.n * ybar_glob ** 2

    results = []
    for blk, s in acc.blocks.items():
        XtX_b, Xty_b, yty_b, sum_y_b, n_b, X_raw_list, y_raw_list = s

        if n_b < min_block_n:
            continue
        if not X_raw_list:
            continue

        X_test = np.vstack(X_raw_list)
        y_test = np.concatenate(y_raw_list)
        n_test = len(y_test)

        XtX_tr = acc.XtX - XtX_b
        Xty_tr = acc.Xty - Xty_b
        n_tr   = acc.n   - n_b

        if n_tr < p:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beta, _, _, _ = np.linalg.lstsq(XtX_tr + ridge * I, Xty_tr, rcond=None)

        y_pred = X_test @ beta
        resid  = y_test - y_pred

        mae  = float(np.mean(np.abs(resid)))
        rmse = float(np.sqrt(np.mean(resid ** 2)))

        ybar_test = float(y_test.mean())
        sst_test  = float(np.sum((y_test - ybar_test) ** 2))
        r2_test   = float(1.0 - np.sum(resid ** 2) / max(sst_test, 1e-14))
        r2_test   = float(np.clip(r2_test, -1.0, 1.0))

        results.append({
            "block":   blk,
            "n_train": n_tr,
            "n_test":  n_test,
            "mae":     round(mae,  4),
            "rmse":    round(rmse, 4),
            "r2":      round(r2_test, 4),
        })

    return results


def _aggregate(block_results: List[dict]) -> dict:
    if not block_results:
        return {}
    n_blocks = len(block_results)
    maes  = [r["mae"]  for r in block_results]
    rmses = [r["rmse"] for r in block_results]
    r2s   = [r["r2"]   for r in block_results]
    return {
        "n_blocks":   n_blocks,
        "mae_mean":   round(float(np.mean(maes)),  4),
        "mae_std":    round(float(np.std(maes)),   4),
        "rmse_mean":  round(float(np.mean(rmses)), 4),
        "rmse_std":   round(float(np.std(rmses)),  4),
        "r2_mean":    round(float(np.mean(r2s)),   4),
        "r2_std":     round(float(np.std(r2s)),    4),
        "r2_median":  round(float(np.median(r2s)), 4),
        "r2_q10":     round(float(np.percentile(r2s, 10)), 4),
        "r2_q90":     round(float(np.percentile(r2s, 90)), 4),
    }


def _random_fold_eval(
    acc: _BlockAccumulator,
    ridge: float,
    n_folds: int,
    rng: np.random.Generator,
) -> dict:
    """Random K-fold on the block test rows as a comparison baseline."""
    all_X = []
    all_y = []
    for s in acc.blocks.values():
        if s[5]:
            all_X.append(np.vstack(s[5]))
            all_y.append(np.concatenate(s[6]))

    if not all_X:
        return {}

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    n     = len(y_all)

    if n < n_folds * 2:
        return {}

    perm  = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    I     = np.eye(acc.p)
    r2s   = []
    maes  = []
    rmses = []

    for k_fold, fold_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != k_fold])
        X_tr = X_all[train_idx]
        y_tr = y_all[train_idx]
        X_te = X_all[fold_idx]
        y_te = y_all[fold_idx]

        XtX_tr = X_tr.T @ X_tr
        Xty_tr = X_tr.T @ y_tr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beta, _, _, _ = np.linalg.lstsq(XtX_tr + ridge * I, Xty_tr, rcond=None)

        resid = y_te - X_te @ beta
        mae   = float(np.mean(np.abs(resid)))
        rmse  = float(np.sqrt(np.mean(resid ** 2)))
        sst   = float(np.sum((y_te - y_te.mean()) ** 2))
        r2    = float(np.clip(1.0 - np.sum(resid ** 2) / max(sst, 1e-14), -1.0, 1.0))
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    return {
        "n_folds":   n_folds,
        "mae_mean":  round(float(np.mean(maes)),  4),
        "mae_std":   round(float(np.std(maes)),   4),
        "rmse_mean": round(float(np.mean(rmses)), 4),
        "rmse_std":  round(float(np.std(rmses)),  4),
        "r2_mean":   round(float(np.mean(r2s)),   4),
        "r2_std":    round(float(np.std(r2s)),    4),
    }


# ── Driver ─────────────────────────────────────────────────────────────────────

def run(
    max_rows:       Optional[int],
    batch_size:     int,
    block_col:      str,
    max_test_rows:  int,
    ridge:          float,
    min_block_n:    int,
    n_random_folds: int,
    dataset:        Optional[str] = None,
) -> dict:
    if dataset:
        from utils import load_parquet_sample
        data_dir = _SRC / "test_data"
        path = data_dir / f"{dataset}.parquet"
        df_full = load_parquet_sample(path, max_rows, seed=42)
        batch_iter   = (df_full.iloc[i:i+batch_size]
                        for i in range(0, len(df_full), batch_size))
        source_label = f"dataset:{dataset} (n={len(df_full):,})"
    else:
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

    acc: Optional[_BlockAccumulator] = None
    feature_cols: List[str] = []

    total_rows = 0
    t0 = time.perf_counter()

    bar = tqdm(desc=f"spatial_cv ({source_label})", unit="rows", dynamic_ncols=True)
    try:
        for batch_df in batch_iter:
            if batch_df is None or len(batch_df) == 0:
                continue

            y    = batch_df["temperature"].to_numpy(dtype=np.float64)
            lons = batch_df["longitude"].to_numpy(dtype=np.float64)
            lats = batch_df["latitude"].to_numpy(dtype=np.float64)

            blocks = _compute_blocks(lons, lats, block_col)
            Z_conf = _build_confounders(batch_df)

            if acc is None:
                feature_cols = [c for c in batch_df.columns
                                if c not in _NON_FEATURE
                                and np.issubdtype(batch_df[c].dtype, np.floating)]
                p = len(feature_cols) + Z_conf.shape[1]
                acc = _BlockAccumulator(p, max_test_rows=max_test_rows)
                tqdm.write(f"features: {len(feature_cols)}  confounders: {Z_conf.shape[1]}  p={p}")

            X_feat = batch_df[feature_cols].to_numpy(dtype=np.float64)
            X_full = np.hstack([X_feat, Z_conf])

            acc.update(X_full, y, blocks)

            total_rows += len(batch_df)
            bar.update(len(batch_df))
    finally:
        bar.close()

    elapsed = time.perf_counter() - t0
    tqdm.write(f"stream done: {total_rows:,} rows in {elapsed/60:.1f} min")

    if acc is None:
        print("No data streamed.", file=sys.stderr)
        return {}

    tqdm.write(f"blocks found: {len(acc.blocks)}  -- running LOO-CV ...")
    block_results = _loo_block_eval(acc, ridge=ridge, min_block_n=min_block_n)
    tqdm.write(f"LOO-CV complete: {len(block_results)} blocks evaluated")

    agg = _aggregate(block_results)

    rng = np.random.default_rng(42)
    rand_cv = _random_fold_eval(acc, ridge=ridge, n_folds=n_random_folds, rng=rng)

    block_results.sort(key=lambda r: r["r2"])

    output = {
        "_meta": {
            "algorithm":           "LOO-block-CV via sufficient-statistics subtraction",
            "block_col":           block_col,
            "ridge_lambda":        ridge,
            "min_block_n":         min_block_n,
            "max_test_rows":       max_test_rows,
            "source":              source_label,
            "total_rows_streamed": total_rows,
            "n_features":          len(feature_cols),
            "elapsed_s":           round(elapsed, 1),
        },
        "summary":        agg,
        "random_fold_cv": rand_cv,
        "per_block":      block_results,
    }

    return output


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--max-rows",       type=int,   default=None)
    ap.add_argument("--batch-size",     type=int,   default=50_000)
    ap.add_argument("--block-col",      default="h3r8",
                    choices=["h3r8", "h3r7", "rect1km", "rect2km"],
                    help="spatial block assignment strategy (default: h3r8)")
    ap.add_argument("--max-test-rows",  type=int,   default=300,
                    help="max raw test rows stored per block (default: 300)")
    ap.add_argument("--ridge",          type=float, default=1e-4,
                    help="ridge regularisation lambda (default: 1e-4)")
    ap.add_argument("--min-block-n",    type=int,   default=50,
                    help="min rows per block to include in LOO-CV (default: 50)")
    ap.add_argument("--n-random-folds", type=int,   default=10,
                    help="folds for random-fold comparison (default: 10)")
    ap.add_argument("--dataset",        type=str,   default=None,
                    help="run on one specific pre-built dataset parquet "
                         "(default: all parquets in test_data/)")
    ap.add_argument("--full-stream",    action="store_true",
                    help="stream all source data instead of using pre-built parquets")
    ap.add_argument("--out",            type=Path,  default=_OUT_DEFAULT)
    args = ap.parse_args()

    if args.full_stream and args.dataset:
        ap.error("--full-stream and --dataset are mutually exclusive")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _call(**kw):
        return run(max_rows=args.max_rows, batch_size=args.batch_size,
                   block_col=args.block_col, max_test_rows=args.max_test_rows,
                   ridge=args.ridge, min_block_n=args.min_block_n,
                   n_random_folds=args.n_random_folds, **kw)

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
