"""
cross_corr.py
=============

Cross-source feature-relation tests over pre-built parquet datasets produced
by ``build_test_datasets.py``.  For every pair of features whose source groups
differ (Trees / DHM / UA / WIS / LST), compute:

  - Pearson r  (linear correlation, scipy.stats.pearsonr)
  - Spearman rho (rank correlation, scipy.stats.spearmanr)
  - Mutual information (sklearn.feature_selection.mutual_info_regression,
                        KNN-based estimator, n_neighbors=3)

Same-source pairs are skipped -- they are related by construction (e.g.
DHM avg/max/min at 50/70/100 m share the same elevation surface).

Source detection
----------------
Feature columns are routed to one of five source groups by name prefix.  See
``_classify_source()`` for the rules.  Columns that don't match any rule
(timestamps, IDs, raw coordinates, ...) are ignored.

Output
------
Results are written to ``src/ds_reports/cross_corr.json``.

Performance
-----------
Each dataset is sub-sampled to at most ``--max-rows`` rows (default 50,000)
using representative per-row-group sampling before computing metrics.

Usage
-----
    python src/ds/cross_corr.py                    # all datasets in test_data/
    python src/ds/cross_corr.py --only full_representative,single_image
    python src/ds/cross_corr.py --build-missing    # build missing then analyse
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from tqdm.auto import tqdm

_DS_DIR       = Path(__file__).parent
_SRC          = _DS_DIR.parent
_DATA_DIR     = _SRC / "test_data"
_RESULTS_PATH = _SRC / "ds_reports" / "cross_corr.json"

sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_DS_DIR))


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------

_NON_FEATURE_COLS = frozenset({
    "longitude", "latitude", "image_id", "timestamp", "partition_key",
    "tile_id", "year", "month_of_year", "day_of_month", "day_of_year",
    "hour_of_day",
})

_LST_COLS = frozenset({"temperature", "aster_lst", "modis_lst", "ndvi"})


def _classify_source(col: str) -> Optional[str]:
    """Return source group of a column, or None to skip it."""
    if col in _NON_FEATURE_COLS:
        return None
    if col in _LST_COLS:
        return "LST"
    if col.startswith("dhm_"):
        return "DHM"
    if col.startswith("trees_"):
        return "Trees"
    if col.startswith("ua_"):
        return "UA"
    if col.startswith("wis_"):
        return "WIS"
    return None


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def _group_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map source -> feature columns present in df, dropping all-NaN/constant cols."""
    groups: Dict[str, List[str]] = {}
    for col in df.columns:
        src = _classify_source(col)
        if src is None:
            continue
        s = df[col]
        if s.isna().all():
            continue
        if s.nunique(dropna=True) < 2:
            continue
        groups.setdefault(src, []).append(col)
    return groups


def _analyse_pair_block(
    X:        np.ndarray,
    Y:        np.ndarray,
    x_cols:   List[str],
    y_cols:   List[str],
    src_x:    str,
    src_y:    str,
    rng_seed: int,
) -> List[dict]:
    """Compute Pearson, Spearman, MI for every (xc, yc) pair in this block."""
    out: List[dict] = []

    mi_matrix = np.empty((X.shape[1], Y.shape[1]), dtype=np.float64)
    for j in range(Y.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi_matrix[:, j] = mutual_info_regression(
                X, Y[:, j], n_neighbors=3, random_state=rng_seed, copy=False,
            )

    for i, xc in enumerate(x_cols):
        for j, yc in enumerate(y_cols):
            x = X[:, i]
            y = Y[:, j]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pr = stats.pearsonr(x, y)
                sp = stats.spearmanr(x, y)
            out.append({
                "feature_a":   xc,         "source_a":   src_x,
                "feature_b":   yc,         "source_b":   src_y,
                "n_valid":     int(X.shape[0]),
                "pearson_r":   float(pr.statistic) if not np.isnan(pr.statistic) else None,
                "pearson_p":   float(pr.pvalue)    if not np.isnan(pr.pvalue)    else None,
                "spearman_r":  float(sp.statistic) if not np.isnan(sp.statistic) else None,
                "spearman_p":  float(sp.pvalue)    if not np.isnan(sp.pvalue)    else None,
                "mi":          float(mi_matrix[i, j]),
            })
    return out


def run(
    dataset:  str,
    max_rows: int = 50_000,
    rng_seed: int = 42,
) -> dict:
    """
    Analyse one pre-built parquet dataset.  Returns the result dict.

    Can be called from the test suite orchestrator::

        from cross_corr import run
        result = run("full_representative", max_rows=50_000)
    """
    from utils import load_parquet_sample

    path = _DATA_DIR / f"{dataset}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"dataset {dataset!r} not found at {path}\n"
            f"Run: python build_test_datasets.py --only {dataset}"
        )

    df = load_parquet_sample(path, max_rows, seed=rng_seed)

    keep = [c for c in df.columns if _classify_source(c) is not None]
    df   = df[keep]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df   = df[num_cols]

    n_rows_total = len(df)

    if n_rows_total < 100:
        return {
            "n_rows_total":        n_rows_total,
            "n_rows_analysed":     0,
            "feature_groups":      {},
            "skipped_same_source": 0,
            "pairs":               [],
            "note":                "fewer than 100 valid rows -- analysis skipped",
        }

    groups = _group_features(df)
    feature_groups = {src: len(cols) for src, cols in groups.items()}
    if len(groups) < 2:
        return {
            "n_rows_total":        n_rows_total,
            "n_rows_analysed":     n_rows_total,
            "feature_groups":      feature_groups,
            "skipped_same_source": 0,
            "pairs":               [],
            "note":                "fewer than 2 source groups present -- no cross-source pairs",
        }

    sources = sorted(groups.keys())
    pairs: List[dict] = []
    skipped_same = 0
    for src, cols in groups.items():
        skipped_same += len(cols) * (len(cols) - 1) // 2

    n_cross_blocks = len(sources) * (len(sources) - 1) // 2
    bar = tqdm(total=n_cross_blocks, desc=f"cross_corr:{dataset}",
               unit="block", dynamic_ncols=True, position=0)
    try:
        for i, src_x in enumerate(sources):
            for src_y in sources[i + 1:]:
                x_cols = groups[src_x]
                y_cols = groups[src_y]
                block = df[x_cols + y_cols].dropna(axis=0, how="any")
                if len(block) < 100:
                    bar.update(1)
                    continue
                X = block[x_cols].to_numpy(dtype=np.float64, copy=False)
                Y = block[y_cols].to_numpy(dtype=np.float64, copy=False)
                pairs.extend(_analyse_pair_block(
                    X, Y, x_cols, y_cols, src_x, src_y, rng_seed,
                ))
                bar.update(1)
    finally:
        bar.close()

    return {
        "n_rows_total":         n_rows_total,
        "n_rows_analysed":      n_rows_total,
        "feature_groups":       feature_groups,
        "skipped_same_source":  skipped_same,
        "pairs":                pairs,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-rows", type=int, default=50_000,
                        help="cap per-dataset sample size (default: 50000)")
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated dataset names to analyse (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for sub-sampling and MI estimator")
    parser.add_argument("--build-missing", action="store_true",
                        help="build any missing test datasets before analysing")
    parser.add_argument("--out", type=Path, default=_RESULTS_PATH,
                        help=f"output JSON path (default: {_RESULTS_PATH})")
    args = parser.parse_args()

    wanted = {n.strip() for n in args.only.split(",") if n.strip()} if args.only else None

    if args.build_missing:
        from build_test_datasets import build as _build_datasets
        _build_datasets(names=wanted, rebuild=False)

    from utils import list_datasets
    datasets = list_datasets()
    if wanted:
        datasets = [(n, p) for n, p in datasets if n in wanted]

    if not datasets:
        print(f"no parquet files found under {_DATA_DIR} -- "
              f"run build_test_datasets.py first (or pass --build-missing)",
              file=sys.stderr)
        return 1

    print(f"datasets to analyse: {[n for n, _ in datasets]}")
    print(f"max rows per dataset: {args.max_rows:,}")

    results: dict = {
        "_meta": {
            "max_rows":      args.max_rows,
            "seed":          args.seed,
            "metrics":       ["pearson_r", "pearson_p",
                              "spearman_r", "spearman_p", "mi"],
            "source_groups": ["DHM", "Trees", "UA", "WIS", "LST"],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for name, _ in datasets:
        print(f"\n=== {name} ===")
        t_d = time.perf_counter()
        results[name] = run(name, max_rows=args.max_rows, rng_seed=args.seed)
        elapsed = time.perf_counter() - t_d
        n_pairs = len(results[name].get("pairs", []))
        print(f"    {n_pairs:,} cross-source pairs in {elapsed:.1f}s")
        args.out.write_text(json.dumps(results, indent=2, default=str),
                            encoding="utf-8")

    total = time.perf_counter() - t0
    print(f"\nall analyses done in {total / 60:.1f} min -- wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
