"""
ds_test_suite.py
================

Cross-source feature-relation tests over the parquet datasets produced by
``build_test_datasets.py``.  For every pair of features whose source groups
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
Results are written to ``src/ds_test_results.json``:

    {
      "<dataset_name>": {
        "n_rows_total":        12345,
        "n_rows_analysed":     12000,
        "feature_groups":      {"DHM": 9, "Trees": 12, ...},
        "skipped_same_source": 234,
        "pairs": [
          {
            "feature_a":   "...",  "source_a":  "DHM",
            "feature_b":   "...",  "source_b":  "WIS",
            "n_valid":     11500,
            "pearson_r":   0.123,  "pearson_p":  1e-9,
            "spearman_r":  0.131,  "spearman_p": 1e-10,
            "mi":          0.034
          },
          ...
        ]
      },
      ...
    }

Performance
-----------
Each dataset is sub-sampled to at most ``--max-analysis-rows`` rows (default
50,000) before computing metrics.  KNN-based MI scales roughly O(n log n)
per call; the per-source-pair vectorised call (one X matrix -> many y) keeps
total runtime per dataset under ~30 minutes for the full registry.
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


_SRC          = Path(__file__).parent
_DATA_DIR     = _SRC / "test_data"
_RESULTS_PATH = _SRC / "ds_test_results.json"


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------

# Identifier / scene-context columns that aren't features.
_NON_FEATURE_COLS = frozenset({
    "longitude", "latitude", "image_id", "timestamp", "partition_key",
    "tile_id", "year", "month_of_year", "day_of_month", "day_of_year",
    "hour_of_day",
})

_LST_COLS = frozenset({"temperature", "aster_lst", "modis_lst", "ndvi"})


def _classify_source(col: str) -> Optional[str]:
    """
    Return the source group of a column, or None to skip it entirely.

    Same-source pairs (intra-group) are excluded by construction: features
    from a single source vary with each other through the source dataset
    itself, so any correlation is a property of that dataset, not of the
    real-world process under study.
    """
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

def _load_and_sample(path: Path, max_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Read parquet, drop non-numeric/non-feature columns, sample to <= max_rows."""
    df = pd.read_parquet(path)
    keep = [c for c in df.columns if _classify_source(c) is not None]
    df = df[keep]

    # Numeric columns only -- categorical sources (none expected here) would
    # need encoding before MI/correlation.
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[num_cols]

    if len(df) > max_rows:
        idx = rng.choice(len(df), size=max_rows, replace=False)
        df = df.iloc[np.sort(idx)].reset_index(drop=True)
    return df


def _group_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map source -> feature columns present in *df*, dropping all-NaN/constant cols."""
    groups: Dict[str, List[str]] = {}
    for col in df.columns:
        src = _classify_source(col)
        if src is None:
            continue
        s = df[col]
        # Skip columns that are entirely NaN or constant (zero variance) --
        # Pearson/Spearman are undefined and MI is identically zero.
        if s.isna().all():
            continue
        if s.nunique(dropna=True) < 2:
            continue
        groups.setdefault(src, []).append(col)
    return groups


def _analyse_pair_block(
    X:        np.ndarray,           # (n, |X cols|), no NaN
    Y:        np.ndarray,           # (n, |Y cols|), no NaN
    x_cols:   List[str],
    y_cols:   List[str],
    src_x:    str,
    src_y:    str,
    rng_seed: int,
) -> List[dict]:
    """
    Compute Pearson, Spearman, MI for every (xc, yc) pair in this block.

    The MI loop calls ``mutual_info_regression`` once per y column -- sklearn
    handles the X matrix internally as a batch, which is faster than nesting.
    Pearson and Spearman are computed pair-by-pair (vectorising spearmanr
    across all columns wastes work since p-values come for free in the loop).
    """
    out: List[dict] = []

    # Pre-compute MI with X as the feature block, looping over y columns.
    mi_matrix = np.empty((X.shape[1], Y.shape[1]), dtype=np.float64)
    for j in range(Y.shape[1]):
        # n_neighbors=3 is sklearn's default; copy=False avoids an internal copy.
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


def _run_dataset(
    name:         str,
    path:         Path,
    max_rows:     int,
    rng_seed:     int,
) -> dict:
    rng = np.random.default_rng(rng_seed)
    df = _load_and_sample(path, max_rows, rng)
    n_rows_total = len(df)

    if n_rows_total < 100:
        return {
            "n_rows_total":     n_rows_total,
            "n_rows_analysed":  0,
            "feature_groups":   {},
            "skipped_same_source": 0,
            "pairs":            [],
            "note":             "fewer than 100 valid rows -- analysis skipped",
        }

    groups = _group_features(df)
    feature_groups = {src: len(cols) for src, cols in groups.items()}
    if len(groups) < 2:
        return {
            "n_rows_total":     n_rows_total,
            "n_rows_analysed":  n_rows_total,
            "feature_groups":   feature_groups,
            "skipped_same_source": 0,
            "pairs":            [],
            "note":             "fewer than 2 source groups present -- no cross-source pairs",
        }

    sources = sorted(groups.keys())
    pairs: List[dict] = []
    skipped_same = 0

    # Estimate same-source pair count for the report.
    for src, cols in groups.items():
        skipped_same += len(cols) * (len(cols) - 1) // 2

    # Cross-source iteration: each unordered (src_x, src_y) once.
    n_cross_blocks = len(sources) * (len(sources) - 1) // 2
    bar = tqdm(total=n_cross_blocks, desc=f"analyse:{name}",
               unit="block", dynamic_ncols=True, position=0)
    try:
        for i, src_x in enumerate(sources):
            for src_y in sources[i + 1:]:
                x_cols = groups[src_x]
                y_cols = groups[src_y]

                # Drop rows with NaN in any column of this block.  Doing this
                # per source-pair (rather than once over the whole frame)
                # preserves more rows when one feature is sparsely populated.
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

def _list_datasets() -> List[Tuple[str, Path]]:
    """Return [(name, parquet_path)] for every dataset in test_data/."""
    if not _DATA_DIR.exists():
        return []
    return sorted(
        (p.stem, p) for p in _DATA_DIR.glob("*.parquet")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-analysis-rows", type=int, default=50_000,
                        help="cap per-dataset sample size for the analysis "
                             "(default: 50000)")
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated dataset names (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for sub-sampling and MI estimator")
    parser.add_argument("--out", type=Path, default=_RESULTS_PATH,
                        help=f"output JSON path (default: {_RESULTS_PATH})")
    args = parser.parse_args()

    datasets = _list_datasets()
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        datasets = [(n, p) for n, p in datasets if n in wanted]

    if not datasets:
        print(f"no parquet files found under {_DATA_DIR} -- "
              f"run build_test_datasets.py first", file=sys.stderr)
        return 1

    print(f"datasets to analyse: {[n for n, _ in datasets]}")
    print(f"max analysis rows per dataset: {args.max_analysis_rows:,}")

    results: dict = {
        "_meta": {
            "max_analysis_rows": args.max_analysis_rows,
            "seed":              args.seed,
            "metrics":           ["pearson_r", "pearson_p",
                                  "spearman_r", "spearman_p", "mi"],
            "source_groups":     ["DHM", "Trees", "UA", "WIS", "LST"],
        },
    }

    t0 = time.perf_counter()
    for name, path in datasets:
        print(f"\n=== {name} ===")
        t_d = time.perf_counter()
        results[name] = _run_dataset(name, path,
                                     max_rows=args.max_analysis_rows,
                                     rng_seed=args.seed)
        elapsed = time.perf_counter() - t_d
        n_pairs = len(results[name].get("pairs", []))
        print(f"    {n_pairs:,} cross-source pairs in {elapsed:.1f}s")

        # Persist after each dataset so partial runs aren't lost.
        args.out.write_text(json.dumps(results, indent=2, default=str),
                            encoding="utf-8")

    total = time.perf_counter() - t0
    print(f"\nall analyses done in {total / 60:.1f} min -- wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
