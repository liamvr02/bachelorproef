"""
lst_full_test.py — End-to-end streaming + training test across all models.

Stream features
---------------
- DHM (merged DHM1+DHM2) aggregate measures for `elevation` at radii 50, 70, 100 m
  (avg, sum, min, max — count omitted since every tile has the same count)
- Tree count at radii 50, 70, 100 m
- Urban Atlas classification fractions at radii 50 and 100 m (UA map)
- WIS bestemming fractions at radii 50 and 100 m (WIS_BESTEMMING map)
- WIS materiaalsoort fractions at radii 50 and 100 m (WIS_MATERIAAL map)

Training
--------
- Cyclical (sin/cos) transforms for day_of_year, month_of_year, hour_of_day
- Distribution-targeted training stream (uniform year / month / hour)
- Outlier date ranges held out of training and evaluated as separate splits

Memory contract
---------------
Training data is streamed batch-by-batch directly into ``train_all`` — never
materialised into a single ``pd.concat`` DataFrame.  Outlier eval splits are
streamed once into per-label parquet files under ``stream_cache/full_test/``
and read back in chunks during evaluation.  Peak RAM is dominated by the
reservoir / model state, not by streamed rows.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from ml import train_all, cyclical
from stream.classification_groups import UA, WIS_BESTEMMING, WIS_MATERIAAL
from stream.config import get_dimension_edges
from stream.features import (
    FeatureRegistry,
    aggregate_in_radius,
    urban_atlas_classifications_fractions,
    wis_fraction,
)
from stream.logging_config import configure_logging
from stream.stream import StreamConfig

configure_logging(level="DEBUG")


_SRC      = Path(__file__).parent
_REPORTS  = _SRC / "reports"
_CACHE    = _SRC / "stream_cache" / "full_test"
_REPORTS.mkdir(exist_ok=True)
_CACHE.mkdir(parents=True, exist_ok=True)


# ============================================================
# Outlier date ranges (inclusive, end-exclusive at month level)
# ============================================================

OUTLIER_RANGES: List[Tuple[date, date, str]] = [
    (date(2006, 6, 1),  date(2006, 9, 1),  "heat_2006_summer"),
    (date(2007, 4, 1),  date(2007, 5, 1),  "warm_2007_apr"),
    (date(2010, 1, 1),  date(2010, 3, 1),  "cold_2010_winter"),
    (date(2012, 2, 1),  date(2012, 3, 1),  "cold_2012_feb"),
    (date(2013, 7, 1),  date(2013, 9, 1),  "heat_2013_summer"),
    (date(2015, 7, 1),  date(2015, 9, 1),  "heat_2015_summer"),
    (date(2018, 5, 1),  date(2018, 9, 1),  "drought_2018"),
    (date(2019, 6, 1),  date(2019, 8, 1),  "heat_2019_jun_jul"),
    (date(2020, 8, 1),  date(2020, 9, 1),  "heat_2020_aug"),
]


def expand_months(start: date, end: date) -> List[str]:
    """Return list of 'YYYY-MM' keys from start (inclusive) to end (exclusive)."""
    out: List[str] = []
    y, m = start.year, start.month
    while (y, m) < (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return out


# ============================================================
# Feature registry
# ============================================================

def build_registry() -> FeatureRegistry:
    reg = FeatureRegistry()

    dhm_radii = (50, 70, 100)
    dhm_aggs = ("avg", "sum", "min", "max")
    for r in dhm_radii:
        for agg in dhm_aggs:
            reg.add(aggregate_in_radius(
                "dhm",
                radius_m=r,
                columns=["elevation"],
                agg=agg,
                temporal="last_previous",
            ))

    for r in (50, 70, 100):
        reg.add(aggregate_in_radius(
            "trees", radius_m=r, columns=[], agg="count", temporal="none",
        ))

    for r in (50, 100):
        reg.add(urban_atlas_classifications_fractions(
            classification_map=UA, radius_m=r,
        ))

    for r in (50, 100):
        for leaves in WIS_BESTEMMING.values():
            for val in leaves:
                reg.add(wis_fraction(attr_col="bestemming",
                                     attr_val=val, radius_m=r))
        for leaves in WIS_MATERIAAL.values():
            for val in leaves:
                reg.add(wis_fraction(attr_col="materiaalsoort",
                                     attr_val=val, radius_m=r))

    return reg


# ============================================================
# Stream configs
# ============================================================

UNIFORM_DISTRIBUTION: Dict = {
    "year":          ({}, get_dimension_edges("year")),
    "month_of_year": ({}, get_dimension_edges("month_of_year")),
    "hour_of_day":   ({}, get_dimension_edges("hour_of_day")),
}


def all_partitions() -> List[str]:
    probe = StreamConfig()
    probe._load_catalog()
    return list(probe._partition_keys)


def training_config(exclude_keys: List[str], batch_size: int) -> StreamConfig:
    exclude = set(exclude_keys)
    train_keys = [k for k in all_partitions() if k not in exclude]
    cfg = StreamConfig(partition_keys=train_keys, batch_size=batch_size)
    cfg.set_distribution(UNIFORM_DISTRIBUTION)
    return cfg


def outlier_config(keys: List[str], batch_size: int) -> StreamConfig:
    return StreamConfig(partition_keys=keys, batch_size=batch_size)


# ============================================================
# Parquet helpers — bounded-memory eval-split caching
# ============================================================

def _stream_to_parquet(
    cfg:        StreamConfig,
    reg:        FeatureRegistry,
    max_rows:   Optional[int],
    batch_size: int,
    path:       Path,
    label:      str,
) -> int:
    """Stream rows into *path* via incremental ParquetWriter; return row count."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if path.exists():
        path.unlink()

    writer: Optional[pq.ParquetWriter] = None
    n = 0
    bar = tqdm(total=max_rows, desc=f"stream:{label}", unit="row",
               dynamic_ncols=True, leave=False)
    try:
        for df in cfg.stream(reg, batch_size=batch_size, max_rows=max_rows):
            if df.empty:
                continue
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            n += len(df)
            bar.update(len(df))
    finally:
        if writer is not None:
            writer.close()
        bar.close()
    return n


def _pq_iter(path: Path, batch_size: int) -> Generator[pd.DataFrame, None, None]:
    """Yield parquet record batches as DataFrames (~batch_size rows each)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch_size):
        yield rb.to_pandas()


def _ensure_outlier_caches(
    reg:           FeatureRegistry,
    keys_by_label: Dict[str, List[str]],
    max_rows:      Optional[int],
    batch_size:    int,
    rebuild:       bool,
) -> Dict[str, Path]:
    """Stream each outlier label into its own parquet file (or reuse if present)."""
    out: Dict[str, Path] = {}
    for label, keys in keys_by_label.items():
        path = _CACHE / f"outlier_{label}.parquet"
        if path.exists() and not rebuild:
            out[label] = path
            continue
        cfg = outlier_config(keys, batch_size)
        n = _stream_to_parquet(cfg, reg, max_rows, batch_size, path, label)
        if n > 0:
            out[label] = path
        else:
            print(f"  {label:<24} empty (no rows streamed)")
    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows",         type=int, default=5_000_000,
                    help="training rows to stream (default 5M; -1 = full stream)")
    ap.add_argument("--outlier-rows", type=int, default=500_000,
                    help="rows per outlier split (default 500k; -1 = unlimited)")
    ap.add_argument("--batch",        type=int, default=100_000)
    ap.add_argument("--rebuild-outlier-cache", action="store_true",
                    help="re-stream every outlier parquet shard from scratch")
    args = ap.parse_args()

    train_max_rows   = None if args.rows         == -1 else args.rows
    outlier_max_rows = None if args.outlier_rows == -1 else args.outlier_rows
    batch_size       = args.batch

    reg = build_registry()

    outlier_keys_by_label: Dict[str, List[str]] = {
        label: expand_months(start, end)
        for start, end, label in OUTLIER_RANGES
    }
    excluded_keys: List[str] = sorted({
        k for ks in outlier_keys_by_label.values() for k in ks
    })

    print("=" * 70)
    print(f"Training partitions exclude {len(excluded_keys)} outlier months")
    print(f"Outlier evaluation splits:   {len(outlier_keys_by_label)}")
    print("=" * 70)

    # ---- Outlier eval shards (parquet, bounded memory on read-back) ----
    outlier_paths = _ensure_outlier_caches(
        reg, outlier_keys_by_label,
        max_rows   = outlier_max_rows,
        batch_size = batch_size,
        rebuild    = args.rebuild_outlier_cache,
    )
    for label, path in outlier_paths.items():
        import pyarrow.parquet as pq
        n = pq.ParquetFile(path).metadata.num_rows
        print(f"  {label:<24} {n:>8,} rows  ({path.name})")

    transforms = [
        cyclical("hour_of_day", 24),
        cyclical("day_of_year", 365),
        cyclical("month_of_year", 12),
    ]

    # ---- Train every model on the streaming source directly ----
    # train_all() handles streaming end-to-end: scaler fits across the stream,
    # streaming models use partial_fit, RF/HGB use reservoir sampling.
    train_cfg = training_config(excluded_keys, batch_size)
    results = train_all(
        source     = train_cfg,
        registry   = reg,
        transforms = transforms,
        batch_size = batch_size,
        max_rows   = train_max_rows,
    )

    # ---- Reports — every eval is a parquet-streamed generator ----
    for name, model in results.items():
        print(f"\n── {name} ──")
        # No stand-alone in-distribution holdout split with this approach;
        # the stream-aggregate metrics from training are the in-distribution
        # number, so we report them via report() with no eval_df.
        model.report("stdout")
        model.report(
            "html",
            path=_REPORTS / f"{name}.html",
            n_shap=1000,
        )
        for label, path in outlier_paths.items():
            print(f"  outlier {label}:")
            # model.evaluate / model.report accept generators of DataFrames
            model.report("stdout",
                         eval_df=_pq_iter(path, batch_size))  # type: ignore[arg-type]
            model.report(
                "html",
                path=_REPORTS / f"{name}__{label}.html",
                eval_df=_pq_iter(path, batch_size),  # type: ignore[arg-type]
                n_shap=500,
            )


if __name__ == "__main__":
    sys.exit(main() or 0)
