# -*- coding: utf-8 -*-
"""
build_full_datasets.py
======================
Two-phase builder: one full streaming pass, then filter-fan-out to each
derived test dataset — so feature computation happens exactly once.

Phase 1  (triggered by --rebuild or absent full_stream.parquet)
    Stream every available LST row with full feature computation
    (canonical registry) and write to test_data/full_stream.parquet.

Phase 2  (always)
    Single sequential scan of full_stream.parquet.  For each batch,
    every derived-dataset filter is applied and matching rows are
    appended to that dataset's open ParquetWriter.  All writers stay
    open simultaneously so the file is visited exactly once.

    The single_image dataset requires knowing the dominant
    (tile_id, timestamp) pair before writing, so its rows are
    accumulated in memory during the scan and finalised at the end.

Datasets produced
-----------------
  full_stream            all rows, all features  (source for Phase 2)
  outlier_<label> x 9   rows whose partition_key falls in the outlier period
  hour_midday            hour_of_day in [10.0, 11.0]
  hour_midnight          hour_of_day in [22.0, 23.0]
  point_a_50m            within ~50 m of point A (lon/lat bbox filter)
  point_b_50m            within ~50 m of point B
  single_image           dominant (tile_id, timestamp) in 2017-07-15
  single_year_2017       year == 2017

Note
----
full_representative is NOT produced here — it uses distribution
reweighting that must be applied at stream time, not post-hoc.
Use build_test_datasets.py for that dataset.

Usage
-----
    uv run python ds/build_full_datasets.py             # Phase 2 only (re-filter)
    uv run python ds/build_full_datasets.py --rebuild   # Phase 1 + Phase 2
    uv run python ds/build_full_datasets.py --only outlier_heat_2006_summer,single_year_2017
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

_DS_DIR   = Path(__file__).parent       # src/ds/
_SRC      = _DS_DIR.parent             # src/
_OUT_DIR  = _SRC / "test_data"
_MANIFEST = _OUT_DIR / "manifest.json"
_FULL_PQ  = _OUT_DIR / "full_stream.parquet"

sys.path.insert(0, str(_SRC))

from stream.logging_config import configure_logging
from stream.stream import StreamConfig
from stream_configs import build_registry, outlier_keys

configure_logging(level="INFO")

_BATCH_SIZE     = 500_000
_POINT_A        = (3 + 43/60 + 54.6/3600,  51 + 2/60 + 44.5/3600)
_POINT_B        = (3 + 43/60 + 19.2/3600,  51 + 3/60 + 14.4/3600)
_POINT_TOL_DEG  = 0.00072
_SINGLE_IMG_DAY = (2017, 7, 15)


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def _promote_null_cols(tbl: pa.Table) -> pa.Table:
    """Promote all-null columns (pyarrow null type) to float64."""
    new_fields, new_arrays = [], []
    for i, field in enumerate(tbl.schema):
        col = tbl.column(i)
        if pa.types.is_null(field.type):
            new_fields.append(pa.field(field.name, pa.float64(), nullable=True))
            new_arrays.append(col.cast(pa.float64()))
        else:
            new_fields.append(field)
            new_arrays.append(col)
    return pa.Table.from_arrays(new_arrays, schema=pa.schema(new_fields))


def _df_to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    tbl = _promote_null_cols(pa.Table.from_pandas(df, preserve_index=False))
    return tbl.cast(schema)


# ---------------------------------------------------------------------------
# Filter predicates  (DataFrame -> boolean Series)
# ---------------------------------------------------------------------------

def _pk_filter(keys: frozenset) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df["partition_key"].isin(keys)

def _hour_filter(center_h: float, tol: float = 0.5) -> Callable:
    return lambda df: (df["hour_of_day"] - center_h).abs() <= tol

def _point_filter(lon: float, lat: float, tol: float) -> Callable:
    return lambda df: (
        ((df["longitude"] - lon).abs() <= tol) &
        ((df["latitude"]  - lat).abs() <= tol)
    )

def _year_filter(year: int) -> Callable:
    return lambda df: df["year"] == year

def _day_filter(year: int, month: int, day: int) -> Callable:
    return lambda df: (
        (df["year"]          == year)  &
        (df["month_of_year"] == month) &
        (df["day_of_month"]  == day)
    )


def _pick_dominant_image(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the (tile_id, timestamp) pair with the most rows."""
    if df.empty:
        return df
    top_tile, top_ts = (
        df.groupby(["tile_id", "timestamp"], dropna=False)
          .size()
          .sort_values(ascending=False)
          .index[0]
    )
    result = df[(df["tile_id"] == top_tile) & (df["timestamp"] == top_ts)].copy()
    print(f"    single_image: tile_id={top_tile!r}  timestamp={top_ts!r}"
          f"  ({len(result):,} / {len(df):,} day rows)")
    return result


# ---------------------------------------------------------------------------
# Derived dataset specs
# ---------------------------------------------------------------------------

def _derived_specs() -> List[Tuple[str, dict]]:
    """
    Return ordered list of (name, spec) for every derived dataset.

    Each spec has:
        filter       : DataFrame -> boolean Series
        post_process : DataFrame -> DataFrame, or None
        description  : human-readable string
    """
    _, by_label = outlier_keys()
    specs: List[Tuple[str, dict]] = []

    for label, keys in by_label.items():
        specs.append((f"outlier_{label}", {
            "filter":       _pk_filter(frozenset(keys)),
            "post_process": None,
            "description":  f"all rows from outlier period '{label}'",
        }))

    specs += [
        ("hour_midday", {
            "filter":       _hour_filter(10.5),
            "post_process": None,
            "description":  "hour_of_day in [10.0, 11.0] (Landsat/ASTER daytime pass)",
        }),
        ("hour_midnight", {
            "filter":       _hour_filter(22.5),
            "post_process": None,
            "description":  "hour_of_day in [22.0, 23.0] (MODIS-Terra nighttime pass)",
        }),
        ("point_a_50m", {
            "filter":       _point_filter(*_POINT_A, _POINT_TOL_DEG),
            "post_process": None,
            "description":  f"within ~50 m of {_POINT_A} (lon, lat)",
        }),
        ("point_b_50m", {
            "filter":       _point_filter(*_POINT_B, _POINT_TOL_DEG),
            "post_process": None,
            "description":  f"within ~50 m of {_POINT_B} (lon, lat)",
        }),
        ("single_image", {
            "filter":       _day_filter(*_SINGLE_IMG_DAY),
            "post_process": _pick_dominant_image,
            "description":  (f"dominant (tile_id, timestamp) in "
                             f"{_SINGLE_IMG_DAY[0]}-{_SINGLE_IMG_DAY[1]:02d}-{_SINGLE_IMG_DAY[2]:02d}"),
        }),
        ("single_year_2017", {
            "filter":       _year_filter(2017),
            "post_process": None,
            "description":  "all rows from year 2017",
        }),
    ]

    return specs


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _read_manifest() -> dict:
    if _MANIFEST.exists():
        try:
            return json.loads(_MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _write_manifest(data: dict) -> None:
    _MANIFEST.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 1: stream full dataset
# ---------------------------------------------------------------------------

def phase1_stream() -> int:
    """Stream all LST rows with features to full_stream.parquet."""
    print(f"\n=== Phase 1: full stream -> {_FULL_PQ.name} ===")
    t0  = time.perf_counter()
    reg = build_registry()
    cfg = StreamConfig(batch_size=_BATCH_SIZE)

    if _FULL_PQ.exists():
        _FULL_PQ.unlink()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    n = 0
    bar = tqdm(desc="full_stream", unit="row", dynamic_ncols=True)
    try:
        for df in cfg.stream(reg, batch_size=_BATCH_SIZE):
            if df.empty:
                continue
            tbl = _promote_null_cols(pa.Table.from_pandas(df, preserve_index=False))
            if writer is None:
                writer = pq.ParquetWriter(_FULL_PQ, tbl.schema, compression="zstd")
            else:
                tbl = tbl.cast(writer.schema)
            writer.write_table(tbl)
            n += len(df)
            bar.update(len(df))
    finally:
        if writer:
            writer.close()
        bar.close()

    elapsed  = time.perf_counter() - t0
    size_gb  = _FULL_PQ.stat().st_size / 1e9 if _FULL_PQ.exists() else 0
    print(f"  {n:,} rows  {size_gb:.1f} GB  {elapsed/60:.1f} min"
          f"  ({n/max(elapsed,1):.0f} rows/s)")

    manifest = _read_manifest()
    manifest["full_stream"] = {
        "rows": n, "elapsed_s": round(elapsed, 1),
        "size_gb": round(size_gb, 2),
        "description": "all rows, all partitions, all features",
        "path": str(_FULL_PQ.relative_to(_SRC)),
    }
    _write_manifest(manifest)
    return n


# ---------------------------------------------------------------------------
# Phase 2: fan-out filter
# ---------------------------------------------------------------------------

def phase2_filter(
    specs: List[Tuple[str, dict]],
    names: Optional[set] = None,
) -> Dict[str, int]:
    """
    Single scan of full_stream.parquet, routing rows to each derived dataset.

    Datasets with post_process (single_image) accumulate in memory; all others
    write directly so memory stays bounded by one batch.

    *names* restricts which derived datasets are (re)written; None = all.
    """
    if not _FULL_PQ.exists():
        raise FileNotFoundError(
            f"{_FULL_PQ} not found — run with --rebuild to build it first"
        )

    active_specs = [(n, s) for n, s in specs if names is None or n in names]
    if not active_specs:
        return {}

    print(f"\n=== Phase 2: fan-out filter ({len(active_specs)} datasets) ===")
    t0 = time.perf_counter()

    pf     = pq.ParquetFile(_FULL_PQ)
    schema = pf.schema_arrow

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-dataset state
    writers: Dict[str, Optional[pq.ParquetWriter]] = {}
    accs:    Dict[str, List[pd.DataFrame]] = {}   # only for post-process datasets
    counts:  Dict[str, int] = {n: 0 for n, _ in active_specs}

    for name, spec in active_specs:
        out = _OUT_DIR / f"{name}.parquet"
        if out.exists():
            out.unlink()
        if spec["post_process"] is not None:
            accs[name] = []
        else:
            writers[name] = None

    n_total = pf.metadata.num_rows
    bar = tqdm(total=n_total, desc="fan-out", unit="row", dynamic_ncols=True)

    try:
        for batch in pf.iter_batches(batch_size=_BATCH_SIZE):
            df = pa.Table.from_batches([batch]).to_pandas()
            bar.update(len(df))

            for name, spec in active_specs:
                mask   = spec["filter"](df)
                subset = df.loc[mask]
                if subset.empty:
                    continue
                counts[name] += len(subset)

                if name in accs:
                    accs[name].append(subset)
                else:
                    tbl = _df_to_table(subset, schema)
                    if writers[name] is None:
                        out = _OUT_DIR / f"{name}.parquet"
                        writers[name] = pq.ParquetWriter(out, schema, compression="zstd")
                    writers[name].write_table(tbl)
    finally:
        for w in writers.values():
            if w:
                w.close()
        bar.close()

    # Finalise post-processed datasets
    spec_by_name = dict(active_specs)
    for name, chunks in accs.items():
        if not chunks:
            print(f"  {name}: WARNING — no rows collected")
            continue
        full    = pd.concat(chunks, ignore_index=True)
        result  = spec_by_name[name]["post_process"](full)
        counts[name] = len(result)
        out = _OUT_DIR / f"{name}.parquet"
        pq.write_table(_df_to_table(result, schema), out, compression="zstd")

    elapsed  = time.perf_counter() - t0
    manifest = _read_manifest()

    print(f"\n  Results ({elapsed/60:.1f} min):")
    for name, spec in active_specs:
        n_rows = counts[name]
        print(f"    {name:<35}  {n_rows:>12,} rows")
        manifest[name] = {
            "rows":        n_rows,
            "elapsed_s":   round(elapsed, 1),
            "description": spec["description"],
            "path":        str((_OUT_DIR / f"{name}.parquet").relative_to(_SRC)),
        }

    _write_manifest(manifest)
    return counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild full_stream.parquet from scratch before filtering",
    )
    ap.add_argument(
        "--only", type=str, default="",
        help="Comma-separated derived dataset names to (re)write "
             "(skips Phase 1 unless --rebuild is also set; default: all)",
    )
    args = ap.parse_args()

    only: Optional[set] = (
        {n.strip() for n in args.only.split(",") if n.strip()} or None
    )

    specs = _derived_specs()

    if only:
        valid   = {n for n, _ in specs}
        unknown = only - valid
        if unknown:
            print(f"Unknown dataset names: {sorted(unknown)}", file=sys.stderr)
            print(f"Valid names: {sorted(valid)}", file=sys.stderr)
            return 2

    t_start = time.perf_counter()

    if args.rebuild or not _FULL_PQ.exists():
        phase1_stream()

    phase2_filter(specs, names=only)

    total = time.perf_counter() - t_start
    print(f"\nDone in {total/60:.1f} min  --  manifest at {_MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
