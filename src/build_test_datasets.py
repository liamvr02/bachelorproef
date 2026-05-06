"""
build_test_datasets.py
======================

Materialise the data-science test datasets defined in the spec to
``src/test_data/*.parquet``.  Each dataset is a single-pass stream of LST
rows + computed features (canonical registry from ``stream_configs.registry``)
saved to its own parquet file.  A manifest at ``src/test_data/manifest.json``
records row counts and the configs used.

Datasets
--------
  full_representative   uniform-distribution stream over non-outlier months
  outlier_<label> x 9   one stream per OUTLIER_RANGES label, natural density
  hour_midday           rows where hour_of_day in [11.5, 12.5]
  hour_midnight         rows where hour_of_day in [-0.5, 0.5]  (UTC midnight)
  point_a_50m           rows within ~50 m of  51 deg02'44.5"N 3 deg43'54.6"E
  point_b_50m           rows within ~50 m of  51 deg03'14.4"N 3 deg43'19.2"E
  single_image          one (tile_id, timestamp) -- the dominant scene of a day
  single_year_2017      uniform-distribution rows from 2017 only

Usage
-----
    python src/build_test_datasets.py            # build only what's missing
    python src/build_test_datasets.py --rebuild  # rebuild everything
    python src/build_test_datasets.py --only point_a_50m,single_image

Each dataset writes incrementally via ``pyarrow.parquet.ParquetWriter`` so
memory stays bounded by ``batch_size`` rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from tqdm.auto import tqdm

from stream.features import FeatureRegistry
from stream.logging_config import configure_logging
from stream.stream import StreamConfig
from stream_configs import (
    OUTLIER_RANGES,
    PointFilterStream,
    build_registry,
    outlier_configs,
    outlier_keys,
    point_filter,
    representative,
)

configure_logging(level="INFO")


_SRC       = Path(__file__).parent
_OUT_DIR   = _SRC / "test_data"
_MANIFEST  = _OUT_DIR / "manifest.json"

# Two reference points -- DMS -> decimal degrees.
_POINT_A = (3 + 43/60 + 54.6/3600,  51 + 2/60 + 44.5/3600)   # (lon, lat)
_POINT_B = (3 + 43/60 + 19.2/3600,  51 + 3/60 + 14.4/3600)

# 50 m tolerance: at lat 51 degN, 50 m ~ 0.00045 deg lat, 0.00072 deg lon (cos 51 deg ~ 0.629).
# Use the looser bound on both axes so the bounding box circumscribes the 50 m
# circle.  A few rows just past 50 m get included; statistically negligible.
_POINT_TOL_DEG = 0.00072

# Single-image probe: a calendar day picked to land on a Landsat-8 daytime pass
# over Ghent.  The actual image_id is selected at collection time (dominant
# (tile_id, timestamp) within the day's rows).  2017 chosen to avoid every
# OUTLIER_RANGES period.
_SINGLE_IMAGE_DAY = (2017, 7, 15)

_BATCH_SIZE = 100_000


# ---------------------------------------------------------------------------
# Stream -> parquet (with all-NaN column promotion)
# ---------------------------------------------------------------------------

def _promote_null_columns(tbl):
    """
    PyArrow infers ``null`` type when an entire batch's column is all-NaN
    (e.g. ``aster_lst`` empty in some partitions).  ParquetWriter then refuses
    to write later batches whose column has the inferred numeric type.  Promote
    every ``null``-typed column to float64 so subsequent batches stay
    compatible with the writer's schema.
    """
    import pyarrow as pa

    new_fields = []
    new_arrays = []
    for i, field in enumerate(tbl.schema):
        col = tbl.column(i)
        if pa.types.is_null(field.type):
            new_fields.append(pa.field(field.name, pa.float64(), nullable=True))
            new_arrays.append(col.cast(pa.float64()))
        else:
            new_fields.append(field)
            new_arrays.append(col)
    return pa.Table.from_arrays(new_arrays, schema=pa.schema(new_fields))


def _stream_to_parquet(
    cfg:        Union[StreamConfig, PointFilterStream],
    reg:        FeatureRegistry,
    max_rows:   Optional[int],
    batch_size: int,
    path:       Path,
    label:      str,
) -> int:
    """
    Stream rows from *cfg* and write to a parquet file at *path*.

    *max_rows* caps the underlying stream's input scan, not the output row
    count.  For non-filtering configs the two are equal; for PointFilterStream
    the output is whatever survives the predicate.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if path.exists():
        path.unlink()

    writer: Optional[pq.ParquetWriter] = None
    n_total = 0
    bar = tqdm(total=max_rows, desc=f"build:{label}", unit="row",
               dynamic_ncols=True, position=0)
    try:
        for df in cfg.stream(reg, batch_size=batch_size, max_rows=max_rows):
            if df.empty:
                continue
            tbl = _promote_null_columns(pa.Table.from_pandas(df, preserve_index=False))
            if writer is None:
                writer = pq.ParquetWriter(path, tbl.schema, compression="zstd")
            else:
                tbl = tbl.cast(writer.schema)
            writer.write_table(tbl)
            n_total += len(df)
            bar.update(len(df))
    finally:
        if writer is not None:
            writer.close()
        bar.close()
    return n_total


# ---------------------------------------------------------------------------
# Single-image collector -- two-pass-in-memory selection
# ---------------------------------------------------------------------------

def _build_single_image(reg: FeatureRegistry, path: Path) -> int:
    """
    Stream one calendar day's rows, identify the most-populated
    (tile_id, timestamp) pair, and write only those rows.

    The whole day's data fits comfortably in memory (a few hundred MB at most),
    so a single in-memory pass is fine.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    y, m, d = _SINGLE_IMAGE_DAY
    cfg = point_filter(year=y, month=m, day=d, batch_size=_BATCH_SIZE)

    chunks: List[pd.DataFrame] = []
    bar = tqdm(desc=f"build:single_image[{y}-{m:02d}-{d:02d}]",
               unit="row", dynamic_ncols=True, position=0)
    try:
        # 5_000_000 input cap is generous -- one day rarely yields that much
        # post-filter, but keeps the run bounded if the partition is huge.
        for df in cfg.stream(reg, batch_size=_BATCH_SIZE, max_rows=5_000_000):
            if df.empty:
                continue
            chunks.append(df)
            bar.update(len(df))
    finally:
        bar.close()

    if not chunks:
        print(f"[single_image] no rows for {y}-{m:02d}-{d:02d}; skipping")
        return 0

    full = pd.concat(chunks, ignore_index=True)
    grouped = (full.groupby(["tile_id", "timestamp"], dropna=False)
                   .size()
                   .sort_values(ascending=False))
    top_tile, top_ts = grouped.index[0]
    sub = full[(full["tile_id"] == top_tile) & (full["timestamp"] == top_ts)].copy()
    print(f"[single_image] dominant scene: tile_id={top_tile!r} timestamp={top_ts!r} "
          f"({len(sub):,}/{len(full):,} day rows)")

    if path.exists():
        path.unlink()
    tbl = _promote_null_columns(pa.Table.from_pandas(sub, preserve_index=False))
    pq.write_table(tbl, path, compression="zstd")
    return len(sub)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def _outlier_dataset_specs(batch_size: int) -> List[Tuple[str, dict]]:
    """One spec per OUTLIER_RANGES label."""
    cfgs = outlier_configs(batch_size=batch_size)
    return [
        (
            f"outlier_{label}",
            {
                "kind":        "stream",
                "builder":     (lambda c=cfg: c),
                "max_rows":    5_000_000,
                "description": f"natural-density rows from {label} months",
            },
        )
        for label, cfg in cfgs.items()
    ]


def _all_specs() -> List[Tuple[str, dict]]:
    """
    Return [(name, spec)] for every dataset.  Each spec is a dict with:
      - kind         : "stream" | "single_image"
      - builder      : callable returning StreamConfig | PointFilterStream
                       (only for kind="stream")
      - max_rows     : input-scan cap (only for kind="stream")
      - description  : human-readable
    """
    excluded, _ = outlier_keys()

    specs: List[Tuple[str, dict]] = [
        ("full_representative", {
            "kind":        "stream",
            "builder":     lambda: representative(
                excluded_keys=excluded, batch_size=_BATCH_SIZE,
            ),
            "max_rows":    5_000_000,
            "description": "uniform year/month/hour distribution, outlier months excluded",
        }),
    ]

    specs.extend(_outlier_dataset_specs(_BATCH_SIZE))

    specs.extend([
        ("hour_midday", {
            "kind":        "stream",
            "builder":     lambda: point_filter(hour=12.0, batch_size=_BATCH_SIZE),
            "max_rows":    5_000_000,   # ~1/24 pass the predicate
            "description": "rows in hour_of_day in [11.5, 12.5]",
        }),
        ("hour_midnight", {
            "kind":        "stream",
            "builder":     lambda: point_filter(hour=0.0, batch_size=_BATCH_SIZE),
            "max_rows":    5_000_000,
            "description": "rows in hour_of_day in [-0.5, 0.5]",
        }),
        ("point_a_50m", {
            "kind":        "stream",
            "builder":     lambda: point_filter(
                lon=_POINT_A[0], lat=_POINT_A[1],
                tol_deg=_POINT_TOL_DEG, batch_size=_BATCH_SIZE,
            ),
            "max_rows":    5_000_000,
            "description": f"within ~50 m of {_POINT_A} (lon, lat)",
        }),
        ("point_b_50m", {
            "kind":        "stream",
            "builder":     lambda: point_filter(
                lon=_POINT_B[0], lat=_POINT_B[1],
                tol_deg=_POINT_TOL_DEG, batch_size=_BATCH_SIZE,
            ),
            "max_rows":    5_000_000,
            "description": f"within ~50 m of {_POINT_B} (lon, lat)",
        }),
        ("single_image", {
            "kind":        "single_image",
            "description": (f"dominant (tile_id, timestamp) within "
                            f"{_SINGLE_IMAGE_DAY[0]}-"
                            f"{_SINGLE_IMAGE_DAY[1]:02d}-"
                            f"{_SINGLE_IMAGE_DAY[2]:02d}"),
        }),
        ("single_year_2017", {
            "kind":        "stream",
            "builder":     lambda: point_filter(year=2017, batch_size=_BATCH_SIZE),
            "max_rows":    5_000_000,
            "description": "all available 2017 rows, year-pre-filtered",
        }),
    ])

    return specs


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _read_manifest() -> Dict[str, dict]:
    if _MANIFEST.exists():
        try:
            return json.loads(_MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_manifest(data: Dict[str, dict]) -> None:
    _MANIFEST.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild every dataset, ignoring existing parquet files")
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated list of dataset names to build "
                             "(default: all)")
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    only = {n.strip() for n in args.only.split(",") if n.strip()}
    specs = _all_specs()
    if only:
        unknown = only - {name for name, _ in specs}
        if unknown:
            print(f"unknown dataset names: {sorted(unknown)}", file=sys.stderr)
            return 2
        specs = [(n, s) for n, s in specs if n in only]

    reg = build_registry()
    n_features = len(reg._descriptors)
    print(f"registry: {n_features} feature descriptors")

    manifest = _read_manifest()
    t_start = time.perf_counter()

    for name, spec in specs:
        out_path = _OUT_DIR / f"{name}.parquet"
        if out_path.exists() and not args.rebuild:
            print(f"[skip] {name} already exists ({out_path})")
            continue

        print(f"\n=== {name} ===")
        print(f"    {spec['description']}")
        t0 = time.perf_counter()

        if spec["kind"] == "stream":
            stream_obj = spec["builder"]()
            n_rows = _stream_to_parquet(
                cfg=stream_obj, reg=reg, max_rows=spec["max_rows"],
                batch_size=_BATCH_SIZE, path=out_path, label=name,
            )
        elif spec["kind"] == "single_image":
            n_rows = _build_single_image(reg, out_path)
        else:
            raise ValueError(f"unknown spec kind: {spec['kind']}")

        elapsed = time.perf_counter() - t0
        print(f"    -> {n_rows:,} rows in {elapsed:.1f}s ({n_rows / max(elapsed, 1e-6):.0f} rows/s)")

        manifest[name] = {
            "rows":         n_rows,
            "elapsed_s":    round(elapsed, 1),
            "description":  spec["description"],
            "max_rows":     spec.get("max_rows"),
            "path":         str(out_path.relative_to(_SRC)),
        }
        _write_manifest(manifest)

    total = time.perf_counter() - t_start
    print(f"\nall datasets done in {total / 60:.1f} min -- manifest at {_MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
