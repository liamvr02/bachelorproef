"""
ingest.py  —  /src/processing/ingest.py
============================================================
Spatiotemporal feature-store ingestion pipeline.

Single entry point.  Reads all raw source data from a downloads directory
and writes two embedded databases that the streaming stage reads:

    prepared_stream_data/
        lst.duckdb          LST temperature + NDVI  (DuckDB)
        spatial.db          DHM elevation + Trees   (SpatiaLite)
        urban_atlas.duckdb  Land-use polygons        (DuckDB)
        catalog.duckdb      Dataset registry         (DuckDB)

Why these formats
-----------------
LST (driving dataset)
    Access pattern: sequential scan with temporal + spatial bbox predicates,
    then full rows yielded to the stream.  DuckDB columnar storage with an
    ART index on (timestamp, longitude, latitude) is optimal.

Urban Atlas
    Access pattern: key join on H3 tile_id after polygons are pre-rasterized
    to cells at ingestion time.  Pure key lookup — DuckDB table with a primary
    key index.  No geometry needed at stream time.

DHM elevation / Trees
    Access pattern: for every LST pixel find the N nearest points or count
    points within radius R.  This is a spatial nearest-neighbour / radius
    query run ~millions of times per session.  SpatiaLite R-tree indexes give
    O(log n) per query vs O(n) linear scan over Parquet.

Distribution-aware sampling
    After ingestion the catalog holds a fixed-bin temperature histogram per
    (partition_key, tile_id) in LST.  The streaming layer reads these once
    on init and uses them to pre-select and weight partitions so the yielded
    rows match a caller-specified DistributionTarget — no parquet scans,
    no full-table reads needed to implement stratified sampling.

Usage
-----
    # All datasets under default downloads/:
    python processing/ingest.py

    # Custom paths or subset:
    python processing/ingest.py --downloads /data/raw --only lst dhm trees

    # Skip already-populated databases:
    python processing/ingest.py --skip-existing

Adding a new dataset
--------------------
    1. Write an ingest_<name>() function following the pattern below.
    2. Add an entry to DATASET_REGISTRY.
    3. Call the function in _run_ingestion().
    The streaming layer discovers everything from catalog.duckdb — no changes
    needed there.

Adding a new feature at stream time
------------------------------------
    If the raw data is already ingested (e.g. you want a new derived column
    from DHM), no re-ingestion is needed — add the computation in the
    streaming layer's enrich step.  If it needs a new source file, follow
    the "adding a dataset" steps above.

Requirements
------------
    pip install duckdb rasterio geopandas pyproj shapely h3 tqdm numpy pandas
    SpatiaLite shared library (mod_spatialite) must be on LD_LIBRARY_PATH.
    On Ubuntu/Debian:  apt install libsqlite3-mod-spatialite
    On macOS:          brew install spatialite-tools
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import duckdb
import h3
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import geopandas as gpd
from pyproj import Transformer
from tqdm import tqdm

log = logging.getLogger("ingest")

# ============================================================
# Paths & constants
# ============================================================

_HERE         = Path(__file__).resolve().parent          # /src/processing/
_SRC          = _HERE.parent                             # /src/
_REPO         = _SRC.parent                              # repo root

DEFAULT_DOWNLOADS = _SRC / "downloads"
DEFAULT_OUTPUT    = _SRC / "prepared_stream_data"

# H3 resolution used for spatial bucketing and Urban Atlas rasterization.
# Resolution 9 ≈ 0.17 km² per cell  (good match for 30 m LST pixels).
H3_RES = 9

# Temperature histogram bin edges stored in the catalog for distribution-
# aware streaming.  2 °C bins from -10 °C to 60 °C  (35 bins).
HIST_EDGES: list[float] = [float(v) for v in np.linspace(-10.0, 60.0, 36)]

# Parquet/DuckDB write settings
CHUNK_ROWS   = 200_000   # rows per DuckDB COPY batch
BLOCK_LIMIT  = None      # set to an int during testing to cap raster blocks

# Belgian Lambert CRS used by DHM rasters
CRS_LAMBERT = "EPSG:31370"
CRS_WGS84   = "EPSG:4326"
CRS_LAEA    = "EPSG:3035"   # equal-area for polygon area computation

_lambert_to_wgs84 = Transformer.from_crs(CRS_LAMBERT, CRS_WGS84, always_xy=True)

# ============================================================
# Dataset registry
# Defines what each dataset is and how it will be used at stream time.
# The catalog.duckdb dataset_metadata table is populated from this dict.
# ============================================================

DATASET_REGISTRY: dict[str, dict] = {
    "lst": {
        "description":         "LST — Land Surface Temperature (Landsat 5, ASTER+MODIS, 30 m)",
        "db_file":             "lst.duckdb",
        "table":               "lst",
        "store":               "duckdb",
        "feature_columns":     ["temperature", "emissivity", "landsat_id", "image_id"],
        "value_column":        "temperature",    # column used for histograms
        "lookup_method":       "driving",        # this is the driving dataset
        "temporal_behavior":   "evolving",
        "partition_column":    "partition_key",  # YYYY-MM monthly bucket
        "source_resolution_m": 30.0,
        "is_driving":          True,
    },
    "ndvi": {
        "description":         "NDVI — Vegetation index (Landsat 5, 30 m)",
        "db_file":             "lst.duckdb",     # shares the LST database
        "table":               "ndvi",
        "store":               "duckdb",
        "feature_columns":     ["ndvi"],
        "value_column":        "ndvi",
        "lookup_method":       "nearest_spatiotemporal",
        "temporal_behavior":   "evolving",
        "partition_column":    "partition_key",
        "source_resolution_m": 30.0,
        "is_driving":          False,
    },
    "dhm1": {
        "description":         "DHM1 — Digital Height Model 1 (~2007, 5 m, Belgian Lambert)",
        "db_file":             "spatial.db",
        "table":               "dhm1",
        "store":               "spatialite",
        "feature_columns":     ["elevation"],
        "value_column":        "elevation",
        "lookup_method":       "nearest_spatial",
        "temporal_behavior":   "static",
        "partition_column":    None,
        "source_resolution_m": 5.0,
        "is_driving":          False,
    },
    "dhm2": {
        "description":         "DHM2 — Digital Height Model 2 (~2015, 5 m, Belgian Lambert)",
        "db_file":             "spatial.db",
        "table":               "dhm2",
        "store":               "spatialite",
        "feature_columns":     ["elevation"],
        "value_column":        "elevation",
        "lookup_method":       "nearest_spatial",
        "temporal_behavior":   "static",
        "partition_column":    None,
        "source_resolution_m": 5.0,
        "is_driving":          False,
    },
    "trees": {
        "description":         "Trees — Point-level tree inventory (CSV, WGS-84)",
        "db_file":             "spatial.db",
        "table":               "trees",
        "store":               "spatialite",
        "feature_columns":     ["species", "height_m", "planting_year", "trunk_diameter_cm"],
        "value_column":        None,
        "lookup_method":       "radius_spatial",
        "temporal_behavior":   "static",
        "partition_column":    None,
        "source_resolution_m": None,
        "is_driving":          False,
    },
    "urban_atlas": {
        "description":         "Urban Atlas — Land use/cover 2006/2012/2018/2021 (vector → H3)",
        "db_file":             "urban_atlas.duckdb",
        "table":               "urban_atlas",
        "store":               "duckdb",
        "feature_columns":     ["luc_code", "ua_year", "area_m2"],
        "value_column":        None,
        "lookup_method":       "tile_join",      # join on (tile_id, ua_year)
        "temporal_behavior":   "static",
        "partition_column":    None,
        "source_resolution_m": None,
        "is_driving":          False,
    },
}


# ============================================================
# Helpers — database connections
# ============================================================

def _duckdb(path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, creating the file if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def _spatialite(path: Path) -> sqlite3.Connection:
    """
    Open a SpatiaLite connection, loading the mod_spatialite extension.

    Tries several common library names across Linux / macOS / Windows.
    Raises RuntimeError if the extension cannot be loaded.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)

    lib_candidates = [
        "mod_spatialite",          # Linux (on LD_LIBRARY_PATH)
        "mod_spatialite.so",
        "mod_spatialite.dylib",    # macOS
        "/usr/lib/x86_64-linux-gnu/mod_spatialite.so",
        "/usr/local/lib/mod_spatialite.dylib",
    ]
    loaded = False
    for lib in lib_candidates:
        try:
            conn.load_extension(lib)
            loaded = True
            break
        except sqlite3.OperationalError:
            continue

    if not loaded:
        raise RuntimeError(
            "Could not load mod_spatialite.  Install it with:\n"
            "  Ubuntu/Debian: sudo apt install libsqlite3-mod-spatialite\n"
            "  macOS:         brew install spatialite-tools\n"
            "Then ensure the library is on LD_LIBRARY_PATH / DYLD_LIBRARY_PATH."
        )

    # Only call InitSpatialMetaData when the metadata tables don't exist yet.
    # Calling it on an existing database prints a C-level error string even
    # when passing 1 (the "if not exists" flag), because the flag only
    # suppresses the exception — the C code still logs to stderr.
    already_init = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='spatial_ref_sys'"
    ).fetchone()[0]
    if not already_init:
        conn.execute("SELECT InitSpatialMetaData(1)")
    conn.commit()
    return conn


# ============================================================
# Helpers — raster reading
# ============================================================

def _iter_raster_blocks(
    tif_path: Path,
    value_dtype=np.float32,
    skip_zeros: bool = True,
    already_wgs84: bool = True,
    transformer: Optional[Transformer] = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream a GeoTIFF one block at a time, yielding DataFrames with columns:
        longitude, latitude, value

    Memory usage is bounded to one raster block regardless of file size.

    Parameters
    ----------
    tif_path      : path to the GeoTIFF
    value_dtype   : numpy dtype to cast raster values to
    skip_zeros    : drop pixels where value == 0
    already_wgs84 : if True, pixel centres are used directly as lon/lat
    transformer   : pyproj.Transformer for reprojection (when not WGS-84)
    """
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        block_iter = list(src.block_windows(1))
        if BLOCK_LIMIT is not None:
            block_iter = block_iter[:BLOCK_LIMIT]

        for _, window in block_iter:
            data = src.read(1, window=window).astype(value_dtype)
            transform = src.window_transform(window)
            h, w = data.shape

            row_idx, col_idx = np.mgrid[0:h, 0:w]
            xs, ys = rasterio.transform.xy(
                transform, row_idx.ravel(), col_idx.ravel(), offset="center"
            )
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
            vals = data.ravel()

            # Build validity mask
            mask = ~np.isnan(vals)
            if nodata is not None:
                mask &= vals != nodata
            if skip_zeros:
                mask &= vals != 0.0

            if not mask.any():
                continue

            xs, ys, vals = xs[mask], ys[mask], vals[mask]

            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)   # → (lon, lat)

            yield pd.DataFrame(
                {"longitude": xs, "latitude": ys, "value": vals}
            )


# ============================================================
# Helpers — H3 indexing (vectorised)
# ============================================================

def _add_h3(df: pd.DataFrame, resolution: int = H3_RES) -> pd.DataFrame:
    """Add a tile_id (H3 cell string) column.  Vectorised via numpy + list-comp."""
    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()
    df["tile_id"] = [
        h3.latlng_to_cell(float(lat), float(lon), resolution)
        for lat, lon in zip(lats, lons)
    ]
    return df


# ============================================================
# Helpers — histogram
# ============================================================

def _histogram(values: np.ndarray, edges: list[float]) -> dict:
    clean = values[~np.isnan(values)]
    counts, _ = np.histogram(clean, bins=edges)
    return {"edges": edges, "counts": counts.tolist()}


# ============================================================
# LST folder-name parser
# ============================================================
# Format: L5_ASTER_20000301_20010301_LT51980242000222FUI00_20000809_101119
#          sat  product  qstart   qend      prod_id            date     time
_FOLDER_RE = re.compile(
    r"^(?P<sat>L\w+)_(?P<product>[A-Z]+)_\d{8}_\d{8}_"
    r"(?P<prod_id>\w+)_(?P<date>\d{8})_(?P<time>\d{6})$"
)


def _parse_lst_folder(name: str) -> Optional[dict]:
    m = _FOLDER_RE.match(name)
    if not m:
        return None
    prod_id   = m.group("prod_id")
    emissivity = prod_id[-6:-2] if len(prod_id) >= 6 else prod_id
    d, t      = m.group("date"), m.group("time")
    timestamp = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}"
    return {
        "satellite":  m.group("sat"),
        "product":    m.group("product").upper(),   # ASTER | MODIS | NDVI
        "landsat_id": m.group("sat"),
        "emissivity": emissivity,
        "image_id":   prod_id,
        "timestamp":  timestamp,
        "partition_key": timestamp[:7],             # YYYY-MM
    }


# ============================================================
# Urban Atlas source file registry
# To add a new UA year, add one entry here.
# ============================================================

UA_SOURCES: dict[int, dict] = {
    2006: {
        "globs":          ["BE003L2_GENT/*/Shapefiles/*.shp"],
        # UA 2006 shapefile uses CODE2006 or the numeric DN/GRIDCODE field
        "luc_candidates": ["CODE2006", "CODE_2006", "UA_2006", "DN", "GRIDCODE", "CODE"],
        "layer":          None,
    },
    2012: {
        "globs":          ["BE003L2_GENT_UA2012_revised_v021/*/Data/*.gpkg"],
        # GPKG has multiple layers; the land-use layer name is the first (default) one.
        # Specify it explicitly to suppress pyogrio's multi-layer warning.
        "luc_candidates": ["CODE2012", "CODE_2012", "UA_2012", "LC_CODE"],
        "layer":          "BE003L2_GENT_UA2012_revised",
    },
    2018: {
        "globs":          ["*/CLMS_UA_LCU_S2018*/*.fgb"],
        "luc_candidates": ["CODE2018", "CODE_2018", "LC_CODE", "CLASS"],
        "layer":          None,
    },
    2021: {
        "globs":          ["*/CLMS_UA_LCU_S2021*/*.fgb"],
        "luc_candidates": ["CODE2021", "CODE_2021", "LC_CODE", "CLASS"],
        "layer":          None,
    },
}


# Column order — single source of truth for conn.append() (positional mapping).
LST_COLUMNS  = ["longitude", "latitude", "temperature", "emissivity",
                 "landsat_id", "image_id", "timestamp", "partition_key", "tile_id"]
NDVI_COLUMNS = ["longitude", "latitude", "ndvi", "image_id",
                 "timestamp", "partition_key", "tile_id"]

APPEND_BATCH_ROWS = 1_000_000   # rows buffered before each conn.append()


# ============================================================
# Ingestor: LST + NDVI  →  lst.duckdb
# ============================================================

def ingest_lst(downloads: Path, output: Path) -> int:
    """
    Convert all LST and NDVI TIF folders to DuckDB tables.

    Schema (lst):  longitude, latitude, temperature, emissivity,
                   landsat_id, image_id, timestamp, partition_key, tile_id
    Schema (ndvi): longitude, latitude, ndvi, image_id,
                   timestamp, partition_key, tile_id

    No ART indexes — DuckDB zone maps on a sorted table are equivalent at
    zero memory cost.  ART indexes on 700 M rows OOM on 12 GB machines.

    Autocommit throughout (no BEGIN/COMMIT): CHECKPOINT is silently ignored
    inside an open transaction, so we never open one.  Each conn.append()
    call completes a mini-transaction immediately.

    Batches of APPEND_BATCH_ROWS are sorted by (partition_key, tile_id)
    before appending so row-groups are locally sorted.  A final ORDER BY
    produces a globally sorted replacement table; DuckDB spills to
    temp_directory so it cannot OOM.
    """
    tif_root = downloads / "lst_tifs"
    if not tif_root.exists():
        log.warning("LST TIF root not found: %s — skipping", tif_root)
        return 0

    db_path = output / "lst.duckdb"
    conn    = _duckdb(db_path)

    CHECKPOINT_EVERY = 50

    conn.execute("SET preserve_insertion_order = false")
    conn.execute(f"SET temp_directory = '{db_path.parent}'")
    conn.execute("SET threads = 4")
    conn.execute("SET memory_limit = '8GB'")

    # Idempotent: drop existing tables so a re-run starts clean.
    conn.execute("DROP TABLE IF EXISTS lst")
    conn.execute("DROP TABLE IF EXISTS ndvi")
    conn.execute("""
        CREATE TABLE lst (
            longitude     DOUBLE,
            latitude      DOUBLE,
            temperature   FLOAT,
            emissivity    VARCHAR,
            landsat_id    VARCHAR,
            image_id      VARCHAR,
            timestamp     VARCHAR,
            partition_key VARCHAR,
            tile_id       VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE ndvi (
            longitude     DOUBLE,
            latitude      DOUBLE,
            ndvi          FLOAT,
            image_id      VARCHAR,
            timestamp     VARCHAR,
            partition_key VARCHAR,
            tile_id       VARCHAR
        )
    """)

    folders = sorted(p for p in tif_root.iterdir() if p.is_dir())
    tqdm.write(f"LST/NDVI: {len(folders)} source folders found")

    lst_rows = ndvi_rows = skipped = 0
    lst_buffer:  list[pd.DataFrame] = []
    ndvi_buffer: list[pd.DataFrame] = []
    lst_buf_rows = ndvi_buf_rows = 0

    def _flush(buffer: list[pd.DataFrame], table: str,
               cols: list[str], sort_cols: list[str]) -> None:
        if not buffer:
            return
        batch = pd.concat(buffer, ignore_index=True)
        buffer.clear()
        batch.sort_values(sort_cols, inplace=True, ignore_index=True)
        conn.append(table, batch[cols])

    for folder_idx, folder in enumerate(
        tqdm(folders, desc="LST/NDVI folders", unit="folder", smoothing=0.1)
    ):
        meta = _parse_lst_folder(folder.name)
        if meta is None:
            skipped += 1
            continue

        tif_files = list(folder.glob("*.tif"))
        if not tif_files:
            skipped += 1
            continue

        tif_path = tif_files[0]
        product  = meta["product"]

        if product == "NDVI":
            for chunk in _iter_raster_blocks(tif_path, skip_zeros=False):
                chunk = _add_h3(chunk)
                chunk["ndvi"]          = chunk.pop("value")
                chunk["image_id"]      = meta["image_id"]
                chunk["timestamp"]     = meta["timestamp"]
                chunk["partition_key"] = meta["partition_key"]
                ndvi_buffer.append(chunk)
                ndvi_buf_rows += len(chunk)
                ndvi_rows     += len(chunk)
                if ndvi_buf_rows >= APPEND_BATCH_ROWS:
                    _flush(ndvi_buffer, "ndvi", NDVI_COLUMNS,
                           ["partition_key", "tile_id"])
                    ndvi_buf_rows = 0
        else:
            for chunk in _iter_raster_blocks(tif_path, skip_zeros=True):
                chunk = _add_h3(chunk)
                chunk["temperature"]   = chunk.pop("value")
                chunk["emissivity"]    = meta["emissivity"]
                chunk["landsat_id"]    = meta["landsat_id"]
                chunk["image_id"]      = meta["image_id"]
                chunk["timestamp"]     = meta["timestamp"]
                chunk["partition_key"] = meta["partition_key"]
                lst_buffer.append(chunk)
                lst_buf_rows += len(chunk)
                lst_rows     += len(chunk)
                if lst_buf_rows >= APPEND_BATCH_ROWS:
                    _flush(lst_buffer, "lst", LST_COLUMNS,
                           ["partition_key", "tile_id"])
                    lst_buf_rows = 0

        # Flush residual buffer content then checkpoint.
        # No open transaction — CHECKPOINT needs autocommit mode.
        if (folder_idx + 1) % CHECKPOINT_EVERY == 0:
            _flush(lst_buffer,  "lst",  LST_COLUMNS,  ["partition_key", "tile_id"])
            _flush(ndvi_buffer, "ndvi", NDVI_COLUMNS, ["partition_key", "tile_id"])
            lst_buf_rows = ndvi_buf_rows = 0
            conn.execute("CHECKPOINT")

    _flush(lst_buffer,  "lst",  LST_COLUMNS,  ["partition_key", "tile_id"])
    _flush(ndvi_buffer, "ndvi", NDVI_COLUMNS, ["partition_key", "tile_id"])
    conn.execute("CHECKPOINT")

    tqdm.write(f"LST: {lst_rows:,} rows | NDVI: {ndvi_rows:,} rows | skipped: {skipped}")

    for table in ("lst", "ndvi"):
        tqdm.write(f"Sorting {table} by (partition_key, tile_id) ...")
        conn.execute(f"""
            CREATE TABLE {table}_sorted AS
            SELECT * FROM {table} ORDER BY partition_key, tile_id
        """)
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f"ALTER TABLE {table}_sorted RENAME TO {table}")
        conn.execute("CHECKPOINT")
        tqdm.write(f"  {table} sort complete")

    conn.close()
    return lst_rows + ndvi_rows


# ============================================================
# Ingestor: DHM (DHM1 + DHM2)  →  spatial.db  (SpatiaLite)
# ============================================================

def _drop_spatialite_table(conn: sqlite3.Connection, table: str) -> None:
    """
    Safely drop a SpatiaLite table and its associated R-tree index.

    Checks sqlite_master and geometry_columns before calling SpatiaLite
    functions so no C-level error strings are printed on a fresh database.
    """
    # Check whether a spatial index exists for this table before disabling it.
    # SpatiaLite prints a C-level error string even inside a try/except when
    # DisableSpatialIndex is called on a table that has no index.
    has_index = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type='table' AND name=?",
        (f"idx_{table}_geom",)
    ).fetchone()[0]
    if has_index:
        conn.execute(f"SELECT DisableSpatialIndex('{table}', 'geom')")
        conn.execute(f"DROP TABLE IF EXISTS idx_{table}_geom")

    conn.execute(f"DROP TABLE IF EXISTS {table}")

    # Remove the entry from geometry_columns so AddGeometryColumn works cleanly.
    conn.execute(
        "DELETE FROM geometry_columns WHERE f_table_name = ?", (table,)
    )
    conn.commit()


def _init_spatialite_elevation_table(conn: sqlite3.Connection, table: str) -> None:
    """Create an elevation table with a SpatiaLite geometry column and R-tree.
    Idempotent: drops cleanly before recreating."""
    _drop_spatialite_table(conn, table)
    conn.execute(f"""
        CREATE TABLE {table} (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            elevation   REAL    NOT NULL,
            source      TEXT    NOT NULL
        )
    """)
    conn.execute(
        f"SELECT AddGeometryColumn('{table}', 'geom', 4326, 'POINT', 'XY')"
    )
    conn.execute(
        f"SELECT CreateSpatialIndex('{table}', 'geom')"
    )
    conn.commit()


def _insert_elevation_chunk(
    conn: sqlite3.Connection,
    table: str,
    chunk: pd.DataFrame,
    source: str,
) -> None:
    """Bulk-insert one elevation chunk using a prepared statement."""
    rows = [
        (float(row.longitude), float(row.latitude), float(row.elevation), source)
        for row in chunk.itertuples(index=False)
    ]
    conn.executemany(
        f"""
        INSERT INTO {table}(elevation, source, geom)
        VALUES (?, ?, MakePoint(?, ?, 4326))
        """,
        # reorder: elevation, source, lon, lat
        [(elev, src, lon, lat) for lon, lat, elev, src in rows],
    )


def ingest_dhm(downloads: Path, output: Path) -> int:
    """
    Convert DHM1 and DHM2 TIFs to SpatiaLite R-tree-indexed tables.

    Schema (dhm1 / dhm2 tables):
        id INTEGER PK, elevation REAL, source TEXT, geom POINT(WGS-84)
        + SpatiaLite virtual R-tree index on geom

    The streaming layer queries:
        SELECT elevation FROM dhm1
        WHERE id IN (
            SELECT id FROM SpatialIndex
            WHERE f_table_name='dhm1'
              AND search_frame=BuildCircleMbr(lon, lat, radius_deg)
        )
        ORDER BY Distance(geom, MakePoint(lon, lat, 4326))
        LIMIT k
    """
    sources = {
        "DHM1": (downloads / "DHM1_extracted", "dhm1"),
        "DHM2": (downloads / "DHM2_extracted", "dhm2"),
    }

    db_path = output / "spatial.db"
    conn    = _spatialite(db_path)
    total   = 0

    for source_label, (src_dir, table) in sources.items():
        if not src_dir.exists():
            log.warning("%s source directory not found: %s — skipping",
                        source_label, src_dir)
            continue

        tif_files = sorted(src_dir.rglob("*.tif"))
        if not tif_files:
            log.warning("No TIF files in %s — skipping", src_dir)
            continue

        log.info("%s: %d TIF file(s)", source_label, len(tif_files))
        _init_spatialite_elevation_table(conn, table)

        rows_written = 0
        for tif_path in tqdm(tif_files, desc=source_label, unit="file"):
            # Count blocks upfront so the nested bar has a known total.
            with rasterio.open(tif_path) as _src:
                n_blocks = len(list(_src.block_windows(1)))
            with tqdm(total=n_blocks, desc=tif_path.name, unit="block",
                      leave=False) as block_bar:
                for chunk in _iter_raster_blocks(
                    tif_path,
                    skip_zeros=True,
                    already_wgs84=False,
                    transformer=_lambert_to_wgs84,
                ):
                    chunk = chunk.rename(columns={"value": "elevation"})
                    _insert_elevation_chunk(conn, table, chunk, source_label)
                    rows_written += len(chunk)
                    block_bar.update(1)

                    if rows_written % CHUNK_ROWS == 0:
                        conn.commit()

        conn.commit()
        log.info("%s: %d rows written", source_label, rows_written)
        total += rows_written

    conn.close()
    return total


# ============================================================
# Ingestor: Trees (CSV)  →  spatial.db  (SpatiaLite)
# ============================================================

# Candidate column names for lon/lat/species in the CSV
_TREE_LON_COLS     = ["longitude", "lon", "x", "lng"]
_TREE_LAT_COLS     = ["latitude",  "lat", "y"]
_TREE_SPECIES_COLS = ["species", "soort", "boomsoort", "naam"]
_TREE_HEIGHT_COLS  = ["height", "hoogte", "height_m", "kroonhoogte"]
_TREE_YEAR_COLS    = ["planting_year", "plantjaar", "jaar"]
_TREE_DIAM_COLS    = ["trunk_diameter_cm", "stamomtrek", "diameter", "omtrek"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def _init_spatialite_trees_table(conn: sqlite3.Connection) -> None:
    """Create the trees table with R-tree index.  Idempotent: drops cleanly first."""
    _drop_spatialite_table(conn, "trees")
    conn.execute("""
        CREATE TABLE trees (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            species             TEXT,
            height_m            TEXT,
            planting_year       INTEGER,
            trunk_diameter_cm   REAL
        )
    """)
    conn.execute("SELECT AddGeometryColumn('trees', 'geom', 4326, 'POINT', 'XY')")
    conn.execute("SELECT CreateSpatialIndex('trees', 'geom')")
    conn.commit()


def ingest_trees(downloads: Path, output: Path) -> int:
    """
    Load all trees CSV files into the SpatiaLite spatial.db.

    Handles both comma- and semicolon-delimited files.
    Handles the 'geo_point_2d' combined coordinate column.
    Tree height is stored as raw text (e.g. '<6 m.', '6-9 m.').

    Schema:
        id INTEGER PK, species TEXT, height_m TEXT,
        planting_year INTEGER, trunk_diameter_cm REAL,
        geom POINT(WGS-84)
        + SpatiaLite virtual R-tree index on geom
    """
    # Collect all candidate files across all glob patterns then deduplicate.
    # Filenames are timestamped as ghent_trees_YYYY-MM-DD.csv so lexicographic
    # sort gives chronological order — take only the last (most recent) file.
    csv_candidates = (
        list(downloads.rglob("ghent_trees_*.csv"))
    )
    csv_candidates = list(dict.fromkeys(csv_candidates))   # deduplicate, preserve order
    if not csv_candidates:
        log.warning("No trees CSV files found under %s — skipping", downloads)
        return 0

    csv_files = [max(csv_candidates, key=lambda p: p.name)]
    log.info("Trees: using latest file: %s  (%d candidate(s) found)",
             csv_files[0].name, len(csv_candidates))

    db_path = output / "spatial.db"
    # Use _spatialite() so InitSpatialMetaData is always called before
    # AddGeometryColumn — this is what makes the geom column actually exist.
    conn = _spatialite(db_path)

    _init_spatialite_trees_table(conn)

    total = 0
    for csv_path in tqdm(csv_files, desc="Trees CSVs", unit="file"):
        # Auto-detect delimiter
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","

        df = pd.read_csv(csv_path, delimiter=delimiter, low_memory=False)

        # Handle geo_point_2d column ("lat, lon" string)
        if "geo_point_2d" in df.columns:
            coords = df["geo_point_2d"].str.split(r",\s*", expand=True)
            df["latitude"]  = pd.to_numeric(coords[0], errors="coerce")
            df["longitude"] = pd.to_numeric(coords[1], errors="coerce")

        lon_col = _find_col(df, _TREE_LON_COLS)
        lat_col = _find_col(df, _TREE_LAT_COLS)
        if lon_col is None or lat_col is None:
            log.warning("Cannot find lon/lat columns in %s — skipping", csv_path.name)
            continue

        df = df.dropna(subset=[lon_col, lat_col])
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df = df.dropna(subset=[lon_col, lat_col])

        species_col = _find_col(df, _TREE_SPECIES_COLS)
        height_col  = _find_col(df, _TREE_HEIGHT_COLS)
        year_col    = _find_col(df, _TREE_YEAR_COLS)
        diam_col    = _find_col(df, _TREE_DIAM_COLS)

        rows = []
        for row in df.itertuples(index=False):
            lon = float(getattr(row, lon_col))
            lat = float(getattr(row, lat_col))
            rows.append((
                str(getattr(row, species_col, "") or "")   if species_col else None,
                str(getattr(row, height_col,  "") or "")   if height_col  else None,
                int(getattr(row, year_col,    0)  or 0)    if year_col    else None,
                float(getattr(row, diam_col,  0)  or 0)    if diam_col    else None,
                lon,
                lat,
            ))

        conn.executemany(
            """
            INSERT INTO trees(species, height_m, planting_year, trunk_diameter_cm, geom)
            VALUES (?, ?, ?, ?, MakePoint(?, ?, 4326))
            """,
            rows,
        )
        conn.commit()
        total += len(rows)
        log.info("Trees %s: %d rows", csv_path.name, len(rows))

    conn.close()
    return total


# ============================================================
# Ingestor: Urban Atlas  →  urban_atlas.duckdb
# ============================================================

def _find_ua_file(year: int, base_dir: Path) -> Optional[Path]:
    for glob in UA_SOURCES[year]["globs"]:
        candidates = list(base_dir.glob(glob))
        if candidates:
            return candidates[0]
    return None


def _normalise_luc(gdf: gpd.GeoDataFrame, candidates: list[str]) -> gpd.GeoDataFrame:
    """Rename the land-use code column to 'luc_code', matching case-insensitively."""
    lower_to_actual = {c.lower(): c for c in gdf.columns}
    for cand in candidates:
        actual = lower_to_actual.get(cand.lower())
        if actual is not None:
            return gdf.rename(columns={actual: "luc_code"})
    non_geom = [c for c in gdf.columns if c != "geometry"]
    log.warning(
        "No luc_code candidate matched %s.  Available columns: %s.  "
        "Using '%s' as fallback — add the correct name to UA_SOURCES[year]['luc_candidates'].",
        candidates, list(gdf.columns), non_geom[0] if non_geom else "none",
    )
    if non_geom:
        return gdf.rename(columns={non_geom[0]: "luc_code"})
    return gdf


# h3-py v3 uses h3.polyfill_geojson(); v4 renamed it to h3.geo_to_cells().
# This wrapper tries the v4 name first and falls back to v3 so the code works
# with whichever version is installed.
def _h3_cells_from_geojson(geojson: dict, resolution: int) -> set:
    if hasattr(h3, "geo_to_cells"):          # h3-py >= 4
        return set(h3.geo_to_cells(geojson, resolution))
    return set(h3.polyfill_geojson(geojson, resolution))  # h3-py 3


def _polygon_to_h3_rows(
    geom,        # shapely geometry, already WGS-84
    luc_code: str,
    ua_year: int,
    area_m2: float,
) -> list[dict]:
    """
    Map one polygon to all H3 cells it covers at H3_RES.

    Interior fill via polyfill_geojson, boundary cells via 1-ring expansion
    + centroid containment test.
    """
    from shapely.geometry import mapping, Point

    geojson          = mapping(geom)
    interior: set    = _h3_cells_from_geojson(geojson, H3_RES)
    candidates: set  = set(interior)
    for cell in list(interior):
        candidates.update(h3.grid_disk(cell, 1))

    rows = []
    for cell in candidates:
        lat_c, lon_c = h3.cell_to_latlng(cell)
        if geom.contains(Point(lon_c, lat_c)):
            rows.append({
                "tile_id":  cell,
                "luc_code": str(luc_code),
                "ua_year":  int(ua_year),
                "area_m2":  float(area_m2),
            })
    return rows


def ingest_urban_atlas(downloads: Path, output: Path) -> int:
    """
    Convert Urban Atlas vector files to a DuckDB key-join table.

    Polygons are rasterized to H3 cells at ingestion time.  At stream time
    the lookup is a pure hash join on (tile_id, ua_year) — no geometry.

    Schema:
        tile_id VARCHAR, luc_code VARCHAR, ua_year SMALLINT, area_m2 FLOAT
    Index:
        PRIMARY KEY (tile_id, ua_year)
    """
    ua_root = downloads / "urban_atlas_extracted"
    if not ua_root.exists():
        log.warning("Urban Atlas source directory not found: %s — skipping", ua_root)
        return 0

    db_path = output / "urban_atlas.duckdb"
    conn    = _duckdb(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS urban_atlas (
            tile_id   VARCHAR,
            luc_code  VARCHAR,
            ua_year   SMALLINT,
            area_m2   FLOAT
        )
    """)

    total = 0

    for year in sorted(UA_SOURCES.keys()):
        source = _find_ua_file(year, ua_root)
        if source is None:
            log.warning("Urban Atlas %d: no file found — skipping", year)
            continue

        # Idempotent: remove any previously ingested rows for this year
        # so re-running a single year doesn't produce duplicates.
        existing = conn.execute(
            "SELECT COUNT(*) FROM urban_atlas WHERE ua_year = ?", [year]
        ).fetchone()[0]
        if existing:
            log.info("Urban Atlas %d: removing %d existing rows before re-ingesting",
                     year, existing)
            conn.execute("DELETE FROM urban_atlas WHERE ua_year = ?", [year])

        log.info("Urban Atlas %d: loading %s ...", year, source.name)
        layer = UA_SOURCES[year].get("layer")
        read_kwargs = {"layer": layer} if layer is not None else {}
        gdf = gpd.read_file(str(source), **read_kwargs)
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()

        # Compute area in equal-area CRS before reprojecting
        gdf_ea       = gdf.to_crs(CRS_LAEA)
        gdf["area_m2"] = gdf_ea.geometry.area.values

        # Reproject to WGS-84 for H3
        gdf = gdf.to_crs(CRS_WGS84)
        gdf = _normalise_luc(gdf, UA_SOURCES[year]["luc_candidates"])

        if "luc_code" not in gdf.columns:
            log.error("Urban Atlas %d: cannot identify luc_code column — skipping", year)
            continue

        log.info("  %d polygons → expanding to H3 cells ...", len(gdf))
        batch: list[dict] = []
        rows_year = 0

        for row in tqdm(gdf.itertuples(), total=len(gdf), desc=f"UA {year} H3"):
            h3_rows = _polygon_to_h3_rows(
                geom     = row.geometry,
                luc_code = getattr(row, "luc_code", "unknown"),
                ua_year  = year,
                area_m2  = getattr(row, "area_m2", 0.0),
            )
            batch.extend(h3_rows)
            rows_year += len(h3_rows)

            # Flush batch to DuckDB
            if len(batch) >= CHUNK_ROWS:
                chunk_df = pd.DataFrame(batch)
                conn.execute("""
                    INSERT INTO urban_atlas (tile_id, luc_code, ua_year, area_m2)
                    SELECT tile_id, luc_code, ua_year, area_m2
                    FROM chunk_df
                """)
                batch = []

        if batch:
            chunk_df = pd.DataFrame(batch)
            conn.execute("""
                INSERT INTO urban_atlas (tile_id, luc_code, ua_year, area_m2)
                SELECT tile_id, luc_code, ua_year, area_m2
                FROM chunk_df
            """)

        log.info("  Urban Atlas %d: %d H3 rows", year, rows_year)
        total += rows_year

    log.info("Building Urban Atlas index ...")
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ua_tile_year
        ON urban_atlas(tile_id, ua_year)
    """)

    conn.close()
    return total


# ============================================================
# Catalog  →  catalog.duckdb
# ============================================================



def write_catalog(output: Path, processed: list[str]) -> None:
    """
    Write (or update) catalog.duckdb with dataset metadata and LST histograms.

    Tables
    ------
    dataset_metadata      one row per logical dataset
    partition_statistics  one row per (dataset, partition_key, tile_id)
                          histogram_json stores fixed-bin temperature counts
    histogram_config      shared bin edges read once by the streaming layer

    All aggregation runs inside DuckDB via ATTACH — no temperature values
    or large result sets cross into Python.  The 1.2 M-row partition_statistics
    table is written with a single bulk INSERT INTO ... SELECT.
    """
    db_path = output / "catalog.duckdb"
    conn    = _duckdb(db_path)

    conn.execute("DROP TABLE IF EXISTS dataset_metadata")
    conn.execute("DROP TABLE IF EXISTS partition_statistics")
    conn.execute("DROP TABLE IF EXISTS histogram_config")

    conn.execute("""
        CREATE TABLE dataset_metadata (
            dataset_id              VARCHAR PRIMARY KEY,
            description             VARCHAR,
            feature_columns         VARCHAR[],
            lookup_method           VARCHAR,
            temporal_behavior       VARCHAR,
            partition_column        VARCHAR,
            source_resolution_m     DOUBLE,
            is_driving              BOOLEAN,
            value_column            VARCHAR,
            db_file                 VARCHAR,
            db_table                VARCHAR,
            store                   VARCHAR,
            registered_at           VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE partition_statistics (
            dataset_id      VARCHAR,
            partition_key   VARCHAR,
            tile_id         VARCHAR,
            row_count       BIGINT,
            value_min       DOUBLE,
            value_max       DOUBLE,
            value_mean      DOUBLE,
            histogram_json  VARCHAR,
            PRIMARY KEY (dataset_id, partition_key, tile_id)
        )
    """)
    conn.execute("""
        CREATE TABLE histogram_config (
            dataset_id      VARCHAR PRIMARY KEY,
            bin_edges_json  VARCHAR,
            n_bins          INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO histogram_config VALUES (?, ?, ?)",
        ["lst", json.dumps(HIST_EDGES), len(HIST_EDGES) - 1],
    )

    now = datetime.now(timezone.utc).isoformat()
    for dataset_id in tqdm(processed, desc="Registering datasets", unit="dataset"):
        spec = DATASET_REGISTRY[dataset_id]
        conn.execute(
            "INSERT OR REPLACE INTO dataset_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                dataset_id, spec["description"], spec["feature_columns"],
                spec["lookup_method"], spec["temporal_behavior"],
                spec.get("partition_column"), spec.get("source_resolution_m"),
                spec["is_driving"], spec.get("value_column"),
                spec["db_file"], spec["table"], spec["store"], now,
            ],
        )

    # --- LST partition histograms ---
    # Everything runs inside DuckDB — no Python loops, no fetchall, no executemany.
    #
    # ATTACH lst.duckdb into catalog.duckdb so both share one DuckDB process.
    # Two full-table scans over lst (unavoidable for exact stats and bin counts),
    # then a single INSERT INTO partition_statistics from a CTE chain:
    #   1. part_stats  — COUNT/MIN/MAX/AVG per (partition_key, tile_id)
    #   2. bin_counts  — sparse bin counts via GREATEST/LEAST/FLOOR
    #   3. bin_maps    — MAP aggregate: {bin_idx -> count} per partition
    #   4. histograms  — dense array via list_transform + range, zero-filled
    #   5. Final SELECT joins part_stats + histograms, serialises JSON with concat
    #
    # No Python touches the 700 M temperature values or the 1.2 M result rows.
    if "lst" in processed:
        lst_db = output / "lst.duckdb"
        if lst_db.exists():
            n_bins    = len(HIST_EDGES) - 1            # 35
            bin_width = HIST_EDGES[1] - HIST_EDGES[0]  # 2.0
            edge_min  = HIST_EDGES[0]                  # -10.0
            edges_json = json.dumps(HIST_EDGES)

            # Build the histogram_json column expression outside the f-string
            # to avoid brace-escaping issues.  The result looks like:
            #   {"edges":[-10.0,...,60.0],"counts":[0,12,847,...]}
            hist_json_expr = (
                "concat('{\"edges\":', '" + edges_json + "', "
                "',\"counts\":', to_json(h.counts), '}')"
            )

            tqdm.write("Computing LST partition histograms (pure SQL) ...")
            conn.execute(f"ATTACH '{lst_db}' AS lst_db (READ_ONLY)")
            conn.execute(f"""
                INSERT INTO partition_statistics
                WITH
                part_stats AS (
                    SELECT
                        partition_key,
                        tile_id,
                        COUNT(*)         AS row_count,
                        MIN(temperature) AS t_min,
                        MAX(temperature) AS t_max,
                        AVG(temperature) AS t_mean
                    FROM lst_db.lst
                    GROUP BY partition_key, tile_id
                ),
                bin_counts AS (
                    SELECT
                        partition_key,
                        tile_id,
                        GREATEST(0, LEAST({n_bins - 1},
                            FLOOR((temperature - ({edge_min})) / {bin_width})::INTEGER
                        )) AS bin_idx,
                        COUNT(*) AS bin_count
                    FROM lst_db.lst
                    GROUP BY partition_key, tile_id, bin_idx
                ),
                bin_maps AS (
                    SELECT
                        partition_key,
                        tile_id,
                        MAP(list(bin_idx), list(bin_count)) AS m
                    FROM bin_counts
                    GROUP BY partition_key, tile_id
                ),
                histograms AS (
                    SELECT
                        partition_key,
                        tile_id,
                        list_transform(
                            range(0, {n_bins}),
                            i -> coalesce(m[i]::BIGINT, 0)
                        ) AS counts
                    FROM bin_maps
                )
                SELECT
                    'lst'              AS dataset_id,
                    s.partition_key,
                    s.tile_id,
                    s.row_count,
                    s.t_min            AS value_min,
                    s.t_max            AS value_max,
                    s.t_mean           AS value_mean,
                    {hist_json_expr}   AS histogram_json
                FROM part_stats  s
                JOIN histograms  h
                  ON s.partition_key = h.partition_key
                 AND s.tile_id       = h.tile_id
            """)
            conn.execute("DETACH lst_db")
            tqdm.write("Histogram catalog complete.")

    conn.close()
    tqdm.write(f"Catalog written to {db_path}")


# ============================================================
# Main orchestrator
# ============================================================

def _run_ingestion(
    downloads:     Path,
    output:        Path,
    only:          list[str],
    skip_existing: bool,
) -> list[str]:
    processed: list[str] = []

    def _duckdb_has_rows(db_file: str, table: str) -> bool:
        """Return True if the DuckDB table exists and has at least one row."""
        db_path = output / db_file
        if not db_path.exists():
            return False
        try:
            conn = _duckdb(db_path, read_only=True)
            exists = conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables "
                f"WHERE table_name = '{table}'"
            ).fetchone()[0]
            if not exists:
                conn.close()
                return False
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            conn.close()
            return n > 0
        except Exception:
            return False

    def _spatialite_has_rows(table: str) -> bool:
        """Return True if the SpatiaLite table exists and has at least one row."""
        db_path = output / "spatial.db"
        if not db_path.exists():
            return False
        try:
            conn = sqlite3.connect(str(db_path))
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            if table not in tables:
                conn.close()
                return False
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            conn.close()
            return n > 0
        except Exception:
            return False

    def _skip(dataset_id: str) -> bool:
        if not skip_existing:
            return False
        spec = DATASET_REGISTRY[dataset_id]
        store = spec["store"]
        db_file = spec["db_file"]
        table = spec["table"]
        if store == "duckdb":
            has_data = _duckdb_has_rows(db_file, table)
        else:
            has_data = _spatialite_has_rows(table)
        if has_data:
            log.info("skip-existing: %s already has data — skipping", dataset_id)
            return True
        return False

    def _banner(name: str) -> None:
        log.info("")
        log.info("=" * 60)
        log.info("  %s", name)
        log.info("=" * 60)

    # LST + NDVI share one processor call and one database.
    # Skip only if BOTH tables have data.
    if any(d in only for d in ("lst", "ndvi")):
        lst_has_data  = _skip("lst")
        ndvi_has_data = _skip("ndvi")
        if not (lst_has_data and ndvi_has_data):
            _banner("LST + NDVI  →  lst.duckdb")
            n = ingest_lst(downloads, output)
            log.info("LST+NDVI total rows: %d", n)
        if "lst"  in only: processed.append("lst")
        if "ndvi" in only: processed.append("ndvi")

    if any(d in only for d in ("dhm1", "dhm2")):
        dhm1_skip = _skip("dhm1")
        dhm2_skip = _skip("dhm2")
        if not (dhm1_skip and dhm2_skip):
            _banner("DHM1 + DHM2  →  spatial.db")
            n = ingest_dhm(downloads, output)
            log.info("DHM total rows: %d", n)
        if "dhm1" in only: processed.append("dhm1")
        if "dhm2" in only: processed.append("dhm2")

    if "trees" in only:
        if not _skip("trees"):
            _banner("Trees  →  spatial.db")
            n = ingest_trees(downloads, output)
            log.info("Trees rows: %d", n)
        processed.append("trees")

    if "urban_atlas" in only:
        if not _skip("urban_atlas"):
            _banner("Urban Atlas  →  urban_atlas.duckdb")
            n = ingest_urban_atlas(downloads, output)
            log.info("Urban Atlas H3 rows: %d", n)
        processed.append("urban_atlas")

    return processed


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Spatiotemporal feature-store ingestion pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1].split("Adding")[0].strip(),
    )
    parser.add_argument(
        "--downloads",
        type=Path,
        default=DEFAULT_DOWNLOADS,
        help=f"Raw data root directory  (default: {DEFAULT_DOWNLOADS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Prepared data output directory  (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(DATASET_REGISTRY.keys()),
        default=list(DATASET_REGISTRY.keys()),
        metavar="DATASET",
        help=(
            "Process only these dataset(s).  Choices: "
            + ", ".join(DATASET_REGISTRY.keys())
            + "  (default: all)"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a dataset if its output database file already exists and is non-empty.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("INGESTION PIPELINE")
    log.info("=" * 60)
    log.info("  Downloads : %s", args.downloads)
    log.info("  Output    : %s", args.output)
    log.info("  Datasets  : %s", args.only)
    log.info("  Skip exist: %s", args.skip_existing)

    processed = _run_ingestion(
        downloads     = args.downloads,
        output        = args.output,
        only          = args.only,
        skip_existing = args.skip_existing,
    )

    if not processed:
        log.info("No datasets processed.")
        return

    log.info("")
    log.info("=" * 60)
    log.info("Writing catalog ...")
    log.info("=" * 60)
    write_catalog(args.output, processed)

    log.info("")
    log.info("=" * 60)
    log.info("Ingestion complete.  Processed: %s", processed)
    log.info("Output: %s", args.output)
    log.info("=" * 60)


if __name__ == "__main__":
    main()