"""
ingest/ingest_lst.py
====================
Ingestor: LST + NDVI GeoTIFFs  →  lst.duckdb

Schema (lst table)
------------------
longitude        DOUBLE
latitude         DOUBLE
aster_lst        FLOAT        — ASTER-emissivity LST (nullable)
modis_lst        FLOAT        — MODIS-emissivity LST (nullable)
ndvi             FLOAT        — NDVI vegetation index (nullable)
image_id         VARCHAR
timestamp        VARCHAR      — ISO-8601 "YYYY-MM-DDTHH:MM:SS"
partition_key    VARCHAR      — "YYYY-MM"
tile_id          VARCHAR      — H3 cell at resolution 9
year             INTEGER      — calendar year  (e.g. 2015)
month_of_year    INTEGER      — month 1–12
day_of_month     INTEGER      — day within month 1–31
day_of_year      INTEGER      — day within year 1–366
hour_of_day      FLOAT        — fractional UTC hour  (e.g. 10.5 = 10h 30m)

The five time-component columns are derived once at ingest time so that
catalog histogramming and streaming partition scoring never need to parse
the timestamp string at runtime.  They expose each temporal granularity
as an independent, targetable dimension.
"""

from __future__ import annotations

import logging
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from config import APPEND_BATCH_ROWS, LST_COLUMNS
from db import open_duckdb
from spatial import (
    add_h3,
    get_ghent_convex_hull_polygon,
    get_ghent_exact_polygon,
    iter_raster_blocks_masked,
)

log = logging.getLogger("ingest.lst")

# ============================================================
# LST folder-name parser
# ============================================================
# Folder format: L5_ASTER_20000301_20010301_LT51980242000222FUI00_20000809_101119
#                                                                  ↑date↑  ↑time↑
_FOLDER_RE = re.compile(
    r"^(?P<sat>L\w+)_(?P<product>[A-Z]+)_\d{8}_\d{8}_"
    r"(?P<prod_id>\w+)_(?P<date>\d{8})_(?P<time>\d{6})$"
)


def _parse_lst_folder(name: str) -> Optional[dict]:
    """
    Parse an LST/NDVI folder name into metadata fields.

    All five time-component fields are derived here so that downstream code
    never needs to re-parse the timestamp string:

        year          — calendar year
        month_of_year — month 1–12
        day_of_month  — day within month 1–31
        day_of_year   — day within year 1–366
        hour_of_day   — fractional hour (HH + MM/60 + SS/3600)

    Returns None when the folder name does not match the expected pattern.
    """
    m = _FOLDER_RE.match(name)
    if not m:
        return None

    prod_id = m.group("prod_id")
    d, t    = m.group("date"), m.group("time")

    # Parse date and time components
    yy   = int(d[0:4])
    mo   = int(d[4:6])
    dd   = int(d[6:8])
    hh   = int(t[0:2])
    mm   = int(t[2:4])
    ss   = int(t[4:6])

    # ISO timestamp string kept for display and legacy temporal queries
    timestamp = f"{yy:04d}-{mo:02d}-{dd:02d}T{hh:02d}:{mm:02d}:{ss:02d}"

    # day_of_year via datetime (handles leap years correctly)
    doy = datetime(yy, mo, dd).timetuple().tm_yday

    # Fractional hour: minutes and seconds expressed as a decimal
    hour_of_day = hh + mm / 60.0 + ss / 3600.0

    return {
        "satellite":     m.group("sat"),
        "product":       m.group("product").upper(),  # ASTER | MODIS | NDVI
        "image_id":      prod_id,
        "timestamp":     timestamp,
        "partition_key": timestamp[:7],               # YYYY-MM
        # Time components
        "year":          yy,
        "month_of_year": mo,
        "day_of_month":  dd,
        "day_of_year":   doy,
        "hour_of_day":   hour_of_day,
    }


# ============================================================
# Ingestor
# ============================================================

def ingest_lst(downloads: Path, output: Path) -> int:
    """
    Convert all LST and NDVI TIF folders into a single unified DuckDB table.

    Processing strategy
    -------------------
    1.  Discover all sub-folders under downloads/lst_tifs/ and parse their
        metadata via _parse_lst_folder().
    2.  Group folders by image_id (one image = one acquisition scene).
    3.  For each image_id read ASTER, MODIS, and NDVI TIFs independently,
        clip each to the Ghent polygon, then outer-join on (lon, lat).
    4.  Tag every row with image_id, timestamp, partition_key, tile_id, and
        the five pre-computed time components before appending to DuckDB.
    5.  After all images are processed, sort by (partition_key, tile_id) for
        efficient zone-map pruning during streaming queries.

    Batches of APPEND_BATCH_ROWS are flushed incrementally to keep memory
    usage bounded regardless of dataset size.
    """
    tif_root = downloads / "lst_tifs"
    if not tif_root.exists():
        log.warning("LST TIF root not found: %s — skipping", tif_root)
        return 0

    db_path = output / "lst.duckdb"
    conn    = open_duckdb(db_path)

    CHECKPOINT_EVERY = 50

    conn.execute("SET preserve_insertion_order = false")
    conn.execute(f"SET temp_directory = '{db_path.parent.as_posix()}'")
    conn.execute("SET threads = 4")
    conn.execute("SET memory_limit = '8GB'")

    conn.execute("DROP TABLE IF EXISTS lst")
    conn.execute("""
        CREATE TABLE lst (
            longitude        DOUBLE,
            latitude         DOUBLE,
            aster_lst        FLOAT,
            modis_lst        FLOAT,
            ndvi             FLOAT,
            image_id         VARCHAR,
            timestamp        VARCHAR,
            partition_key    VARCHAR,
            tile_id          VARCHAR,
            year             INTEGER,
            month_of_year    INTEGER,
            day_of_month     INTEGER,
            day_of_year      INTEGER,
            hour_of_day      FLOAT
        )
    """)

    # ---- Collect and group folders by image_id -------------------------
    folders = sorted(p for p in tif_root.iterdir() if p.is_dir())
    tqdm.write(f"LST/NDVI: {len(folders)} source folders found")

    folder_meta: dict[str, dict]           = {}
    image_id_groups: dict[str, list[Path]] = {}

    for folder in folders:
        meta = _parse_lst_folder(folder.name)
        if meta is None:
            continue
        folder_meta[folder.name] = meta
        image_id_groups.setdefault(meta["image_id"], []).append(folder)

    tqdm.write(f"  Grouped into {len(image_id_groups)} unique image_ids")

    log.info("LST: Loading Ghent polygon for filtering...")
    ghent_exact  = get_ghent_exact_polygon()
    ghent_convex = get_ghent_convex_hull_polygon()
    log.info("LST: Polygon filters loaded")

    lst_rows          = 0
    skipped           = 0
    lst_buffer: list[pd.DataFrame] = []
    lst_buf_rows      = 0
    processed_folders: set = set()

    def _flush(buffer: list[pd.DataFrame]) -> None:
        """Sort a batch by (partition_key, tile_id) and append to DuckDB."""
        if not buffer:
            return
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning,
                                    message=".*empty or all-NA entries.*")
            batch = pd.concat(buffer, ignore_index=True)
        buffer.clear()
        batch.sort_values(["partition_key", "tile_id"], inplace=True, ignore_index=True)
        conn.append("lst", batch[LST_COLUMNS])

    # ---- Process each image_id group -----------------------------------
    image_id_list = sorted(image_id_groups.keys())
    for image_id_idx, image_id in enumerate(
        tqdm(image_id_list, desc="Image ID groups", unit="group", smoothing=0.1)
    ):
        folders_for_id = image_id_groups[image_id]

        if all(f in processed_folders for f in folders_for_id):
            continue

        active_folders = [f for f in folders_for_id if f not in processed_folders]
        by_product: dict[str, Path] = {
            folder_meta[f.name]["product"]: f for f in active_folders
        }

        # ---- Read raster data for each emissivity product --------------
        data_by_product: dict[str, pd.DataFrame] = {}

        # ASTER and MODIS — clipped to the exact Ghent boundary
        for product in ("ASTER", "MODIS"):
            if product not in by_product:
                continue
            tif_files = list(by_product[product].glob("*.tif"))
            if not tif_files:
                skipped += 1
                continue
            chunks = [
                chunk for chunk in iter_raster_blocks_masked(
                    tif_files[0], ghent_exact, skip_zeros=True
                )
                if len(chunk) > 0
            ]
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                df = add_h3(df)
                df = df.rename(columns={"value": f"{product.lower()}_lst"})
                data_by_product[product] = df

        # NDVI — clipped to convex hull to avoid boundary artefacts
        if "NDVI" in by_product:
            tif_files = list(by_product["NDVI"].glob("*.tif"))
            if tif_files:
                chunks = []
                for chunk in iter_raster_blocks_masked(
                    tif_files[0], ghent_convex, skip_zeros=False
                ):
                    if len(chunk) == 0:
                        continue
                    chunk = chunk[chunk["value"] != 0.0].reset_index(drop=True)
                    if len(chunk) > 0:
                        chunks.append(chunk)
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                    df = add_h3(df)
                    df = df.rename(columns={"value": "ndvi"})
                    data_by_product["NDVI"] = df

        if not data_by_product:
            for folder in active_folders:
                processed_folders.add(folder)
            continue

        # ---- Outer-join all products on (longitude, latitude) ----------
        merged_df: Optional[pd.DataFrame] = None
        for product, df in data_by_product.items():
            if merged_df is None:
                merged_df = df.copy()
            else:
                value_col = (
                    f"{product.lower()}_lst" if product in ("ASTER", "MODIS") else "ndvi"
                )
                merged_df = merged_df.merge(
                    df[["longitude", "latitude", value_col]],
                    on=["longitude", "latitude"],
                    how="outer",
                )

        # Ensure all LST value columns exist and are float
        for col in ("aster_lst", "modis_lst", "ndvi"):
            if col not in merged_df.columns:
                merged_df[col] = None
            merged_df[col] = merged_df[col].astype("float64", errors="ignore")

        # ---- Attach spatiotemporal metadata ----------------------------
        meta = folder_meta[active_folders[0].name]
        merged_df["image_id"]      = meta["image_id"]
        merged_df["timestamp"]     = meta["timestamp"]
        merged_df["partition_key"] = meta["partition_key"]
        # Pre-computed time components — one scalar per acquisition scene
        merged_df["year"]          = meta["year"]
        merged_df["month_of_year"] = meta["month_of_year"]
        merged_df["day_of_month"]  = meta["day_of_month"]
        merged_df["day_of_year"]   = meta["day_of_year"]
        merged_df["hour_of_day"]   = meta["hour_of_day"]

        # ---- Buffer and flush ------------------------------------------
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning,
                                    message=".*empty or all-NA entries.*")
            lst_buffer.append(merged_df)
        lst_buf_rows += len(merged_df)
        lst_rows     += len(merged_df)

        if lst_buf_rows >= APPEND_BATCH_ROWS:
            _flush(lst_buffer)
            lst_buf_rows = 0

        for folder in active_folders:
            processed_folders.add(folder)

        if (image_id_idx + 1) % CHECKPOINT_EVERY == 0:
            _flush(lst_buffer)
            lst_buf_rows = 0
            conn.execute("CHECKPOINT")

    _flush(lst_buffer)
    conn.execute("CHECKPOINT")

    tqdm.write(f"LST: {lst_rows:,} unified rows | skipped: {skipped}")

    # ---- Global sort for zone-map pruning ------------------------------
    tqdm.write("Sorting lst by (partition_key, tile_id) ...")
    conn.execute("""
        CREATE TABLE lst_sorted AS
        SELECT * FROM lst ORDER BY partition_key, tile_id
    """)
    conn.execute("DROP TABLE lst")
    conn.execute("ALTER TABLE lst_sorted RENAME TO lst")
    conn.execute("CHECKPOINT")
    tqdm.write("  lst sort complete")

    conn.close()
    return lst_rows