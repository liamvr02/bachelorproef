"""
ingest/ingest_lst.py
====================
Ingestor: LST + NDVI GeoTIFFs  →  lst.duckdb
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from config import (
    APPEND_BATCH_ROWS,
    LST_COLUMNS,
)
from db import open_duckdb
from spatial import (
    add_h3,
    filter_by_polygon,
    get_ghent_convex_hull_polygon,
    get_ghent_exact_polygon,
    iter_raster_blocks,
    iter_raster_blocks_masked,
)

log = logging.getLogger("ingest.lst")

# ============================================================
# LST folder-name parser
# ============================================================
# Format: L5_ASTER_20000301_20010301_LT51980242000222FUI00_20000809_101119
_FOLDER_RE = re.compile(
    r"^(?P<sat>L\w+)_(?P<product>[A-Z]+)_\d{8}_\d{8}_"
    r"(?P<prod_id>\w+)_(?P<date>\d{8})_(?P<time>\d{6})$"
)


def _parse_lst_folder(name: str) -> Optional[dict]:
    """Parse an LST/NDVI folder name into metadata fields."""
    parts = name.split("_")
    if len(parts) < 7:
        return None
    emissivity = parts[1]

    m = _FOLDER_RE.match(name)
    if not m:
        return None
    prod_id   = m.group("prod_id")
    d, t      = m.group("date"), m.group("time")
    timestamp = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}"
    return {
        "satellite":     m.group("sat"),
        "product":       m.group("product").upper(),   # ASTER | MODIS | NDVI
        "landsat_id":    m.group("sat"),
        "emissivity":    emissivity,
        "image_id":      prod_id,
        "timestamp":     timestamp,
        "partition_key": timestamp[:7],                # YYYY-MM
    }


# ============================================================
# Ingestor
# ============================================================

def ingest_lst(downloads: Path, output: Path) -> int:
    """
    Convert all LST and NDVI TIF folders into a single unified DuckDB table.

    Schema (lst):  longitude, latitude, aster_lst, modis_lst, ndvi,
                   image_id, timestamp, partition_key, tile_id

    Processing strategy:
    - Group TIFs by image_id
    - For each image_id, collect all emissivity variants (ASTER, MODIS, NDVI)
    - Read all variants in parallel, merge on (lon, lat) coordinates
    - Create unified rows with aster_lst, modis_lst, ndvi columns (nulls allowed)
    - Track processed TIFs to avoid duplicate work

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
    conn    = open_duckdb(db_path)

    CHECKPOINT_EVERY = 50

    conn.execute("SET preserve_insertion_order = false")
    conn.execute(f"SET temp_directory = '{db_path.parent.as_posix()}'")
    conn.execute("SET threads = 4")
    conn.execute("SET memory_limit = '8GB'")

    conn.execute("DROP TABLE IF EXISTS lst")
    conn.execute("""
        CREATE TABLE lst (
            longitude     DOUBLE,
            latitude      DOUBLE,
            aster_lst     FLOAT,
            modis_lst     FLOAT,
            ndvi          FLOAT,
            image_id      VARCHAR,
            timestamp     VARCHAR,
            partition_key VARCHAR,
            tile_id       VARCHAR
        )
    """)

    # ========== Collect and group folders by image_id ==========
    folders = sorted(p for p in tif_root.iterdir() if p.is_dir())
    tqdm.write(f"LST/NDVI: {len(folders)} source folders found")

    # Parse metadata for all folders
    folder_meta: dict[str, dict] = {}  # folder_name -> metadata
    image_id_groups: dict[str, list[Path]] = {}  # image_id -> list of folder paths

    for folder in folders:
        meta = _parse_lst_folder(folder.name)
        if meta is None:
            continue
        folder_meta[folder.name] = meta
        image_id = meta["image_id"]
        if image_id not in image_id_groups:
            image_id_groups[image_id] = []
        image_id_groups[image_id].append(folder)

    tqdm.write(f"  Grouped into {len(image_id_groups)} unique image_ids")

    log.info("LST: Loading Ghent polygon for filtering...")
    ghent_exact  = get_ghent_exact_polygon()
    ghent_convex = get_ghent_convex_hull_polygon()
    log.info("LST: Polygon filters loaded")

    lst_rows = skipped = 0
    lst_buffer: list[pd.DataFrame] = []
    lst_buf_rows = 0
    processed_folders = set()

    def _flush(buffer: list[pd.DataFrame], table: str,
               cols: list[str], sort_cols: list[str]) -> None:
        if not buffer:
            return
        # Suppress FutureWarning about all-NA columns in concat (occurs with outer joins)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning,
                                  message=".*empty or all-NA entries.*")
            batch = pd.concat(buffer, ignore_index=True)
        buffer.clear()
        batch.sort_values(sort_cols, inplace=True, ignore_index=True)
        conn.append(table, batch[cols])

    # ========== Process by image_id groups ==========
    image_id_list = sorted(image_id_groups.keys())
    for image_id_idx, image_id in enumerate(
        tqdm(image_id_list, desc="Image ID groups", unit="group", smoothing=0.1)
    ):
        folders_for_id = image_id_groups[image_id]

        # Skip if all folders already processed
        if all(f in processed_folders for f in folders_for_id):
            continue

        # Collect active folders (not yet processed)
        active_folders = [f for f in folders_for_id if f not in processed_folders]

        # Group active folders by emissivity product
        by_product: dict[str, Path] = {}  # product -> folder
        for folder in active_folders:
            meta = folder_meta[folder.name]
            product = meta["product"]
            by_product[product] = folder

        # ========== Read all emissivity variants for this image_id ==========
        data_by_product: dict[str, pd.DataFrame] = {}  # product -> DataFrame

        # Read LST products (ASTER, MODIS)
        for product in ["ASTER", "MODIS"]:
            if product not in by_product:
                continue
            folder = by_product[product]
            tif_files = list(folder.glob("*.tif"))
            if not tif_files:
                skipped += 1
                continue

            tif_path = tif_files[0]
            chunks = []
            for chunk in iter_raster_blocks_masked(tif_path, ghent_exact, skip_zeros=True):
                if len(chunk) == 0:
                    continue
                chunks.append(chunk)

            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                df = add_h3(df)
                df = df.rename(columns={"value": f"{product.lower()}_lst"})
                data_by_product[product] = df

        # Read NDVI (uses convex hull, not exact polygon)
        if "NDVI" in by_product:
            folder = by_product["NDVI"]
            tif_files = list(folder.glob("*.tif"))
            if tif_files:
                tif_path = tif_files[0]
                chunks = []
                for chunk in iter_raster_blocks_masked(tif_path, ghent_convex, skip_zeros=False):
                    if len(chunk) == 0:
                        continue
                    # Filter out zeros (clouds)
                    chunk = chunk[chunk["value"] != 0.0].reset_index(drop=True)
                    if len(chunk) == 0:
                        continue
                    chunks.append(chunk)

                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                    df = add_h3(df)
                    df = df.rename(columns={"value": "ndvi"})
                    data_by_product["NDVI"] = df

        # ========== Merge data from all products ==========
        if not data_by_product:
            for folder in active_folders:
                processed_folders.add(folder)
            continue

        # Merge on (longitude, latitude) — outer join to keep all coordinates
        merged_df = None
        for product, df in data_by_product.items():
            if merged_df is None:
                merged_df = df.copy()
            else:
                # Column name depends on product (ASTER/MODIS use "_lst" suffix, NDVI is just "ndvi")
                value_col = f"{product.lower()}_lst" if product in ("ASTER", "MODIS") else "ndvi"
                merge_cols = ["longitude", "latitude", value_col]
                merged_df = merged_df.merge(
                    df[merge_cols],
                    on=["longitude", "latitude"],
                    how="outer"
                )

        # Ensure all columns exist (fill missing with None/null)
        for col in ["aster_lst", "modis_lst", "ndvi"]:
            if col not in merged_df.columns:
                merged_df[col] = None
            # Explicitly ensure float dtype for all LST value columns
            merged_df[col] = merged_df[col].astype("float64", errors="ignore")

        # Add spatiotemp​oral columns
        meta = folder_meta[active_folders[0].name]
        merged_df["image_id"]      = meta["image_id"]
        merged_df["timestamp"]     = meta["timestamp"]
        merged_df["partition_key"] = meta["partition_key"]

        # Buffer for batch append (suppress FutureWarning about all-NA columns in concat)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, 
                                  message=".*empty or all-NA entries.*")
            lst_buffer.append(merged_df)
        lst_buf_rows += len(merged_df)
        lst_rows     += len(merged_df)

        if lst_buf_rows >= APPEND_BATCH_ROWS:
            _flush(lst_buffer, "lst", LST_COLUMNS, ["partition_key", "tile_id"])
            lst_buf_rows = 0

        # Mark all folders for this image_id as processed
        for folder in active_folders:
            processed_folders.add(folder)

        if (image_id_idx + 1) % CHECKPOINT_EVERY == 0:
            _flush(lst_buffer, "lst", LST_COLUMNS, ["partition_key", "tile_id"])
            lst_buf_rows = 0
            conn.execute("CHECKPOINT")

    _flush(lst_buffer, "lst", LST_COLUMNS, ["partition_key", "tile_id"])
    conn.execute("CHECKPOINT")

    tqdm.write(f"LST: {lst_rows:,} unified rows | skipped: {skipped}")

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
