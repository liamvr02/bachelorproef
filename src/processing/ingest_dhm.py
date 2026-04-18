"""
ingest/ingest_dhm.py
====================
Ingestor: DHM1 + DHM2 elevation rasters  →  dhm1.duckdb / dhm2.duckdb
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import rasterio
from tqdm import tqdm

from config import CHUNK_ROWS, _lambert_to_wgs84
from db import open_duckdb
from spatial import get_ghent_convex_hull_polygon, iter_raster_blocks_masked

log = logging.getLogger("ingest.dhm")


def ingest_dhm(downloads: Path, output: Path) -> int:
    """
    Convert DHM1 and DHM2 TIFs to separate DuckDB files — no SpatiaLite, no GEOS.

    Rows are stored sorted by (latitude, longitude) so DuckDB's zone-map
    pruning handles bbox queries without any explicit index.

    Schema (dhm1 / dhm2):
        longitude  DOUBLE  WGS-84
        latitude   DOUBLE  WGS-84
        elevation  FLOAT   metres above sea level
        is_water   BOOLEAN True if value was -9999 (invalid/water), False otherwise
    """
    sources = {
        "DHM1": (downloads / "DHM1_extracted", "dhm1", output / "dhm1.duckdb"),
        "DHM2": (downloads / "DHM2_extracted", "dhm2", output / "dhm2.duckdb"),
    }

    log.info("DHM: Loading Ghent convex hull polygon for filtering...")
    ghent_convex = get_ghent_convex_hull_polygon()
    log.info("DHM: Polygon filter loaded")

    total = 0

    for source_label, (src_dir, table, db_path) in sources.items():
        if not src_dir.exists():
            log.warning("%s source directory not found: %s — skipping", source_label, src_dir)
            continue

        tif_files = sorted(src_dir.rglob("*.tif"))
        if not tif_files:
            log.warning("No TIF files in %s — skipping", src_dir)
            continue

        log.info("%s: %d TIF file(s) → %s", source_label, len(tif_files), db_path.name)
        
        # Verify CRS of first file for debugging
        with rasterio.open(tif_files[0]) as first_src:
            log.info("%s: First TIF CRS: %s, bounds: %s", source_label, first_src.crs, first_src.bounds)
        conn = open_duckdb(db_path)
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"""
            CREATE TABLE {table} (
                longitude  DOUBLE NOT NULL,
                latitude   DOUBLE NOT NULL,
                elevation  FLOAT  NOT NULL,
                is_water   BOOLEAN NOT NULL
            )
        """)
        conn.execute("SET preserve_insertion_order = false")
        conn.execute(f"SET temp_directory = '{db_path.parent.as_posix()}'")
        conn.execute("SET memory_limit = '4GB'")

        rows_written = 0
        buffer:   list[pd.DataFrame] = []
        buf_rows  = 0

        for tif_path in tqdm(tif_files, desc=source_label, unit="file"):
            for chunk in iter_raster_blocks_masked(
                tif_path,
                ghent_convex,
                skip_zeros=False,
                already_wgs84=False,
                transformer=_lambert_to_wgs84,
            ):
                # Already clipped to convex hull via rasterio.mask at GDAL level
                if len(chunk) == 0:
                    log.debug(f"{tif_path.name}: chunk filtered to 0 rows")
                    continue
                chunk = chunk.rename(columns={"value": "elevation"})
                # Track which values are water (were -9999), then convert to 0
                chunk["is_water"] = chunk["elevation"] == -9999.0
                chunk["elevation"] = chunk["elevation"].replace(-9999.0, 0.0)
                buffer.append(chunk[["longitude", "latitude", "elevation", "is_water"]])
                buf_rows     += len(chunk)
                rows_written += len(chunk)

                if buf_rows >= CHUNK_ROWS:
                    conn.append(table, pd.concat(buffer, ignore_index=True))
                    buffer.clear()
                    buf_rows = 0

        if buffer:
            conn.append(table, pd.concat(buffer, ignore_index=True))

        tqdm.write(f"{source_label}: sorting by (latitude, longitude) ...")
        conn.execute(f"""
            CREATE TABLE {table}_sorted AS
            SELECT * FROM {table} ORDER BY latitude, longitude
        """)
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f"ALTER TABLE {table}_sorted RENAME TO {table}")
        conn.execute("CHECKPOINT")
        conn.close()

        log.info("%s: %d rows written to %s", source_label, rows_written, db_path.name)
        total += rows_written

    return total
