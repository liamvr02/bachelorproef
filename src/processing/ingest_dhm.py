"""
ingest/ingest_dhm.py
====================
Ingestor: DHM1 + DHM2 elevation rasters  ->  dhm.duckdb (single table `dhm`).

Both survey periods live in one table, distinguished by a `dhm_year` column,
mirroring how Urban Atlas stores multiple survey years side-by-side with
`ua_year`.  The stream layer picks the appropriate survey at query time via
a last-previous filter on `dhm_year`.

Survey year assignment
----------------------
DHM1 was acquired 2001-2004  -> dhm_year = 2001
DHM2 was acquired 2013-2015  -> dhm_year = 2013

The stream-side cutoff (LST rows < 2013 use DHM1, >= 2013 use DHM2) is
expressed directly as `dhm_year <= EXTRACT(YEAR FROM lst_timestamp)`.
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
    Convert DHM1 and DHM2 TIFs into a single DuckDB file - no SpatiaLite, no GEOS.

    Rows are stored sorted by (dhm_year, latitude, longitude) so DuckDB's
    zone-map pruning handles both bbox queries and year filters without an
    explicit index.

    Schema:
        longitude  DOUBLE   WGS-84
        latitude   DOUBLE   WGS-84
        elevation  FLOAT    metres above sea level
        is_water   BOOLEAN  True if value was -9999 (invalid/water)
        dhm_year   SMALLINT survey start year (2001 for DHM1, 2013 for DHM2)
    """
    sources = {
        "DHM1": (downloads / "DHM1_extracted", 2001),
        "DHM2": (downloads / "DHM2_extracted", 2013),
    }

    table   = "dhm"
    db_path = output / "dhm.duckdb"

    log.info("DHM: Loading Ghent convex hull polygon for filtering...")
    ghent_convex = get_ghent_convex_hull_polygon()
    log.info("DHM: Polygon filter loaded")

    conn = open_duckdb(db_path)
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"""
        CREATE TABLE {table} (
            longitude  DOUBLE   NOT NULL,
            latitude   DOUBLE   NOT NULL,
            elevation  FLOAT    NOT NULL,
            is_water   BOOLEAN  NOT NULL,
            dhm_year   SMALLINT NOT NULL
        )
    """)
    conn.execute("SET preserve_insertion_order = false")
    conn.execute(f"SET temp_directory = '{db_path.parent.as_posix()}'")
    conn.execute("SET memory_limit = '4GB'")

    total = 0

    for source_label, (src_dir, dhm_year) in sources.items():
        if not src_dir.exists():
            log.warning("%s source directory not found: %s - skipping", source_label, src_dir)
            continue

        tif_files = sorted(src_dir.rglob("*.tif"))
        if not tif_files:
            log.warning("No TIF files in %s - skipping", src_dir)
            continue

        log.info("%s: %d TIF file(s) -> table %s (dhm_year=%d)",
                 source_label, len(tif_files), table, dhm_year)

        with rasterio.open(tif_files[0]) as first_src:
            log.info("%s: First TIF CRS: %s, bounds: %s",
                     source_label, first_src.crs, first_src.bounds)

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
                if len(chunk) == 0:
                    log.debug(f"{tif_path.name}: chunk filtered to 0 rows")
                    continue
                chunk = chunk.rename(columns={"value": "elevation"})
                chunk["is_water"]  = chunk["elevation"] == -9999.0
                chunk["elevation"] = chunk["elevation"].replace(-9999.0, 0.0)
                chunk["dhm_year"]  = dhm_year
                buffer.append(chunk[["longitude", "latitude",
                                     "elevation", "is_water", "dhm_year"]])
                buf_rows     += len(chunk)
                rows_written += len(chunk)

                if buf_rows >= CHUNK_ROWS:
                    conn.append(table, pd.concat(buffer, ignore_index=True))
                    buffer.clear()
                    buf_rows = 0

        if buffer:
            conn.append(table, pd.concat(buffer, ignore_index=True))
            buffer.clear()

        log.info("%s: %d rows appended (dhm_year=%d)",
                 source_label, rows_written, dhm_year)
        total += rows_written

    tqdm.write(f"DHM: sorting by (dhm_year, latitude, longitude) ...")
    conn.execute(f"""
        CREATE TABLE {table}_sorted AS
        SELECT * FROM {table} ORDER BY dhm_year, latitude, longitude
    """)
    conn.execute(f"DROP TABLE {table}")
    conn.execute(f"ALTER TABLE {table}_sorted RENAME TO {table}")
    conn.execute("CHECKPOINT")
    conn.close()

    log.info("DHM total: %d rows written to %s", total, db_path.name)
    return total
