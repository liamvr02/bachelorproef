"""
migrate_tile_columns.py
=======================
Add alternative tile columns to an already-ingested lst.duckdb
without re-running the full ingest pipeline.

Strategy:
  1. Read distinct (longitude, latitude) pairs from lst
  2. Compute tile columns via add_tile_columns()
  3. Register the lookup as a DuckDB view
  4. CREATE TABLE lst_migrated AS SELECT ... FROM lst LEFT JOIN lookup
  5. Swap tables

Run from:  src/processing/
Usage:     python migrate_tile_columns.py [--output PATH] [--ngi PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from db import open_duckdb
from spatial import add_tile_columns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("migrate_tile_columns")

NEW_TILE_COLS = ["tile_h3_r8", "tile_h3_r7", "tile_rect_1km", "tile_rect_2km", "tile_ngi"]

# Column order to preserve after migration (must match LST_COLUMNS in config.py).
_ORDERED_COLS = [
    "longitude", "latitude",
    "aster_lst", "modis_lst", "ndvi",
    "image_id", "timestamp", "partition_key", "tile_id",
    "tile_h3_r8", "tile_h3_r7", "tile_rect_1km", "tile_rect_2km", "tile_ngi",
    "year", "month_of_year", "day_of_month", "day_of_year", "hour_of_day",
]


def migrate(db_path: Path, ngi_shapefile: Path | None = None) -> None:
    ngi_gdf = None
    if ngi_shapefile is not None:
        import geopandas as gpd
        log.info("Loading NGI shapefile: %s", ngi_shapefile)
        ngi_gdf = gpd.read_file(ngi_shapefile)

    conn = open_duckdb(db_path)
    conn.execute(f"SET temp_directory = '{db_path.parent.as_posix()}'")
    conn.execute("SET memory_limit = '8GB'")
    conn.execute("SET threads = 4")

    existing_cols = {row[0] for row in conn.execute("DESCRIBE lst").fetchall()}
    already_done = all(c in existing_cols for c in NEW_TILE_COLS)
    if already_done:
        log.info("All tile columns already present — nothing to do.")
        conn.close()
        return

    log.info("Loading distinct (longitude, latitude) pairs...")
    coords = conn.execute("SELECT DISTINCT longitude, latitude FROM lst").df()
    log.info("  %d distinct coordinate pairs", len(coords))

    log.info("Computing tile columns...")
    coords = add_tile_columns(coords, ngi_gdf=ngi_gdf)
    tile_lookup = coords[["longitude", "latitude"] + NEW_TILE_COLS]

    conn.register("_tile_lookup", tile_lookup)

    select_old = ", ".join(
        f"lst.{c}"
        for c in _ORDERED_COLS
        if c not in NEW_TILE_COLS
    )
    select_new = ", ".join(f"lu.{c}" for c in NEW_TILE_COLS)

    log.info("Building lst_migrated table (this may take a while)...")
    conn.execute(f"""
        CREATE TABLE lst_migrated AS
        SELECT {select_old}, {select_new}
        FROM lst
        LEFT JOIN _tile_lookup lu
            ON (lst.longitude = lu.longitude AND lst.latitude = lu.latitude)
        ORDER BY lst.partition_key, lst.tile_id
    """)

    conn.execute("DROP TABLE lst")
    conn.execute("ALTER TABLE lst_migrated RENAME TO lst")
    conn.execute("CHECKPOINT")
    log.info("Migration complete: %s", db_path)
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Add tile columns to existing lst.duckdb")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "prepared_stream_data",
        help="Directory containing lst.duckdb (default: src/prepared_stream_data)",
    )
    parser.add_argument(
        "--ngi",
        type=Path,
        default=None,
        help="Path to NGI Kbl.shp shapefile for tile_ngi (column used: CODE)",
    )
    args = parser.parse_args()

    db_path = args.output / "lst.duckdb"
    if not db_path.exists():
        log.error("lst.duckdb not found at %s", db_path)
        sys.exit(1)

    migrate(db_path, ngi_shapefile=args.ngi)


if __name__ == "__main__":
    main()
