"""
ingest.py  -  /src/processing/ingest.py
============================================================
Spatiotemporal feature-store ingestion pipeline - entry point.

Reads all raw source data from a downloads directory and writes the
embedded databases that the streaming stage reads:

    prepared_stream_data/
        lst.duckdb          LST temperature + NDVI  (DuckDB)
        dhm.duckdb          DHM1 + DHM2 elevation points, one table keyed by dhm_year (DuckDB)
        trees.duckdb        Tree inventory           (DuckDB)
        spatial.db          Urban Atlas + WIS polygons (SpatiaLite)
        catalog.duckdb      Dataset registry + histograms (DuckDB)

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
    1. Write an ingest_<n>() function in an ingest/ingest_<n>.py module.
    2. Add an entry to ingest/config.py :: DATASET_REGISTRY.
    3. Call the function in _run_ingestion() below.
    The streaming layer discovers everything from catalog.duckdb - no changes
    needed there.

Requirements
------------
    pip install duckdb rasterio geopandas pyproj shapely h3 tqdm numpy pandas
    SpatiaLite shared library (mod_spatialite) must be on LD_LIBRARY_PATH.
    On Ubuntu/Debian:  apt install libsqlite3-mod-spatialite
    On macOS:          brew install spatialite-tools
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

from catalog import write_catalog
from config import DATASET_REGISTRY, DEFAULT_DOWNLOADS, DEFAULT_OUTPUT
from db import open_duckdb
from ingest_dhm import ingest_dhm
from ingest_lst import ingest_lst
from ingest_spatial import ingest_urban_atlas, ingest_wis
from ingest_trees import ingest_trees

log = logging.getLogger("ingest")


# ============================================================
# Orchestrator
# ============================================================

def _run_ingestion(
    downloads:     Path,
    output:        Path,
    only:          list[str],
    skip_existing: bool,
) -> list[str]:
    processed: list[str] = []

    def _duckdb_has_rows(db_file: str, table: str) -> bool:
        db_path = output / db_file
        if not db_path.exists():
            return False
        try:
            conn = open_duckdb(db_path, read_only=True)
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
        spec  = DATASET_REGISTRY[dataset_id]
        store = spec["store"]
        if store == "duckdb":
            has_data = _duckdb_has_rows(spec["db_file"], spec["table"])
        else:
            has_data = _spatialite_has_rows(spec["table"])
        if has_data:
            log.info("skip-existing: %s already has data - skipping", dataset_id)
            return True
        return False

    def _banner(name: str) -> None:
        log.info("")
        log.info("=" * 60)
        log.info("  %s", name)
        log.info("=" * 60)

    # LST + NDVI share one processor call and one database.
    # Note: NDVI is now integrated into the LST table (3-column schema).
    if any(d in only for d in ("lst", "ndvi")):
        if not _skip("lst"):
            _banner("LST (with ASTER/MODIS/NDVI)  ->  lst.duckdb")
            n = ingest_lst(downloads, output)
            log.info("LST total rows: %d", n)
        if "lst"  in only: processed.append("lst")
        if "ndvi" in only: processed.append("ndvi")

    if "dhm" in only:
        if not _skip("dhm"):
            _banner("DHM1 + DHM2  ->  dhm.duckdb (table `dhm`, keyed by dhm_year)")
            n = ingest_dhm(downloads, output)
            log.info("DHM total rows: %d", n)
        processed.append("dhm")

    if "trees" in only:
        if not _skip("trees"):
            _banner("Trees  ->  trees.duckdb")
            n = ingest_trees(downloads, output)
            log.info("Trees rows: %d", n)
        processed.append("trees")

    if "urban_atlas" in only:
        if not _skip("urban_atlas"):
            _banner("Urban Atlas  ->  spatial.db")
            n = ingest_urban_atlas(downloads, output)
            log.info("Urban Atlas polygon rows: %d", n)
        processed.append("urban_atlas")

    if "wis" in only:
        if not _skip("wis"):
            _banner("WIS  ->  spatial.db")
            n = ingest_wis(downloads, output)
            log.info("WIS polygon rows: %d", n)
        processed.append("wis")

    return processed


# ============================================================
# CLI entry point
# ============================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Spatiotemporal feature-store ingestion pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Skip a dataset if its output database already exists and is non-empty.",
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
