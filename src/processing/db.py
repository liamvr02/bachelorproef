"""
ingest/db.py
============
Database connection helpers for DuckDB and SpatiaLite.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import duckdb

log = logging.getLogger("ingest.db")


def open_duckdb(path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, creating the file and parent directories if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def open_spatialite(path: Path) -> sqlite3.Connection:
    """
    Open a SpatiaLite connection, loading the mod_spatialite extension.

    Tries several common library names across Linux / macOS / Windows.
    Raises RuntimeError if the extension cannot be loaded.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)

    lib_candidates = [
        "mod_spatialite",
        "mod_spatialite.so",
        "mod_spatialite.dylib",
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
    already_init = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='spatial_ref_sys'"
    ).fetchone()[0]
    if not already_init:
        conn.execute("SELECT InitSpatialMetaData(1)")
    conn.commit()
    return conn


def drop_spatialite_table(conn: sqlite3.Connection, table: str) -> None:
    """
    Safely drop a SpatiaLite table and its associated R-tree index.

    Checks sqlite_master and geometry_columns before calling SpatiaLite
    functions so no C-level error strings are printed on a fresh database.
    Safe to call on tables that do not exist or have no spatial index.
    """
    has_index = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (f"idx_{table}_geom",),
    ).fetchone()[0]
    if has_index:
        conn.execute(f"SELECT DisableSpatialIndex('{table}', 'geom')")
        conn.execute(f"DROP TABLE IF EXISTS idx_{table}_geom")

    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(
        "DELETE FROM geometry_columns WHERE f_table_name = ?", (table,)
    )
    conn.commit()
