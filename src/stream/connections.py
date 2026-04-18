"""
connections.py  -  /src/stream/connections.py
=============================================
Per-thread database connection management for DuckDB and SpatiaLite.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Dict

import duckdb

log = logging.getLogger("stream")


class Connections:
    """
    Per-thread database connections opened lazily.

    Passed to every feature callable so they can issue their own queries
    without going through the main streaming cursor.
    """

    def __init__(self, prepared: Path, catalog_meta: dict):
        self._prepared   = prepared
        self._meta       = catalog_meta          # dataset_id -> metadata row
        self._duckdb:    Dict[str, duckdb.DuckDBPyConnection] = {}
        self._spatialite: Dict[str, sqlite3.Connection]        = {}

    def duckdb(self, db_file: str) -> duckdb.DuckDBPyConnection:
        if db_file not in self._duckdb:
            path = self._prepared / db_file
            log.debug("duckdb: opening %s", path)
            t0 = time.perf_counter()
            conn = duckdb.connect(str(path), read_only=True)
            # Point DuckDB at a valid writable temp dir.  Without this,
            # on Windows the spill path resolves to "\.tmp" (relative to
            # the filesystem root) which raises an IOException.
            _tmp = Path(tempfile.gettempdir()) / "stream_duckdb_tmp"
            _tmp.mkdir(parents=True, exist_ok=True)
            try:
                conn.execute(f"SET temp_directory = '{_tmp.as_posix()}'")
            except Exception:
                pass
            self._duckdb[db_file] = conn
            log.debug("duckdb: connection ready in %.3fs", time.perf_counter() - t0)
        return self._duckdb[db_file]

    def spatialite(self, db_file: str) -> sqlite3.Connection:
        if db_file not in self._spatialite:
            path = self._prepared / db_file
            log.debug("spatialite: opening %s", path)
            t0 = time.perf_counter()
            conn = sqlite3.connect(str(path))
            conn.enable_load_extension(True)
            for lib in ["mod_spatialite", "mod_spatialite.so",
                        "mod_spatialite.dylib",
                        "/usr/lib/x86_64-linux-gnu/mod_spatialite.so"]:
                try:
                    log.debug("spatialite: about to call load_extension(%r)", lib)
                    conn.load_extension(lib)
                    log.debug("spatialite: loaded extension %r in %.3fs", lib, time.perf_counter() - t0)
                    break
                except sqlite3.OperationalError:
                    continue
            else:
                log.warning("spatialite: could not load mod_spatialite - spatial queries will fail")
            self._spatialite[db_file] = conn
            log.debug("spatialite: connection ready in %.3fs", time.perf_counter() - t0)
        return self._spatialite[db_file]

    def close(self):
        for c in self._duckdb.values():
            try: c.close()
            except Exception: pass
        for c in self._spatialite.values():
            try: c.close()
            except Exception: pass
        self._duckdb.clear()
        self._spatialite.clear()