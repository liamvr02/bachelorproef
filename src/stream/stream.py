"""
stream.py  —  /src/stream/stream.py
====================================
Spatiotemporal LST streaming layer.

Quick start
-----------
    from stream import StreamConfig, FeatureRegistry, nearest, aggregate_in_radius
    from pathlib import Path

    cfg = StreamConfig(prepared_data=Path("prepared_stream_data"))

    # Optional: target a temperature distribution
    cfg.set_distribution(target={15: 0.20, 20: 0.30, 25: 0.30, 30: 0.15, 35: 0.05})

    # Register features from the built-in framework
    reg = FeatureRegistry()
    reg.add(nearest("dhm1",        columns=["elevation"],  temporal="last_previous"))
    reg.add(nearest("ndvi",        columns=["ndvi"],       temporal="nearest"))
    reg.add(aggregate_in_radius("trees", radius_m=50,      columns=["height_m"],
                                 agg="count",              temporal="none"))

    # Custom feature using framework building-blocks (row-level API)
    from stream import query_nearest, query_radius
    def my_feature(row, connections):
        elev = query_nearest(connections, "dhm2", row.longitude, row.latitude,
                             row.timestamp, columns=["elevation"],
                             temporal="last_previous")
        return {"dhm2_elev_log": math.log1p(elev.get("elevation", 0))}
    reg.add_custom(my_feature, name="dhm2_log_elevation")

    # Capture as DataFrame
    import pandas as pd
    df = pd.concat(cfg.stream(reg, batch_size=10_000), ignore_index=True)

    # Or feed a model batch by batch
    for batch_df in cfg.stream(reg, batch_size=512):
        model.fit(batch_df[feature_cols], batch_df["temperature"])

Architecture
------------
LST rows are read from lst.duckdb with DuckDB cursor.fetchmany() so the
725 M-row table is never loaded into memory.

Feature computation uses two paths:

BATCH path (framework features — nearest, aggregate_in_radius):
  Each batch's coordinates are bulk-loaded into a SpatiaLite temporary table.
  A single spatial JOIN per feature replaces N per-row queries.
  For aggregate features, results are deduplicated by tile_id so rows sharing
  a tile share one query result (valid because all pixels in an H3 cell at
  resolution 9 are within ~200 m of each other).

ROW path (custom callables added via add_custom):
  Custom features receive one FeatureRow at a time plus a Connections object.
  They may call query_nearest() and query_radius() to reuse framework logic.
  Results are accumulated into pre-allocated column arrays (not list-of-dicts)
  and assembled into a DataFrame at batch end.

The 2×2 temporal×spatial framework
-----------------------------------
Spatial:
  nearest(dataset, columns, temporal, ...)
      → the single closest point in space (optionally filtered by time)
  aggregate_in_radius(dataset, radius_m, columns, agg, temporal, ...)
      → COUNT / AVG / SUM / MIN / MAX of all points within radius_m metres

Temporal (applies to non-static datasets):
  "last_previous"   → most recent observation with ts <= driving_ts
  "nearest"         → observation with smallest |ts - driving_ts|
  "none"            → no temporal filter (static datasets)
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections import namedtuple
from pathlib import Path
from typing import (
    Callable, Dict, Generator, Iterable, Iterator,
    List, Optional, Sequence, Tuple
)

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent        # /src/stream/
_SRC  = _HERE.parent                           # /src/
DEFAULT_PREPARED = _SRC / "prepared_stream_data"

# Degrees per metre at Belgian latitudes (~51 °N).
# Used to convert radius_m to a rough degree bounding box for the R-tree
# pre-filter before the exact Distance() check.
_LAT_DEG_PER_M  = 1.0 / 111_320.0
_LON_DEG_PER_M  = 1.0 / (111_320.0 * math.cos(math.radians(51.0)))


# ---------------------------------------------------------------------------
# FeatureRow — typed view of one LST row passed to feature callables
# ---------------------------------------------------------------------------
FeatureRow = namedtuple(
    "FeatureRow",
    ["longitude", "latitude", "temperature", "emissivity",
     "landsat_id", "image_id", "timestamp", "partition_key", "tile_id"],
)


# ---------------------------------------------------------------------------
# Connections — live database handles, one per thread
# ---------------------------------------------------------------------------
class Connections:
    """
    Per-thread database connections opened lazily.

    Passed to every feature callable so they can issue their own queries
    without going through the main streaming cursor.
    """

    def __init__(self, prepared: Path, catalog_meta: dict):
        self._prepared   = prepared
        self._meta       = catalog_meta          # dataset_id → metadata row
        self._duckdb:    Dict[str, duckdb.DuckDBPyConnection] = {}
        self._spatialite: Dict[str, sqlite3.Connection]        = {}

    def duckdb(self, db_file: str) -> duckdb.DuckDBPyConnection:
        if db_file not in self._duckdb:
            path = self._prepared / db_file
            self._duckdb[db_file] = duckdb.connect(str(path), read_only=True)
        return self._duckdb[db_file]

    def spatialite(self, db_file: str) -> sqlite3.Connection:
        if db_file not in self._spatialite:
            path = self._prepared / db_file
            conn = sqlite3.connect(str(path))
            conn.enable_load_extension(True)
            for lib in ["mod_spatialite", "mod_spatialite.so",
                        "mod_spatialite.dylib",
                        "/usr/lib/x86_64-linux-gnu/mod_spatialite.so"]:
                try:
                    conn.load_extension(lib)
                    break
                except sqlite3.OperationalError:
                    continue
            conn.row_factory = sqlite3.Row
            self._spatialite[db_file] = conn
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


# ---------------------------------------------------------------------------
# Core spatial/temporal query helpers (public — usable in custom features)
# ---------------------------------------------------------------------------

def _temporal_clause(temporal: str, ts_col: str, driving_ts: str) -> str:
    """Return a SQL WHERE fragment for temporal filtering."""
    if temporal == "last_previous":
        return f"AND {ts_col} <= '{driving_ts}'"
    if temporal == "nearest":
        return ""   # handled via ORDER BY in the calling query
    return ""       # "none"


def _temporal_order(temporal: str, ts_col: str, driving_ts: str) -> str:
    """Return a SQL ORDER BY fragment for temporal ordering."""
    if temporal == "last_previous":
        return f"{ts_col} DESC"
    if temporal == "nearest":
        return f"ABS(strftime('%s', {ts_col}) - strftime('%s', '{driving_ts}'))"
    return "1"   # no ordering preference for static datasets


def query_nearest(
    conn: Connections,
    dataset_id: str,
    lon: float,
    lat: float,
    timestamp: str,
    columns: List[str],
    temporal: str = "none",
    radius_m: float = 500.0,
) -> dict:
    """
    Find the single spatially nearest point in *dataset_id*, optionally
    filtered/ordered by time.

    Returns a dict of {column: value} for the nearest row, or {} if none found.

    This is the low-level building block used by the nearest() factory and
    available to custom feature callables.
    """
    meta    = conn._meta[dataset_id]
    store   = meta["store"]
    table   = meta["db_table"]
    db_file = meta["db_file"]
    ts_col  = meta.get("timestamp_column", "timestamp")
    has_ts  = meta["temporal_behavior"] != "static"

    cols_sql = ", ".join(columns)
    t_where  = _temporal_clause(temporal, ts_col, timestamp) if has_ts else ""
    t_order  = _temporal_order(temporal, ts_col, timestamp)  if has_ts else "1"

    if store == "spatialite":
        db = conn.spatialite(db_file)
        # Bounding-box pre-filter via R-tree, then exact distance ordering
        dlat = radius_m * _LAT_DEG_PER_M
        dlon = radius_m * _LON_DEG_PER_M
        sql = f"""
            SELECT {cols_sql},
                   Distance(geom, MakePoint(?, ?, 4326)) AS _dist
            FROM {table}
            WHERE id IN (
                SELECT id FROM SpatialIndex
                WHERE f_table_name = '{table}'
                  AND search_frame = BuildMbr(?, ?, ?, ?, 4326)
            )
            {t_where}
            ORDER BY _dist ASC, {t_order}
            LIMIT 1
        """
        params = (lon, lat,
                  lon - dlon, lat - dlat, lon + dlon, lat + dlat)
        cur = db.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else {}

    else:  # duckdb
        db = conn.duckdb(db_file)
        sql = f"""
            SELECT {cols_sql},
                   sqrt(pow((longitude - {lon}) * 111320, 2)
                      + pow((latitude  - {lat}) * 111320
                          * cos(radians({lat})), 2)) AS _dist
            FROM {table}
            WHERE longitude BETWEEN {lon - radius_m * _LON_DEG_PER_M}
                              AND {lon + radius_m * _LON_DEG_PER_M}
              AND latitude  BETWEEN {lat - radius_m * _LAT_DEG_PER_M}
                              AND {lat + radius_m * _LAT_DEG_PER_M}
            {t_where}
            ORDER BY _dist ASC, {t_order}
            LIMIT 1
        """
        row = db.execute(sql).fetchone()
        if row is None:
            return {}
        desc = db.description
        return {desc[i][0]: row[i] for i in range(len(desc))}


def query_radius(
    conn: Connections,
    dataset_id: str,
    lon: float,
    lat: float,
    timestamp: str,
    radius_m: float,
    columns: List[str],
    agg: str,
    temporal: str = "none",
) -> dict:
    """
    Aggregate all points within *radius_m* metres of (lon, lat).

    *agg* is one of: "count", "avg", "sum", "min", "max".
    Returns {f"{col}_{agg}": value, "count": n} or {"count": 0} if empty.

    This is the low-level building block used by aggregate_in_radius() and
    available to custom feature callables.
    """
    meta    = conn._meta[dataset_id]
    store   = meta["store"]
    table   = meta["db_table"]
    db_file = meta["db_file"]
    ts_col  = meta.get("timestamp_column", "timestamp")
    has_ts  = meta["temporal_behavior"] != "static"

    t_where = _temporal_clause(temporal, ts_col, timestamp) if has_ts else ""
    agg_up  = agg.upper()
    agg_sql = ", ".join(
        f"{agg_up}(CAST({c} AS REAL)) AS {c}_{agg}" for c in columns
        if agg_up != "COUNT"
    )
    count_sql = "COUNT(*) AS count"
    select_sql = f"{count_sql}" + (f", {agg_sql}" if agg_sql else "")

    if store == "spatialite":
        db = conn.spatialite(db_file)
        dlat = radius_m * _LAT_DEG_PER_M
        dlon = radius_m * _LON_DEG_PER_M
        sql = f"""
            SELECT {select_sql}
            FROM {table}
            WHERE id IN (
                SELECT id FROM SpatialIndex
                WHERE f_table_name = '{table}'
                  AND search_frame = BuildMbr(?, ?, ?, ?, 4326)
            )
              AND Distance(geom, MakePoint(?, ?, 4326)) <= ?
            {t_where}
        """
        params = (lon - dlon, lat - dlat, lon + dlon, lat + dlat,
                  lon, lat, radius_m / 111_320.0)
        cur = db.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else {"count": 0}

    else:  # duckdb
        db = conn.duckdb(db_file)
        radius_deg = radius_m / 111_320.0
        sql = f"""
            SELECT {select_sql}
            FROM {table}
            WHERE longitude BETWEEN {lon - radius_m * _LON_DEG_PER_M}
                              AND {lon + radius_m * _LON_DEG_PER_M}
              AND latitude  BETWEEN {lat - radius_m * _LAT_DEG_PER_M}
                              AND {lat + radius_m * _LAT_DEG_PER_M}
              AND sqrt(pow((longitude - {lon}) * 111320, 2)
                     + pow((latitude  - {lat}) * 111320
                         * cos(radians({lat})), 2)) <= {radius_m}
            {t_where}
        """
        row = db.execute(sql).fetchone()
        if row is None:
            return {"count": 0}
        desc = db.description
        return {desc[i][0]: row[i] for i in range(len(desc))}


# ---------------------------------------------------------------------------
# Feature descriptor — stores both a batch path and a row-level fallback
# ---------------------------------------------------------------------------
class _FeatureDescriptor:
    """
    Describes one registered feature.

    Framework features (nearest, aggregate_in_radius) provide compute_batch()
    which issues one SQL query per feature per batch against a temporary
    coordinate table loaded into SpatiaLite or DuckDB.

    Custom features provide compute_row() which is called once per row and
    may use query_nearest() / query_radius() to reuse framework logic.
    """
    __slots__ = ("name", "prefix", "_batch_fn", "_row_fn")

    def __init__(
        self,
        name:     str,
        prefix:   str,
        batch_fn: Optional[Callable] = None,   # (df, conns) -> pd.Series per column
        row_fn:   Optional[Callable] = None,   # (row, conns) -> dict
    ):
        self.name     = name
        self.prefix   = prefix
        self._batch_fn = batch_fn
        self._row_fn   = row_fn

    @property
    def is_batchable(self) -> bool:
        return self._batch_fn is not None

    def compute_batch(
        self, df: pd.DataFrame, conns: "Connections"
    ) -> pd.DataFrame:
        """
        Compute this feature for all rows in df at once.
        Returns a DataFrame with one column per output key, same index as df.
        """
        result = self._batch_fn(df, conns)
        if self.prefix:
            result = result.rename(columns={c: f"{self.prefix}{c}" for c in result.columns})
        return result

    def compute_row(self, row: "FeatureRow", conns: "Connections") -> dict:
        """Compute this feature for a single row (custom callables only)."""
        result = self._row_fn(row, conns)
        if self.prefix:
            return {f"{self.prefix}{k}": v for k, v in result.items()}
        return result


# ---------------------------------------------------------------------------
# Batch spatial query helpers — one query per feature per batch
# ---------------------------------------------------------------------------

def _load_temp_points(
    db: sqlite3.Connection,
    df: pd.DataFrame,
    temp_table: str = "_batch_pts",
) -> None:
    """
    Load (row_idx, lon, lat, ts) from df into a SpatiaLite temp table.

    Uses a single executemany call so the entire batch crosses the Python/
    SQLite boundary in one round-trip.  The geometry column lets SpatiaLite
    join against the R-tree index of the feature table.
    """
    db.execute(f"DROP TABLE IF EXISTS {temp_table}")
    db.execute(f"""
        CREATE TEMPORARY TABLE {temp_table} (
            row_idx INTEGER PRIMARY KEY,
            lon     REAL,
            lat     REAL,
            ts      TEXT
        )
    """)
    db.executemany(
        f"INSERT INTO {temp_table}(row_idx, lon, lat, ts) VALUES (?,?,?,?)",
        [
            (int(i), float(row.longitude), float(row.latitude), str(row.timestamp))
            for i, row in df.iterrows()
        ],
    )


def batch_nearest(
    df: pd.DataFrame,
    conns: Connections,
    dataset_id: str,
    columns: List[str],
    temporal: str,
    radius_m: float,
) -> pd.DataFrame:
    """
    Find the nearest point in *dataset_id* for every row in *df*.

    Issues one SQL query against a temporary coordinate table rather than
    N per-row queries.  Returns a DataFrame indexed like df with one column
    per requested feature column (None where no point is found).

    For aggregate features (see batch_radius), tile-level deduplication is
    applied separately by the caller.
    """
    meta    = conns._meta[dataset_id]
    store   = meta["store"]
    table   = meta["db_table"]
    db_file = meta["db_file"]
    has_ts  = meta["temporal_behavior"] != "static"
    ts_col  = meta.get("timestamp_column", "timestamp")

    cols_select = ", ".join(f"f.{c}" for c in columns)
    dlat = radius_m * _LAT_DEG_PER_M
    dlon = radius_m * _LON_DEG_PER_M

    result = pd.DataFrame(index=df.index, columns=columns, dtype=object)

    if store == "spatialite":
        db = conns.spatialite(db_file)
        _load_temp_points(db, df)

        # Temporal WHERE / ORDER fragments
        if has_ts and temporal == "last_previous":
            t_where = "AND f.timestamp <= p.ts"
            t_order = "f.timestamp DESC,"
        elif has_ts and temporal == "nearest":
            t_where = ""
            t_order = "ABS(strftime('%s', f.timestamp) - strftime('%s', p.ts)),"
        else:
            t_where = ""
            t_order = ""

        sql = f"""
            SELECT p.row_idx, {cols_select}
            FROM _batch_pts p
            JOIN {table} f ON f.id IN (
                SELECT id FROM SpatialIndex
                WHERE f_table_name = '{table}'
                  AND search_frame = BuildMbr(
                      p.lon - {dlon}, p.lat - {dlat},
                      p.lon + {dlon}, p.lat + {dlat}, 4326)
            )
            {t_where}
            GROUP BY p.row_idx
            HAVING Distance(f.geom, MakePoint(p.lon, p.lat, 4326))
                   = MIN(Distance(f.geom, MakePoint(p.lon, p.lat, 4326)))
            ORDER BY p.row_idx, {t_order}
                   Distance(f.geom, MakePoint(p.lon, p.lat, 4326))
        """
        # Use DISTINCT ON equivalent: keep first row per row_idx
        # SQLite doesn't have DISTINCT ON, so we post-process in Python
        rows = db.execute(sql).fetchall()
        seen: set = set()
        for r in rows:
            idx = r[0]
            if idx not in seen:
                seen.add(idx)
                for j, col in enumerate(columns):
                    result.at[idx, col] = r[j + 1]

    else:  # duckdb
        db = conns.duckdb(db_file)
        # Build values literal from batch
        pts_values = ", ".join(
            f"({int(i)}, {float(row.longitude)}, {float(row.latitude)}, "
            f"'{row.timestamp}')"
            for i, row in df.iterrows()
        )
        if has_ts and temporal == "last_previous":
            t_where = "AND f.timestamp <= p.ts"
            t_order = "f.timestamp DESC,"
        elif has_ts and temporal == "nearest":
            t_where = ""
            t_order = "ABS(epoch(CAST(f.timestamp AS TIMESTAMP)) - epoch(CAST(p.ts AS TIMESTAMP))),"
        else:
            t_where = ""
            t_order = ""

        sql = f"""
            WITH pts(row_idx, lon, lat, ts) AS (
                VALUES {pts_values}
            ),
            ranked AS (
                SELECT p.row_idx, {cols_select},
                       ROW_NUMBER() OVER (
                           PARTITION BY p.row_idx
                           ORDER BY {t_order}
                               sqrt(pow((f.longitude - p.lon) * 111320, 2)
                                  + pow((f.latitude  - p.lat) * 111320
                                      * cos(radians(p.lat)), 2))
                       ) AS rn
                FROM pts p
                JOIN {table} f
                  ON f.longitude BETWEEN p.lon - {dlon} AND p.lon + {dlon}
                 AND f.latitude  BETWEEN p.lat - {dlat} AND p.lat + {dlat}
                 AND sqrt(pow((f.longitude - p.lon) * 111320, 2)
                        + pow((f.latitude  - p.lat) * 111320
                            * cos(radians(p.lat)), 2)) <= {radius_m}
                {t_where}
            )
            SELECT row_idx, {", ".join(columns)}
            FROM ranked
            WHERE rn = 1
            ORDER BY row_idx
        """
        rows = db.execute(sql).fetchall()
        desc = [d[0] for d in db.description]
        col_positions = {c: desc.index(c) for c in columns}
        for r in rows:
            idx = r[0]
            for col in columns:
                result.at[idx, col] = r[col_positions[col]]

    return result


def batch_radius(
    df: pd.DataFrame,
    conns: Connections,
    dataset_id: str,
    columns: List[str],
    agg: str,
    temporal: str,
    radius_m: float,
) -> pd.DataFrame:
    """
    Aggregate all points within *radius_m* metres for every row in *df*.

    Applies tile-level deduplication: all rows sharing the same tile_id
    receive the same aggregate result (valid because H3 cells at resolution 9
    are ~200 m across, smaller than any meaningful search radius for
    aggregate features).  This collapses N queries down to N_unique_tiles.

    Returns a DataFrame indexed like df with result columns named
    {col}_{agg} and "count".
    """
    meta    = conns._meta[dataset_id]
    store   = meta["store"]
    table   = meta["db_table"]
    db_file = meta["db_file"]
    has_ts  = meta["temporal_behavior"] != "static"
    ts_col  = meta.get("timestamp_column", "timestamp")

    agg_up = agg.upper()
    agg_sql = ", ".join(
        f"{agg_up}(CAST(f.{c} AS REAL)) AS {c}_{agg}"
        for c in columns if agg_up != "COUNT"
    )
    out_cols = ["count"] + [f"{c}_{agg}" for c in columns if agg_up != "COUNT"]
    result   = pd.DataFrame(index=df.index,
                             columns=out_cols, dtype=object)

    dlat = radius_m * _LAT_DEG_PER_M
    dlon = radius_m * _LON_DEG_PER_M

    if has_ts and temporal == "last_previous":
        t_where_tpl = "AND f.timestamp <= '{ts}'"
    elif has_ts and temporal == "nearest":
        # For radius aggregates, "nearest" means within a time window ±Δt
        # approximated as the same as last_previous (most recent snapshot)
        t_where_tpl = "AND f.timestamp <= '{ts}'"
    else:
        t_where_tpl = ""

    # Tile-level deduplication: compute once per unique tile
    tile_cache: Dict[str, dict] = {}

    for tile_id, tile_df in df.groupby("tile_id"):
        # Use the first row's coordinates as representative for the tile
        rep = tile_df.iloc[0]
        lon, lat, ts = float(rep.longitude), float(rep.latitude), str(rep.timestamp)
        cache_key = f"{tile_id}:{ts[:10] if has_ts else 'static'}"

        if cache_key not in tile_cache:
            t_where = t_where_tpl.format(ts=ts) if t_where_tpl else ""
            select_sql = f"COUNT(*) AS count" + (f", {agg_sql}" if agg_sql else "")

            if store == "spatialite":
                db = conns.spatialite(db_file)
                sql = f"""
                    SELECT {select_sql}
                    FROM {table} f
                    WHERE f.id IN (
                        SELECT id FROM SpatialIndex
                        WHERE f_table_name = '{table}'
                          AND search_frame = BuildMbr(
                              {lon - dlon}, {lat - dlat},
                              {lon + dlon}, {lat + dlat}, 4326)
                    )
                      AND Distance(f.geom, MakePoint(?, ?, 4326)) <= ?
                    {t_where}
                """
                row = db.execute(sql, (lon, lat, radius_m / 111_320.0)).fetchone()
            else:
                db = conns.duckdb(db_file)
                sql = f"""
                    SELECT {select_sql}
                    FROM {table} f
                    WHERE f.longitude BETWEEN {lon - dlon} AND {lon + dlon}
                      AND f.latitude  BETWEEN {lat - dlat} AND {lat + dlat}
                      AND sqrt(pow((f.longitude - {lon}) * 111320, 2)
                             + pow((f.latitude  - {lat}) * 111320
                                 * cos(radians({lat})), 2)) <= {radius_m}
                    {t_where}
                """
                row = db.execute(sql).fetchone()

            if row:
                tile_cache[cache_key] = dict(zip(out_cols, row))
            else:
                tile_cache[cache_key] = {"count": 0}

        tile_result = tile_cache[cache_key]
        for idx in tile_df.index:
            for col in out_cols:
                result.at[idx, col] = tile_result.get(col)

    return result


# ---------------------------------------------------------------------------
# Feature factories  (the 2×2 framework)
# ---------------------------------------------------------------------------

def nearest(
    dataset_id: str,
    columns: List[str],
    temporal: str = "none",
    radius_m: float = 500.0,
    prefix: str = "",
) -> _FeatureDescriptor:
    """
    Factory: nearest point in *dataset_id* to the driving LST pixel.

    Parameters
    ----------
    dataset_id : registered dataset (e.g. "dhm1", "ndvi")
    columns    : which columns to return from the nearest row
    temporal   : "none"          — ignore timestamps (static datasets)
                 "last_previous" — most recent observation with ts <= driving_ts
                 "nearest"       — observation closest in time to driving_ts
    radius_m   : search radius in metres (R-tree pre-filter)
    prefix     : optional column name prefix in the output

    Implementation
    --------------
    Issues one batch spatial JOIN per feature per batch, not one query per row.
    Custom callables can reuse the row-level helper:
        result = query_nearest(conns, "dhm1", row.longitude, row.latitude,
                               row.timestamp, ["elevation"], temporal="nearest")
    """
    if temporal not in ("none", "last_previous", "nearest"):
        raise ValueError(f"temporal must be 'none', 'last_previous' or 'nearest', got {temporal!r}")

    _prefix = prefix or f"{dataset_id}_"

    def _batch(df: pd.DataFrame, conns: Connections) -> pd.DataFrame:
        return batch_nearest(df, conns, dataset_id, columns, temporal, radius_m)

    def _row(row: FeatureRow, conns: Connections) -> dict:
        return query_nearest(conns, dataset_id,
                             row.longitude, row.latitude, row.timestamp,
                             columns=columns, temporal=temporal, radius_m=radius_m)

    return _FeatureDescriptor(
        name=f"nearest_{dataset_id}_{temporal}",
        prefix=_prefix,
        batch_fn=_batch,
        row_fn=_row,
    )


def aggregate_in_radius(
    dataset_id: str,
    radius_m: float,
    columns: List[str],
    agg: str = "count",
    temporal: str = "none",
    prefix: str = "",
) -> _FeatureDescriptor:
    """
    Factory: aggregate all points within *radius_m* metres of each driving pixel.

    Parameters
    ----------
    dataset_id : registered dataset
    radius_m   : search radius in metres
    columns    : columns to aggregate (ignored when agg="count")
    agg        : "count", "avg", "sum", "min", or "max"
    temporal   : same as nearest()
    prefix     : optional column name prefix

    Implementation
    --------------
    Applies tile-level deduplication: all pixels in the same H3 tile share
    one aggregate result, reducing queries to N_unique_tiles per batch.
    Custom callables can reuse the row-level helper:
        result = query_radius(conns, "trees", row.longitude, row.latitude,
                              row.timestamp, radius_m=25.0,
                              columns=[], agg="count", temporal="none")
    """
    if agg not in ("count", "avg", "sum", "min", "max"):
        raise ValueError(f"agg must be one of count/avg/sum/min/max, got {agg!r}")
    if temporal not in ("none", "last_previous", "nearest"):
        raise ValueError(f"temporal must be 'none', 'last_previous' or 'nearest', got {temporal!r}")

    _prefix = prefix or f"{dataset_id}_{agg}{int(radius_m)}m_"

    def _batch(df: pd.DataFrame, conns: Connections) -> pd.DataFrame:
        return batch_radius(df, conns, dataset_id, columns, agg, temporal, radius_m)

    def _row(row: FeatureRow, conns: Connections) -> dict:
        return query_radius(conns, dataset_id,
                            row.longitude, row.latitude, row.timestamp,
                            radius_m=radius_m, columns=columns,
                            agg=agg, temporal=temporal)

    return _FeatureDescriptor(
        name=f"radius_{dataset_id}_{agg}_{int(radius_m)}m_{temporal}",
        prefix=_prefix,
        batch_fn=_batch,
        row_fn=_row,
    )


# ---------------------------------------------------------------------------
# FeatureRegistry
# ---------------------------------------------------------------------------
class FeatureRegistry:
    """
    Holds all registered feature descriptors.

    Usage
    -----
        reg = FeatureRegistry()
        reg.add(nearest("dhm1", ["elevation"], temporal="last_previous"))
        reg.add(aggregate_in_radius("trees", 50, [], agg="count"))
        reg.add_custom(my_fn, name="my_feature")
    """

    def __init__(self):
        self._descriptors: List[_FeatureDescriptor] = []

    def add(self, descriptor: _FeatureDescriptor) -> "FeatureRegistry":
        """Register a feature produced by nearest() or aggregate_in_radius()."""
        self._descriptors.append(descriptor)
        return self

    def add_custom(
        self,
        fn: Callable[["FeatureRow", "Connections"], dict],
        name: str,
        prefix: str = "",
    ) -> "FeatureRegistry":
        """
        Register a custom feature callable.

        The callable receives (row: FeatureRow, conns: Connections) and returns
        a dict of {column_name: value}.  It can reuse framework logic via:
            from stream import query_nearest, query_radius
        """
        self._descriptors.append(
            _FeatureDescriptor(name=name, prefix=prefix, row_fn=fn)
        )
        return self

    @property
    def _batch_descriptors(self) -> List[_FeatureDescriptor]:
        return [d for d in self._descriptors if d.is_batchable]

    @property
    def _row_descriptors(self) -> List[_FeatureDescriptor]:
        return [d for d in self._descriptors if not d.is_batchable]

    def compute_batch_features(
        self, df: pd.DataFrame, conns: "Connections"
    ) -> pd.DataFrame:
        """
        Compute all batchable features for the entire df in bulk SQL.
        Returns a DataFrame with one column per output, indexed like df.
        """
        parts = []
        for desc in self._batch_descriptors:
            try:
                parts.append(desc.compute_batch(df, conns))
            except Exception as exc:
                # Fill with None so the row is still emitted
                err_col = f"_err_{desc.name}"
                parts.append(pd.DataFrame(
                    {err_col: str(exc)}, index=df.index
                ))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

    def compute_row_features(
        self, raw_rows: list, conns: "Connections"
    ) -> pd.DataFrame:
        """
        Compute all custom (row-level) features.
        Accumulates into pre-allocated column arrays to avoid list-of-dicts overhead.
        Returns a DataFrame with one column per output, indexed 0..N-1.
        """
        if not self._row_descriptors:
            return pd.DataFrame()

        n = len(raw_rows)
        # Pre-allocate: discover column names from first non-erroring row
        col_arrays: Dict[str, list] = {}
        for i, raw_row in enumerate(raw_rows):
            row = FeatureRow(*raw_row)
            for desc in self._row_descriptors:
                try:
                    result = desc.compute_row(row, conns)
                except Exception as exc:
                    result = {f"_err_{desc.name}": str(exc)}
                for k, v in result.items():
                    if k not in col_arrays:
                        col_arrays[k] = [None] * n
                    col_arrays[k][i] = v
        return pd.DataFrame(col_arrays)

    def __len__(self) -> int:
        return len(self._descriptors)


# ---------------------------------------------------------------------------
# DistributionTarget
# ---------------------------------------------------------------------------
class DistributionTarget:
    """
    Target temperature distribution for weighted partition sampling.

    Parameters
    ----------
    target : dict mapping temperature bin *lower edge* (°C) to desired proportion.
             Proportions are normalised to sum to 1.
             Example:  {15: 0.20, 20: 0.30, 25: 0.30, 30: 0.15, 35: 0.05}

    The histogram bins in catalog.duckdb use 2 °C edges from -10 to 60 °C.
    Each key in *target* is matched to the closest bin lower edge.
    """

    def __init__(self, target: Dict[float, float]):
        total = sum(target.values())
        self.target = {float(k): v / total for k, v in target.items()}

    def partition_weight(
        self,
        hist_counts: List[int],
        bin_edges: List[float],
    ) -> float:
        """
        Score a partition by how much it contributes to the target distribution.

        Returns a weight ∈ [0, 1].  Partitions with weight 0 are skipped.
        """
        total = sum(hist_counts)
        if total == 0:
            return 0.0
        score = 0.0
        for temp_edge, desired_prop in self.target.items():
            # Find the bin index for this edge
            bin_idx = min(
                range(len(bin_edges) - 1),
                key=lambda i: abs(bin_edges[i] - temp_edge),
            )
            actual_prop = hist_counts[bin_idx] / total
            score += min(actual_prop, desired_prop)
        return score


# ---------------------------------------------------------------------------
# StreamConfig — main user-facing object
# ---------------------------------------------------------------------------
class StreamConfig:
    """
    Configure and start an LST feature stream.

    Parameters
    ----------
    prepared_data : Path to the prepared_stream_data directory produced by ingest.py.
    batch_size    : rows yielded per DataFrame batch.
    partition_keys: optional list of "YYYY-MM" strings to restrict streaming to
                    specific months.  Default: all months.

    Example
    -------
        cfg = StreamConfig(Path("prepared_stream_data"))
        cfg.set_distribution({20: 0.4, 25: 0.4, 30: 0.2})

        reg = FeatureRegistry()
        reg.add(nearest("dhm1", ["elevation"], temporal="last_previous"))
        reg.add(aggregate_in_radius("trees", 50, [], agg="count"))

        for batch_df in cfg.stream(reg, batch_size=5_000):
            model.partial_fit(batch_df[X_cols], batch_df["temperature"])
    """

    def __init__(
        self,
        prepared_data: Path = DEFAULT_PREPARED,
        batch_size: int = 10_000,
        partition_keys: Optional[List[str]] = None,
    ):
        self.prepared      = Path(prepared_data)
        self.batch_size    = batch_size
        self._partitions   = partition_keys   # None → all
        self._distribution: Optional[DistributionTarget] = None
        self._catalog_meta: Optional[dict] = None
        self._bin_edges:    Optional[List[float]] = None
        self._partition_stats: Optional[List[dict]] = None

    def set_distribution(self, target: Dict[float, float]) -> "StreamConfig":
        """Define a target temperature distribution for weighted sampling."""
        self._distribution = DistributionTarget(target)
        return self

    def _load_catalog(self) -> None:
        """Read catalog.duckdb once at stream start."""
        cat_path = self.prepared / "catalog.duckdb"
        if not cat_path.exists():
            raise FileNotFoundError(
                f"Catalog not found: {cat_path}\n"
                "Run ingest.py first to build the feature store."
            )
        conn = duckdb.connect(str(cat_path), read_only=True)

        # Dataset metadata — 6 rows, fine to fetchall
        rows = conn.execute("SELECT * FROM dataset_metadata").fetchall()
        cols = [d[0] for d in conn.description]
        self._catalog_meta = {}
        for r in rows:
            row_dict = dict(zip(cols, r))
            self._catalog_meta[row_dict["dataset_id"]] = row_dict

        # Histogram config — 1 row
        cfg_row = conn.execute(
            "SELECT bin_edges_json FROM histogram_config WHERE dataset_id = 'lst'"
        ).fetchone()
        self._bin_edges = json.loads(cfg_row[0]) if cfg_row else None

        # Partition statistics — 1.2 M rows: stream with fetchmany so we
        # never hold the full result set in memory.
        cursor = conn.execute(
            "SELECT partition_key, tile_id, row_count, histogram_json "
            "FROM partition_statistics WHERE dataset_id = 'lst'"
        )
        self._partition_stats = []
        while True:
            batch = cursor.fetchmany(50_000)
            if not batch:
                break
            for r in batch:
                self._partition_stats.append({
                    "partition_key": r[0],
                    "tile_id":       r[1],
                    "row_count":     r[2],
                    "histogram":     json.loads(r[3]) if r[3] else None,
                })
        conn.close()

    def _select_partitions(self) -> List[Tuple[str, float]]:
        """
        Return list of (partition_key, weight) sorted by weight descending.

        Weight is 1.0 for all partitions when no distribution is set.
        Partitions with weight 0 are excluded.
        """
        stats = self._partition_stats or []

        # Apply partition_keys filter
        if self._partitions:
            pk_set = set(self._partitions)
            stats  = [s for s in stats if s["partition_key"] in pk_set]

        if not stats:
            return []

        if self._distribution is None or self._bin_edges is None:
            # All partitions with equal weight, deduplicated by partition_key
            seen = {}
            for s in stats:
                pk = s["partition_key"]
                if pk not in seen:
                    seen[pk] = 1.0
            return sorted(seen.items())

        # Weighted by distribution match
        pk_weights: Dict[str, float] = {}
        for s in stats:
            pk  = s["partition_key"]
            h   = s["histogram"]
            if h is None:
                continue
            counts = h.get("counts", [])
            w = self._distribution.partition_weight(counts, self._bin_edges)
            pk_weights[pk] = pk_weights.get(pk, 0.0) + w

        result = [(pk, w) for pk, w in pk_weights.items() if w > 0.0]
        return sorted(result, key=lambda x: -x[1])   # highest weight first

    def stream(
        self,
        registry: Optional[FeatureRegistry] = None,
        batch_size: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Stream LST data with engineered features, one DataFrame batch at a time.

        Parameters
        ----------
        registry   : FeatureRegistry with registered features.
                     Pass None (or an empty registry) to stream raw LST only.
        batch_size : override the batch_size set in __init__.

        Yields
        ------
        pd.DataFrame, shape (batch_size, 9 + n_features)
        Columns: longitude, latitude, temperature, emissivity, landsat_id,
                 image_id, timestamp, partition_key, tile_id,
                 [feature columns...]

        Memory model
        ------------
        The full LST table is never loaded.  DuckDB cursor.fetchmany() pulls
        batch_size rows per iteration.  Batch features issue one SQL query per
        feature per batch.  Custom (row-level) features accumulate into
        pre-allocated column arrays, not list-of-dicts.
        """
        if registry is None:
            registry = FeatureRegistry()

        bs = batch_size or self.batch_size
        self._load_catalog()
        partitions = self._select_partitions()

        if not partitions:
            return

        lst_meta = self._catalog_meta.get("lst", {})
        lst_db   = self.prepared / lst_meta.get("db_file", "lst.duckdb")

        lst_conn      = duckdb.connect(str(lst_db), read_only=True)
        feature_conns = Connections(self.prepared, self._catalog_meta)

        has_batch = len(registry._batch_descriptors) > 0
        has_row   = len(registry._row_descriptors)   > 0

        try:
            for partition_key, _weight in tqdm(
                partitions, desc="Streaming partitions", unit="partition"
            ):
                cursor = lst_conn.execute(
                    "SELECT longitude, latitude, temperature, emissivity, "
                    "       landsat_id, image_id, timestamp, partition_key, tile_id "
                    "FROM lst "
                    "WHERE partition_key = ? "
                    "ORDER BY tile_id",
                    [partition_key],
                )

                while True:
                    raw_rows = cursor.fetchmany(bs)
                    if not raw_rows:
                        break

                    base_df = pd.DataFrame(raw_rows, columns=FeatureRow._fields)

                    if not has_batch and not has_row:
                        yield base_df
                        continue

                    parts = [base_df]

                    # Batch features: one SQL query per feature per batch
                    if has_batch:
                        batch_feat_df = registry.compute_batch_features(
                            base_df, feature_conns
                        )
                        if not batch_feat_df.empty:
                            parts.append(
                                batch_feat_df.reset_index(drop=True)
                            )

                    # Row features: custom callables, pre-allocated arrays
                    if has_row:
                        row_feat_df = registry.compute_row_features(
                            raw_rows, feature_conns
                        )
                        if not row_feat_df.empty:
                            parts.append(row_feat_df)

                    yield pd.concat(parts, axis=1)

        finally:
            lst_conn.close()
            feature_conns.close()

    def to_dataframe(
        self,
        registry: Optional[FeatureRegistry] = None,
        max_rows: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Convenience method: collect the stream into a single DataFrame.

        Warning: loads all streamed rows into memory.  Use stream() for
        large datasets or model training loops.

        Parameters
        ----------
        max_rows : stop after this many rows (useful for exploration).
        """
        chunks = []
        total  = 0
        for batch_df in self.stream(registry, batch_size=batch_size):
            if max_rows is not None:
                remaining = max_rows - total
                if remaining <= 0:
                    break
                batch_df = batch_df.iloc[:remaining]
            chunks.append(batch_df)
            total += len(batch_df)
            if max_rows is not None and total >= max_rows:
                break
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()