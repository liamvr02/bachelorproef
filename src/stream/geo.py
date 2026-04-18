"""
geo.py  -  /src/stream/geo.py
==============================
Low-level geographic constants, temporal SQL helpers, and SpatiaLite
bbox fetch utilities shared by the rest of the streaming stack.

Design rule - SpatiaLite + GEOS deadlock
-----------------------------------------
On conda Windows, mod_spatialite and shapely share geos_c.dll.  Any
SpatiaLite function that calls into GEOS (Distance, MakePoint, Buffer,
ST_Intersection, Area, ...) deadlocks when shapely has already initialised
its GEOS handle in the same process - the global mutex is never released
and Ctrl+C cannot interrupt it.

RULE: every SpatiaLite query in this file uses ONLY:
  - SpatialIndex / BuildMbr  (pure R-tree, no GEOS)
  - AsBinary(geom)           (returns raw WKB bytes, no GEOS computation)
  - scalar columns (id, luc_code, elevation, timestamp, ...)

All distance and geometry math is done in Python using numpy (for points)
or shapely (for polygons, which has its own independent GEOS handle).
"""

from __future__ import annotations

import math
import sqlite3
import struct
from datetime import datetime
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from pathlib import Path

_HERE = Path(__file__).resolve().parent        # /src/stream/
_SRC  = _HERE.parent                           # /src/
DEFAULT_PREPARED = _SRC / "prepared_stream_data"

# Degrees per metre at Belgian latitudes (~51 degrees N).
# Used to convert radius_m to a rough degree bounding box for the R-tree
# pre-filter before the exact Distance() check.
_LAT_DEG_PER_M  = 1.0 / 111_320.0
_LON_DEG_PER_M  = 1.0 / (111_320.0 * math.cos(math.radians(51.0)))


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------

def _ts_epoch(ts: str) -> float:
    """Parse 'YYYY-MM-DDTHH:MM:SS' or 'YYYY-MM-DD HH:MM:SS' to a float epoch."""
    try:
        ts = ts.replace(" ", "T")
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0


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


# ---------------------------------------------------------------------------
# Vectorised distance
# ---------------------------------------------------------------------------

def _haversine_m(lon1: float, lat1: float,
                 lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Vectorised approximate distance in metres from (lon1, lat1) to each
    point in (lons, lats).  Uses the flat-earth approximation valid for the
    search radii used here (<= 1 km).
    """
    dlat = (lats - lat1) * 111_320.0
    dlon = (lons - lon1) * 111_320.0 * math.cos(math.radians(lat1))
    return np.sqrt(dlat * dlat + dlon * dlon)


# ---------------------------------------------------------------------------
# SpatiaLite WKB decode (no GEOS)
# ---------------------------------------------------------------------------

def _wkb_point_xy(blob: bytes) -> tuple:
    """
    Extract (x, y) from a WKB Point blob without calling any GEOS function.

    WKB Point layout (21 bytes, little-endian):
        byte  0    : byte order (1 = little-endian)
        bytes 1-4  : geometry type (1 = Point)
        bytes 5-12 : X (float64)
        bytes 13-20: Y (float64)

    SpatiaLite adds a 4-byte SRID prefix before the standard WKB, making the
    blob 25 bytes.  We detect this by checking the blob length.
    """
    if len(blob) == 25:
        # SpatiaLite extended WKB: 4-byte SRID header
        x, y = struct.unpack_from("<dd", blob, 9)
    else:
        x, y = struct.unpack_from("<dd", blob, 5)
    return x, y


def _spatialite_fetch_bbox(
    db: sqlite3.Connection,
    table: str,
    lon: float,
    lat: float,
    dlat: float,
    dlon: float,
    extra_cols: str,
    extra_where: str = "",
) -> list:
    """
    R-tree bounding-box pre-filter: return rows whose geometry bbox overlaps
    the search box.  Returns raw tuples with (id, _lon, _lat, ...) where
    _lon/_lat are decoded from AsBinary(geom) - no GEOS functions called.

    *extra_cols* is a comma-prefixed SQL fragment e.g. ", elevation, timestamp"
    *extra_where* is an optional AND-prefixed SQL fragment for non-spatial filters
    """
    sql = f"""
        SELECT id, AsBinary(geom) AS _wkb {extra_cols}
        FROM {table}
        WHERE id IN (
            SELECT id FROM SpatialIndex
            WHERE f_table_name = '{table}'
              AND search_frame = BuildMbr(?, ?, ?, ?, 4326)
        )
        {extra_where}
    """
    raw = db.execute(sql, (lon - dlon, lat - dlat,
                           lon + dlon, lat + dlat)).fetchall()

    # Decode WKB points into (id, lon, lat, *rest) tuples
    result = []
    for r in raw:
        blob = r[1]
        if blob is None:
            continue
        try:
            x, y = _wkb_point_xy(blob)
        except Exception:
            continue
        result.append((r[0], x, y) + r[2:])
    return result