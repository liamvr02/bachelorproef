"""
ingest/config.py
================
Shared constants, path defaults, histogram bin edges, and the dataset registry.

Time decomposition
------------------
Every LST row stores its acquisition timestamp both as a raw ISO string and as
five pre-computed numeric components that can each be independently targeted
by the streaming distribution system:

  year          INT    — calendar year          (e.g. 2015)
  month_of_year INT    — month within year      (1 = Jan … 12 = Dec)
  day_of_month  INT    — day within month       (1 … 31)
  day_of_year   INT    — day within year        (1 … 366)
  hour_of_day   FLOAT  — fractional UTC hour    (e.g. 10.5 = 10h 30m)

Storing them at ingest time avoids all string-parsing overhead in catalog
histogramming and streaming partition scoring.

DIMENSION_CATALOG
-----------------
A single ordered dict that maps every scoreable dimension name to its
metadata.  This is the single source of truth consumed by:

  catalog.py         — which histogram arrays to compute and register
  stream.py          — which columns to SELECT, filter, and weight
  distribution.py    — which dimension names are valid

Adding a new dimension requires only a new entry here plus the corresponding
SQL extraction expression in catalog.py's _write_lst_histograms().
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pyproj import Transformer

# ============================================================
# Paths & constants
# ============================================================

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
_REPO = _SRC.parent

DEFAULT_DOWNLOADS = _SRC / "downloads"
DEFAULT_OUTPUT    = _SRC / "prepared_stream_data"

# H3 resolution — resolution 9 ≈ 0.17 km² per cell.
H3_RES = 9

# Temperature histogram: 2 °C bins from -10 °C to 60 °C (35 bins).
HIST_EDGES: list[float] = [float(v) for v in np.linspace(-10.0, 60.0, 36)]

# Timestamp edges — quarterly labels for year-level coarse scoring.
TIMESTAMP_EDGES_START_YEAR = 2000
TIMESTAMP_EDGES_END_YEAR   = 2025

# Year bins: one per calendar year (2000–2025 → 26 bins).
YEAR_EDGES: list[float] = [float(y) for y in range(2000, 2027)]

# Month-of-year bins: one per calendar month (12 bins).
# Edges 1.0–13.0 so floor(value - 1) gives 0-based bin index.
MONTH_OF_YEAR_EDGES: list[float] = [float(m) for m in range(1, 14)]

# Day-of-month bins: one per day (31 bins).
# Edges 1.0–32.0.
DAY_OF_MONTH_EDGES: list[float] = [float(d) for d in range(1, 33)]

# Day-of-year bins: 12 bins of ~30 days each (edges 1, 32, 60, … 366+).
# Coarser than daily to keep the array size manageable.
_DOY_BREAKPOINTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 367]
DAY_OF_YEAR_EDGES: list[float] = [float(d) for d in _DOY_BREAKPOINTS]  # 12 bins

# Hour-of-day bins: 24 half-open bins, one per UTC hour.
# Edges 0.0–24.0; fractional hours (e.g. 10.5) fall into bin floor(value).
HOUR_OF_DAY_EDGES: list[float] = [float(h) for h in range(0, 25)]

# Coordinate bin edges across Ghent.
LON_EDGES: list[float] = [float(v) for v in np.linspace(3.57, 3.86, 30)]   # 29 bins
LAT_EDGES: list[float] = [float(v) for v in np.linspace(50.96, 51.21, 26)] # 25 bins

# ============================================================
# DIMENSION_CATALOG
# ============================================================
# Maps every scoreable dimension to the metadata needed by catalog.py and
# stream.py.  Both modules iterate this dict so adding a new dimension here
# automatically propagates everywhere.
#
# Keys:
#   col          — column name in partition_statistics  (BIGINT[])
#   edges        — bin-edge list (float or str)
#   numeric      — True → edges are float (parsed from VARCHAR[] at load time)
#   sql_alias    — short alias used in the DuckDB SELECT built by _select_partitions
#
# The dict is ordered: dimensions are scored in declaration order, which is
# also the column order in partition_statistics.
DIMENSION_CATALOG: Dict[str, dict] = {
    "temperature": {
        "col":       "histogram_counts",
        "edges":     HIST_EDGES,
        "numeric":   True,
        "sql_alias": "temp_counts",
    },
    "timestamp": {
        # Quarterly string labels; kept for backwards-compatible coarse scoring.
        "col":       "timestamp_histogram_counts",
        "edges":     None,          # populated lazily by _compute_timestamp_edges()
        "numeric":   False,
        "sql_alias": "ts_counts",
    },
    "year": {
        "col":       "year_histogram_counts",
        "edges":     YEAR_EDGES,
        "numeric":   True,
        "sql_alias": "year_counts",
    },
    "month_of_year": {
        "col":       "month_histogram_counts",
        "edges":     MONTH_OF_YEAR_EDGES,
        "numeric":   True,
        "sql_alias": "month_counts",
    },
    "day_of_month": {
        "col":       "day_of_month_histogram_counts",
        "edges":     DAY_OF_MONTH_EDGES,
        "numeric":   True,
        "sql_alias": "dom_counts",
    },
    "day_of_year": {
        "col":       "day_of_year_histogram_counts",
        "edges":     DAY_OF_YEAR_EDGES,
        "numeric":   True,
        "sql_alias": "doy_counts",
    },
    "hour_of_day": {
        "col":       "hour_histogram_counts",
        "edges":     HOUR_OF_DAY_EDGES,
        "numeric":   True,
        "sql_alias": "hour_counts",
    },
    "longitude": {
        "col":       "longitude_histogram_counts",
        "edges":     LON_EDGES,
        "numeric":   True,
        "sql_alias": "lon_counts",
    },
    "latitude": {
        "col":       "latitude_histogram_counts",
        "edges":     LAT_EDGES,
        "numeric":   True,
        "sql_alias": "lat_counts",
    },
}

# ============================================================
# Write/ingest constants
# ============================================================

CHUNK_ROWS  = 200_000
BLOCK_LIMIT = None

CRS_LAMBERT = "EPSG:31370"
CRS_WGS84   = "EPSG:4326"
CRS_LAEA    = "EPSG:3035"

_lambert_to_wgs84 = Transformer.from_crs(CRS_LAMBERT, CRS_WGS84, always_xy=True)
_wgs84_to_lambert = Transformer.from_crs(CRS_WGS84, CRS_LAMBERT, always_xy=True)

# LST_COLUMNS — ordered column list for conn.append().
# The five time-component columns are stored alongside timestamp so that
# all temporal histogram computations and output DataFrames are self-contained.
LST_COLUMNS = [
    "longitude", "latitude",
    "aster_lst", "modis_lst", "ndvi",
    "image_id", "timestamp", "partition_key", "tile_id",
    "tile_h3_r8", "tile_h3_r7", "tile_rect_1km", "tile_rect_2km", "tile_ngi",
    "year", "month_of_year", "day_of_month", "day_of_year", "hour_of_day",
]

# Path to the NGI kaartbladversnijdingen shapefile (optional).
# Set to a Path when running ingest_lst to enable tile_ngi computation.
NGI_SHAPEFILE_PATH: Optional[Path] = DEFAULT_DOWNLOADS / "NGI" / "Kaartbladversnijdingen_NGI_numerieke_reeks_Shapefile" / "Shapefile"

APPEND_BATCH_ROWS = 1_000_000

# ============================================================
# Dataset registry
# ============================================================

DATASET_REGISTRY: dict[str, dict] = {
    "lst": {
        "description":           "LST — Land Surface Temperature (Landsat 5, ASTER/MODIS/NDVI, 30 m)",
        "db_file":               "lst.duckdb",
        "table":                 "lst",
        "store":                 "duckdb",
        "feature_columns":       [
            "aster_lst", "modis_lst", "ndvi", "image_id",
            "year", "month_of_year", "day_of_month", "day_of_year", "hour_of_day",
        ],
        "value_columns":         ["aster_lst", "modis_lst", "ndvi"],
        "lookup_method":         "driving",
        "temporal_behavior":     "evolving",
        "partition_column":      "partition_key",
        "source_resolution_m":   30.0,
        "is_driving":            True,
        "emissivity_modes":      ["aster", "modis", "ndvi"],
        "fallback_strategy":     "aster_modis_only",
    },
    "dhm": {
        "description":         "DHM — Digital Height Model (DHM1 ~2001-2004 + DHM2 ~2013-2015, 5 m, Belgian Lambert)",
        "db_file":             "dhm.duckdb",
        "table":               "dhm",
        "store":               "duckdb",
        "feature_columns":     ["elevation", "is_water", "dhm_year"],
        "value_column":        "elevation",
        "lookup_method":       "nearest_spatial",
        "temporal_behavior":   "static",
        "partition_column":    None,
        "year_column":         "dhm_year",
        "source_resolution_m": 5.0,
        "is_driving":          False,
    },
    "trees": {
        "description":         "Trees — Point-level tree inventory (CSV, WGS-84)",
        "db_file":             "trees.duckdb",
        "table":               "trees",
        "store":               "duckdb",
        "feature_columns":     ["sortiment", "hoogte", "aanlegjaar", "stamomtrek", "beheerfase", "genus"],
        "value_column":        None,
        "lookup_method":       "radius_spatial",
        "temporal_behavior":   "static",
        "partition_column":    None,
        "source_resolution_m": None,
        "is_driving":          False,
    },
    "urban_atlas": {
        "description":         "Urban Atlas — Land use/cover 2006/2012/2018/2021 (polygon geometry)",
        "db_file":             "spatial.db",
        "table":               "urban_atlas",
        "store":               "spatialite",
        "feature_columns":     ["luc_code", "ua_year", "area_m2", "geom"],
        "value_column":        None,
        "lookup_method":       "radius_spatial",
        "temporal_behavior":   "static",
        "partition_column":    None,
        "source_resolution_m": None,
        "is_driving":          False,
    },
    "wis": {
        "description":         "WIS — Ghent Road Information System (polygon geometry)",
        "db_file":             "spatial.db",
        "table":               "wis",
        "store":               "spatialite",
        "feature_columns":     ["bestemming", "materiaalsoort", "area_m2", "geom"],
        "value_column":        None,
        "lookup_method":       "radius_spatial",
        "temporal_behavior":   "static",
        "partition_column":    None,
        "source_resolution_m": None,
        "is_driving":          False,
    },
}

# ============================================================
# Urban Atlas source file registry
# ============================================================

UA_SOURCES: dict[int, dict] = {
    2006: {
        "globs":          ["BE003L2_GENT/*/Shapefiles/*.shp"],
        "luc_candidates": ["CODE2006"],
        "layer":          None,
    },
    2012: {
        "globs":          ["BE003L2_GENT_UA2012_revised_v021/*/Data/*.gpkg"],
        "luc_candidates": ["code_2012"],
        "layer":          "BE003L2_GENT_UA2012_revised",
    },
    2018: {
        "globs":          ["*/CLMS_UA_LCU_S2018*/*.fgb"],
        "luc_candidates": ["code_2018"],
        "layer":          None,
    },
    2021: {
        "globs":          ["*/CLMS_UA_LCU_S2021*/*.fgb"],
        "luc_candidates": ["code_2021"],
        "layer":          None,
    },
}


# ============================================================
# Bin-edge generators
# ============================================================

def _compute_timestamp_edges(
    start_year: int = TIMESTAMP_EDGES_START_YEAR,
    end_year:   int = TIMESTAMP_EDGES_END_YEAR,
) -> list[str]:
    """
    Generate quarterly bin labels "YYYY-Q{1..4}" for every year in range.

    Used for the coarse 'timestamp' dimension.  For finer temporal control
    use 'year', 'month_of_year', 'day_of_year', or 'hour_of_day'.
    """
    edges: list[str] = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            edges.append(f"{year:04d}-Q{quarter}")
    return edges


def get_dimension_edges(dim: str) -> list:
    """
    Return the bin-edge list for a named dimension.

    For 'timestamp' the edges are generated lazily (string labels).
    For all other dimensions the edges come directly from the module-level
    constant referenced in DIMENSION_CATALOG.
    """
    if dim == "timestamp":
        return _compute_timestamp_edges()
    meta = DIMENSION_CATALOG.get(dim)
    if meta is None:
        raise KeyError(f"Unknown dimension '{dim}'. "
                       f"Valid dimensions: {list(DIMENSION_CATALOG)}")
    edges = meta["edges"]
    if edges is None:
        raise RuntimeError(f"Dimension '{dim}' has no edges defined.")
    return edges