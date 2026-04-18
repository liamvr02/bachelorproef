"""
ingest/config.py
================
Shared constants, path defaults, histogram bin edges, and the dataset registry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyproj import Transformer

# ============================================================
# Paths & constants
# ============================================================

_HERE = Path(__file__).resolve().parent   # /src/processing/
_SRC  = _HERE.parent                             # /src/
_REPO = _SRC.parent                              # repo root

DEFAULT_DOWNLOADS = _SRC / "downloads"
DEFAULT_OUTPUT    = _SRC / "prepared_stream_data"

# H3 resolution used for spatial bucketing and Urban Atlas rasterization.
# Resolution 9 ≈ 0.17 km² per cell  (good match for 30 m LST pixels).
H3_RES = 9

# Temperature histogram bin edges stored in the catalog for distribution-
# aware streaming.  2 °C bins from -10 °C to 60 °C  (35 bins).
HIST_EDGES: list[float] = [float(v) for v in np.linspace(-10.0, 60.0, 36)]

# Timestamp histogram bin edges: seasonal bins for all years in data range.
TIMESTAMP_EDGES_START_YEAR = 2000
TIMESTAMP_EDGES_END_YEAR   = 2025

# Coordinate histogram bin edges: 0.01 degree precision over Ghent bounds.
# Ghent bounds (from shapely.geometry.Polygon.bounds):
#   Lon: 3.5797616 to 3.8493413 (span ~0.27°)
#   Lat: 50.9795422 to 51.188891 (span ~0.209°)
# Bins: 0.01° precision gives ~28 lon bins and ~23 lat bins.
LON_EDGES: list[float] = [float(v) for v in np.linspace(3.57, 3.86, 30)]  # 29 bins
LAT_EDGES: list[float] = [float(v) for v in np.linspace(50.96, 51.21, 26)]  # 25 bins

# Parquet/DuckDB write settings
CHUNK_ROWS  = 200_000   # rows per DuckDB COPY batch
BLOCK_LIMIT = None      # set to an int during testing to cap raster blocks

# CRS constants
CRS_LAMBERT = "EPSG:31370"
CRS_WGS84   = "EPSG:4326"
CRS_LAEA    = "EPSG:3035"   # equal-area for polygon area computation

_lambert_to_wgs84 = Transformer.from_crs(CRS_LAMBERT, CRS_WGS84, always_xy=True)

# Column order — single source of truth for conn.append() (positional mapping).
# LST table now combines ASTER, MODIS, and NDVI into unified rows:
#   - aster_lst, modis_lst: FLOAT (nullable) — LST values from respective emissivity products
#   - ndvi: FLOAT (nullable) — NDVI vegetation index; never used as LST fallback
LST_COLUMNS  = ["longitude", "latitude", "aster_lst", "modis_lst", "ndvi",
                 "image_id", "timestamp", "partition_key", "tile_id"]

APPEND_BATCH_ROWS = 1_000_000   # rows buffered before each conn.append()

# ============================================================
# Dataset registry
# ============================================================

DATASET_REGISTRY: dict[str, dict] = {
    "lst": {
        "description":           "LST — Land Surface Temperature (Landsat 5, ASTER/MODIS/NDVI, 30 m)",
        "db_file":               "lst.duckdb",
        "table":                 "lst",
        "store":                 "duckdb",
        "feature_columns":       ["aster_lst", "modis_lst", "ndvi", "image_id"],
        "value_columns":         ["aster_lst", "modis_lst", "ndvi"],
        "lookup_method":         "driving",
        "temporal_behavior":     "evolving",
        "partition_column":      "partition_key",
        "source_resolution_m":   30.0,
        "is_driving":            True,
        "emissivity_modes":      ["aster", "modis", "ndvi"],
        "fallback_strategy":     "aster_modis_only",
    },
    "dhm1": {
        "description":         "DHM1 — Digital Height Model 1 (~2007, 5 m, Belgian Lambert)",
        "db_file":             "dhm1.duckdb",
        "table":               "dhm1",
        "store":               "duckdb",
        "feature_columns":     ["elevation", "is_water"],
        "value_column":        "elevation",
        "lookup_method":       "nearest_spatial",
        "temporal_behavior":   "static",
        "partition_column":    None,
        "source_resolution_m": 5.0,
        "is_driving":          False,
    },
    "dhm2": {
        "description":         "DHM2 — Digital Height Model 2 (~2015, 5 m, Belgian Lambert)",
        "db_file":             "dhm2.duckdb",
        "table":               "dhm2",
        "store":               "duckdb",
        "feature_columns":     ["elevation", "is_water"],
        "value_column":        "elevation",
        "lookup_method":       "nearest_spatial",
        "temporal_behavior":   "static",
        "partition_column":    None,
        "source_resolution_m": 5.0,
        "is_driving":          False,
    },
    "trees": {
        "description":         "Trees — Point-level tree inventory (CSV, WGS-84)",
        "db_file":             "trees.duckdb",
        "table":               "trees",
        "store":               "duckdb",
        "feature_columns":     ["species", "height_m", "planting_year", "trunk_circumference_cm"],
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


def _compute_timestamp_edges(
    start_year: int = TIMESTAMP_EDGES_START_YEAR,
    end_year:   int = TIMESTAMP_EDGES_END_YEAR,
) -> list[str]:
    """Generate seasonal bin edges (YYYY-Q1/Q2/Q3/Q4) for all years."""
    edges = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            edges.append(f"{year:04d}-Q{quarter}")
    return edges
