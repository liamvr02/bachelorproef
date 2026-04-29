"""
ingest/ingest_spatial.py
========================
Ingestors for polygon datasets stored in SpatiaLite:
  - Urban Atlas  ->  spatial.db :: urban_atlas
  - WIS          ->  spatial.db :: wis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from tqdm import tqdm

from config import CHUNK_ROWS, CRS_LAEA, CRS_WGS84, UA_SOURCES
from db import drop_spatialite_table, open_spatialite

log = logging.getLogger("ingest.spatial")


# ============================================================
# Shared geometry helpers
# ============================================================

def _to_multipolygon_wkt(geom) -> str:
    """Coerce any Polygon/MultiPolygon to a MULTIPOLYGON WKT string."""
    if geom.geom_type == "Polygon":
        return ShapelyMultiPolygon([geom]).wkt
    return geom.wkt


def _keep_valid_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop null, empty, invalid, or non-polygon geometries."""
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()


def _add_area_m2(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute polygon area in equal-area CRS and store in area_m2 column."""
    gdf_ea        = gdf.to_crs(CRS_LAEA)
    gdf["area_m2"] = gdf_ea.geometry.area.values
    return gdf


# ============================================================
# Urban Atlas
# ============================================================

def _find_ua_file(year: int, base_dir: Path) -> Optional[Path]:
    for glob in UA_SOURCES[year]["globs"]:
        candidates = list(base_dir.glob(glob))
        if candidates:
            return candidates[0]
    return None


def _normalise_luc(gdf: gpd.GeoDataFrame, candidates: list[str]) -> gpd.GeoDataFrame:
    """Rename the land-use code column to 'luc_code', matching case-insensitively."""
    lower_to_actual = {c.lower(): c for c in gdf.columns}
    for cand in candidates:
        actual = lower_to_actual.get(cand.lower())
        if actual is not None:
            return gdf.rename(columns={actual: "luc_code"})
    non_geom = [c for c in gdf.columns if c != "geometry"]
    log.warning(
        "No luc_code candidate matched %s.  Available columns: %s.  "
        "Using '%s' as fallback - add the correct name to UA_SOURCES[year]['luc_candidates'].",
        candidates, list(gdf.columns), non_geom[0] if non_geom else "none",
    )
    if non_geom:
        return gdf.rename(columns={non_geom[0]: "luc_code"})
    return gdf


def ingest_urban_atlas(downloads: Path, output: Path) -> int:
    """
    Ingest Urban Atlas as full geometries into SpatiaLite.

    No rasterization. No loss of shape.
    Enables exact spatial queries like:
        "area of luc_code X within radius R of (lon, lat)"

    Schema:
        id INTEGER PK
        luc_code TEXT
        ua_year INTEGER
        area_m2 REAL
        geom MULTIPOLYGON (WGS84)

    Spatial index: R-tree on geom
    """
    ua_root = downloads / "urban_atlas_extracted"
    if not ua_root.exists():
        log.warning("Urban Atlas source directory not found: %s - skipping", ua_root)
        return 0

    conn = open_spatialite(output / "spatial.db")
    drop_spatialite_table(conn, "urban_atlas")

    conn.execute("""
        CREATE TABLE urban_atlas (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            luc_code  TEXT,
            ua_year   INTEGER,
            area_m2   REAL
        )
    """)
    conn.execute("SELECT AddGeometryColumn('urban_atlas', 'geom', 4326, 'MULTIPOLYGON', 'XY')")
    conn.execute("SELECT CreateSpatialIndex('urban_atlas', 'geom')")
    conn.commit()

    total = 0

    for year in sorted(UA_SOURCES.keys()):
        source = _find_ua_file(year, ua_root)
        if source is None:
            log.warning("Urban Atlas %d: no file found - skipping", year)
            continue

        log.info("Urban Atlas %d: loading %s ...", year, source.name)

        layer        = UA_SOURCES[year].get("layer")
        read_kwargs  = {"layer": layer} if layer is not None else {}
        gdf          = gpd.read_file(str(source), **read_kwargs)
        gdf          = _keep_valid_polygons(gdf)
        gdf          = _add_area_m2(gdf)
        gdf          = gdf.to_crs(CRS_WGS84)
        gdf          = _normalise_luc(gdf, UA_SOURCES[year]["luc_candidates"])

        if "luc_code" not in gdf.columns:
            log.error("Urban Atlas %d: cannot identify luc_code column - skipping", year)
            continue

        gdf["_geom_wkt"] = gdf.geometry.apply(_to_multipolygon_wkt)

        rows = []
        for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc=f"UA {year}"):
            rows.append((
                str(row.get("luc_code", "unknown")),
                int(year),
                float(row.get("area_m2", 0.0)),
                row["_geom_wkt"],
            ))
            if len(rows) >= CHUNK_ROWS:
                conn.executemany("""
                    INSERT INTO urban_atlas (luc_code, ua_year, area_m2, geom)
                    VALUES (?, ?, ?, GeomFromText(?, 4326))
                """, rows)
                conn.commit()
                total += len(rows)
                rows = []

        if rows:
            conn.executemany("""
                INSERT INTO urban_atlas (luc_code, ua_year, area_m2, geom)
                VALUES (?, ?, ?, GeomFromText(?, 4326))
            """, rows)
            conn.commit()
            total += len(rows)

        log.info("Urban Atlas %d: %d polygons", year, len(gdf))

    conn.execute("CREATE INDEX IF NOT EXISTS idx_ua_luc_code ON urban_atlas(luc_code)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ua_year     ON urban_atlas(ua_year)")
    conn.commit()

    conn.close()
    return total


# ============================================================
# WIS
# ============================================================

def ingest_wis(downloads: Path, output: Path) -> int:
    """
    Ingest the Ghent Road Information System (WIS) GeoJSON into SpatiaLite.

    Source: downloads/wis/wis.geojson
    Skips features with null geometry and sentinel rows with 'geometry' as a
    property key (export artifact).

    Schema:
        id             INTEGER PK
        bestemming     TEXT      road surface purpose
        materiaalsoort TEXT      surface material (may be null)
        area_m2        REAL      polygon area in equal-area m²
        geom           MULTIPOLYGON (WGS84)

    Spatial index: R-tree on geom
    """
    geojson_path = downloads / "wis" / "wis.geojson"
    if not geojson_path.exists():
        log.warning("WIS GeoJSON not found: %s - skipping", geojson_path)
        return 0

    log.info("WIS: loading %s ...", geojson_path.name)
    gdf = gpd.read_file(str(geojson_path))
    gdf = _keep_valid_polygons(gdf)
    log.info("WIS: %d valid polygons after filtering", len(gdf))

    gdf = _add_area_m2(gdf)
    gdf = gdf.to_crs(CRS_WGS84)

    conn = open_spatialite(output / "spatial.db")
    drop_spatialite_table(conn, "wis")

    conn.execute("""
        CREATE TABLE wis (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            bestemming     TEXT,
            materiaalsoort TEXT,
            area_m2        REAL
        )
    """)
    conn.execute("SELECT AddGeometryColumn('wis', 'geom', 4326, 'MULTIPOLYGON', 'XY')")
    conn.execute("SELECT CreateSpatialIndex('wis', 'geom')")
    conn.commit()

    gdf["_geom_wkt"] = gdf.geometry.apply(_to_multipolygon_wkt)

    rows  = []
    total = 0
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="WIS"):
        rows.append((
            str(row.get("bestemming", "") or ""),
            str(row.get("materiaalsoort", "") or "") if row.get("materiaalsoort") else None,
            float(row.get("area_m2", 0.0)),
            row["_geom_wkt"],
        ))
        if len(rows) >= CHUNK_ROWS:
            conn.executemany("""
                INSERT INTO wis (bestemming, materiaalsoort, area_m2, geom)
                VALUES (?, ?, ?, GeomFromText(?, 4326))
            """, rows)
            conn.commit()
            total += len(rows)
            rows = []

    if rows:
        conn.executemany("""
            INSERT INTO wis (bestemming, materiaalsoort, area_m2, geom)
            VALUES (?, ?, ?, GeomFromText(?, 4326))
        """, rows)
        conn.commit()
        total += len(rows)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_wis_bestemming     ON wis(bestemming)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wis_materiaalsoort ON wis(materiaalsoort)")
    conn.commit()

    conn.close()
    log.info("WIS: %d polygons ingested", total)
    return total
