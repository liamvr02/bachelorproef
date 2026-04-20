"""
poly_raster.py  -  /src/stream/poly_raster.py
==============================================
Precomputed polygon-fraction grid (_PolyRaster) and the supporting
geometry helpers used to build and query it.

Includes:
  _PolyRaster          - regular lon/lat grid of fraction values (UA + WIS)
  _geom_cache          - module-level WKB -> Shapely geometry cache
  _decode_wkb          - cached WKB decoder (no GEOS round-trip via SpatiaLite)
  _ua_make_circle      - degree-space circle approximation
  _ua_compute_fraction - covered area / circle area via Shapely
  _ua_fetch_candidates - SpatiaLite R-tree pre-filter for UA polygons
  _wis_fetch_candidates- SpatiaLite R-tree pre-filter for WIS polygons
  _ua_fetch_all_in_bbox- bulk UA polygon fetch for raster precomputation
  _wis_fetch_all_in_bbox-bulk WIS polygon fetch for raster precomputation
  _rasterise_layer     - fill one raster layer from a pre-fetched blob list
"""

from __future__ import annotations

import logging
import math
import sqlite3
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from geo import _LAT_DEG_PER_M, _LON_DEG_PER_M

log = logging.getLogger("stream")


# ---------------------------------------------------------------------------
# Module-level WKB cache  (live batch path only)
# ---------------------------------------------------------------------------
# Used by _ua_compute_fraction / _wis_fetch_candidates during streaming to
# avoid re-decoding the same polygon blob on repeated per-row queries.
# NOT used during raster precomputation — rasterisation decodes blobs locally
# per layer so memory is released between layers (see _rasterise_layer).
_geom_cache: Dict[bytes, object] = {}


def _decode_wkb(blob: bytes):
    if blob not in _geom_cache:
        from shapely import wkb as shapely_wkb
        _geom_cache[blob] = shapely_wkb.loads(blob)
    return _geom_cache[blob]


# ---------------------------------------------------------------------------
# Circle construction
# ---------------------------------------------------------------------------

def _ua_make_circle(lon: float, lat: float, radius_m: float):
    """
    Return a Shapely polygon in WGS-84 degree-space approximating a circle of
    *radius_m* metres centred at (lon, lat).

    Build a unit circle around the origin, scale each axis by the local
    metres-to-degrees conversion, then translate to (lon, lat).  Numerator and
    denominator of the fraction both use this geometry, so the approximation
    cancels and the resulting fraction is accurate.
    """
    from shapely.geometry import Point
    from shapely import affinity
    r_lat = radius_m * _LAT_DEG_PER_M   # radius in latitude-degrees
    r_lon = radius_m * _LON_DEG_PER_M   # radius in longitude-degrees (latitude-corrected)
    unit_circle = Point(0.0, 0.0).buffer(1.0, resolution=64)
    ellipse     = affinity.scale(unit_circle, xfact=r_lon, yfact=r_lat, origin=(0.0, 0.0, 0.0))
    return affinity.translate(ellipse, xoff=lon, yoff=lat)


# ---------------------------------------------------------------------------
# Shapely fraction computation
# ---------------------------------------------------------------------------

def _ua_compute_fraction(
    wkb_blobs: list,
    circle,   # shapely.geometry - pre-built by caller
) -> float:
    """
    Intersect *wkb_blobs* (raw WKB from SpatiaLite) with *circle* in Shapely.
    Returns covered_area / circle_area, clamped to [0, 1].

    Uses a prepared circle for repeated intersects() checks (cheap GEOS
    predicate) before computing the full intersection area.  Decoded geometries
    are cached module-wide so each polygon is deserialised at most once.
    """
    from shapely.prepared import prep

    circle_area = circle.area
    if circle_area == 0.0:
        return 0.0

    prepared_circle = prep(circle)
    covered = 0.0
    for blob in wkb_blobs:
        try:
            poly = _decode_wkb(blob)
            if not prepared_circle.intersects(poly):
                continue
            covered += circle.intersection(poly).area
        except Exception:
            continue

    return min(covered / circle_area, 1.0)


# ---------------------------------------------------------------------------
# SpatiaLite candidate fetchers (R-tree bbox pre-filter, no GEOS)
# ---------------------------------------------------------------------------

def _ua_fetch_candidates(
    db: sqlite3.Connection,
    lon: float,
    lat: float,
    radius_m: float,
    luc_code: str,
    ua_year: Optional[int],
) -> list:
    """
    Return raw WKB bytes for every urban_atlas polygon that:
      - matches luc_code (and ua_year if given)
      - whose bounding box overlaps the query circle's bounding box

    The bounding-box check is an R-tree pre-filter only; exact containment /
    intersection is handled by the caller in Shapely.
    """
    dlat = radius_m * _LAT_DEG_PER_M
    dlon = radius_m * _LON_DEG_PER_M
    year_filter = f"AND ua_year = {int(ua_year)}" if ua_year is not None else ""

    sql = f"""
        SELECT AsBinary(geom)
        FROM urban_atlas
        WHERE luc_code = ?
          {year_filter}
          AND id IN (
              SELECT id FROM SpatialIndex
              WHERE f_table_name = 'urban_atlas'
                AND search_frame = BuildMbr(?, ?, ?, ?, 4326)
          )
    """
    params = (luc_code, lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    return [r[0] for r in db.execute(sql, params).fetchall() if r[0] is not None]


def _wis_fetch_candidates(
    db: sqlite3.Connection,
    lon: float,
    lat: float,
    radius_m: float,
    attr_col: str,
    attr_val: str,
) -> list:
    """
    Return raw WKB bytes for every WIS polygon that:
      - has attr_col = attr_val
      - whose bounding box overlaps the query circle's bounding box

    Mirrors _ua_fetch_candidates exactly; the R-tree pre-filter is identical.
    Exact intersection is handled by the caller in Shapely (_ua_compute_fraction
    is reused since the computation — covered area / circle area — is identical).
    """
    dlat = radius_m * _LAT_DEG_PER_M
    dlon = radius_m * _LON_DEG_PER_M
    sql = f"""
        SELECT AsBinary(geom)
        FROM wis
        WHERE {attr_col} = ?
          AND id IN (
              SELECT id FROM SpatialIndex
              WHERE f_table_name = 'wis'
                AND search_frame = BuildMbr(?, ?, ?, ?, 4326)
          )
    """
    params = (attr_val, lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    return [r[0] for r in db.execute(sql, params).fetchall() if r[0] is not None]


def _ua_fetch_all_in_bbox(
    db: sqlite3.Connection,
    luc_code: str,
    ua_year: Optional[int],
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
) -> list:
    """
    Fetch all Urban Atlas WKB blobs for (luc_code, ua_year) whose bbox
    overlaps (lon_min, lat_min, lon_max, lat_max).  Used during raster
    precomputation to load the entire grid extent in one SQL query.
    """
    year_filter = f"AND ua_year = {int(ua_year)}" if ua_year is not None else ""
    sql = f"""
        SELECT AsBinary(geom)
        FROM urban_atlas
        WHERE luc_code = ?
          {year_filter}
          AND id IN (
              SELECT id FROM SpatialIndex
              WHERE f_table_name = 'urban_atlas'
                AND search_frame = BuildMbr(?, ?, ?, ?, 4326)
          )
    """
    params = (luc_code, lon_min, lat_min, lon_max, lat_max)
    return [r[0] for r in db.execute(sql, params).fetchall() if r[0] is not None]


def _wis_fetch_all_in_bbox(
    db: sqlite3.Connection,
    attr_col: str,
    attr_val: str,
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
) -> list:
    """
    Fetch all WIS WKB blobs for (attr_col=attr_val) in the given bbox.
    Used during raster precomputation.
    """
    sql = f"""
        SELECT AsBinary(geom)
        FROM wis
        WHERE {attr_col} = ?
          AND id IN (
              SELECT id FROM SpatialIndex
              WHERE f_table_name = 'wis'
                AND search_frame = BuildMbr(?, ?, ?, ?, 4326)
          )
    """
    params = (attr_val, lon_min, lat_min, lon_max, lat_max)
    return [r[0] for r in db.execute(sql, params).fetchall() if r[0] is not None]


# ---------------------------------------------------------------------------
# Raster fill helper
# ---------------------------------------------------------------------------


def _rasterise_layer(
    arr: "np.ndarray",
    raster: "_PolyRaster",
    all_blobs: list,
    radius_m: float,
    desc: str = "",
    simplify_tolerance: float = 0.00005,
    circle_resolution: int = 16,
) -> int:
    """
    Fill one raster layer using a pre-fetched list of polygon WKB blobs.

    Parameters
    ----------
    arr                : output float32 array, shape (n_lon, n_lat) — filled in-place.
    raster             : _PolyRaster instance supplying grid coordinates.
    all_blobs          : raw WKB bytes from SpatiaLite, one per source polygon.
    radius_m           : query-circle radius in metres.
    desc               : tqdm label.
    simplify_tolerance : Shapely simplify tolerance in degrees applied once per
                         polygon after decoding.  Default 0.00005deg ~ 5 m.
                         Reduces vertex count on detailed road polygons
                         (UA 12220) while keeping fraction error well below 1%.
                         Set 0 to disable.
    circle_resolution  : vertices used to approximate the query ellipse.
                         Default 16 (was 32); error vs a true circle is ~1.0%,
                         within the accepted 5% MSE margin.  Halving from 32
                         roughly halves GEOS intersection cost per candidate.

    Algorithm
    ---------
    1.  Decode WKB locally; simplify each polygon once to reduce vertex count.
    2.  Build one STRtree.  Precompute per-polygon bounding boxes as a numpy
        float64 array for fast column and candidate filtering.
    3.  Selectively prep polygons: only polygons whose bbox spans > 2 grid
        columns are wrapped in PreparedGeometry.  Tiny slivers touched by
        only 1-2 cells do not benefit from the prep overhead.
    4.  Build a template ellipse at origin (circle_resolution vertices).
    5.  Per column (lon):
          a.  Numpy bbox check: if no polygon x-range overlaps this column
              band, write 0.0 for the entire column (zero GEOS calls).
          b.  Compute the set of polygon indices eligible for this column
              (O(1) membership test inside the cell loop).
    6.  Per cell (lat) within a non-empty column:
          a.  Query STRtree with a bbox rectangle, not the ellipse — avoids
              creating a Shapely geometry for cells with 0 candidates.
          b.  Filter candidates against the column set (O(1) set lookup).
          c.  Build the translated ellipse only when candidates exist.
          d.  For each candidate: contains fast-path -> bbox reject ->
              full intersection.  Early exit when covered >= circle_area.
    7.  Explicit del of all GEOS objects before return.

    Returns the count of non-zero cells (for logging).
    """
    from shapely import affinity, wkb as shapely_wkb
    from shapely.geometry import box as shapely_box, Point
    from shapely.strtree import STRtree
    from shapely.prepared import prep

    n_lon, n_lat = raster.n_lon, raster.n_lat

    if not all_blobs:
        arr[:, :] = 0.0
        return 0

    # -- Step 1: decode blobs locally, optional simplify ----------------------
    geoms = []
    for blob in all_blobs:
        try:
            g = shapely_wkb.loads(blob)
            if simplify_tolerance > 0:
                g = g.simplify(simplify_tolerance, preserve_topology=True)
            geoms.append(g)
        except Exception:
            continue
    if not geoms:
        arr[:, :] = 0.0
        return 0

    # -- Step 2: STRtree + numpy bbox arrays ----------------------------------
    tree = STRtree(geoms)
    poly_bounds = np.array([g.bounds for g in geoms], dtype=np.float64)
    poly_minx = poly_bounds[:, 0]
    poly_miny = poly_bounds[:, 1]
    poly_maxx = poly_bounds[:, 2]
    poly_maxy = poly_bounds[:, 3]

    # -- Step 3: selective PreparedGeometry -----------------------------------
    # Only prep polygons wide enough to appear in more than 2 grid columns.
    # Tiny slivers do not recoup the prep cost.
    prep_threshold_lon = 2.0 * raster.step_lon
    prepared_geoms = [
        prep(g) if (poly_maxx[i] - poly_minx[i]) > prep_threshold_lon else None
        for i, g in enumerate(geoms)
    ]

    # -- Step 4: template ellipse at origin -----------------------------------
    r_lat = radius_m * _LAT_DEG_PER_M
    r_lon = radius_m * _LON_DEG_PER_M
    template = Point(0.0, 0.0).buffer(1.0, resolution=circle_resolution)
    template = affinity.scale(template, xfact=r_lon, yfact=r_lat,
                              origin=(0.0, 0.0, 0.0))
    circle_area = template.area

    n_nonzero  = 0
    tqdm_every = max(1, n_lon // 100)

    with tqdm(total=n_lon, desc=desc, unit="col",
              position=1, leave=False, dynamic_ncols=True) as col_bar:
        for ix in range(n_lon):
            lon_q  = raster.grid_lon(ix)
            col_lo = lon_q - r_lon
            col_hi = lon_q + r_lon

            # -- Step 5a: column bbox skip ------------------------------------
            col_mask = (poly_minx <= col_hi) & (poly_maxx >= col_lo)
            if not col_mask.any():
                arr[ix, :] = 0.0
                if ix % tqdm_every == 0:
                    col_bar.update(tqdm_every)
                continue

            col_geom_set = set(np.where(col_mask)[0].tolist())

            # -- Step 6: per-cell work ----------------------------------------
            for iy in range(n_lat):
                lat_q       = raster.grid_lat(iy)
                cell_lo_lat = lat_q - r_lat
                cell_hi_lat = lat_q + r_lat

                # 6a: bbox rectangle query — no ellipse object created yet
                cell_box = shapely_box(col_lo, cell_lo_lat, col_hi, cell_hi_lat)
                raw_candidates = tree.query(cell_box, predicate="intersects")

                # 6b: filter to column-eligible polygons (O(1) set lookup)
                candidates = [ci for ci in raw_candidates if ci in col_geom_set]
                if not candidates:
                    arr[ix, iy] = 0.0
                    continue

                # Only build translated ellipse when there are real candidates
                circle  = affinity.translate(template, xoff=lon_q, yoff=lat_q)
                covered = 0.0

                for ci in candidates:
                    pg = prepared_geoms[ci]

                    # 6c: contains fast-path (only for prepped large polygons)
                    if pg is not None and pg.contains(circle):
                        covered = circle_area
                        break

                    # 6d: cheap numpy bbox reject before full intersection
                    if (poly_maxx[ci] < col_lo or poly_minx[ci] > col_hi or
                            poly_maxy[ci] < cell_lo_lat or
                            poly_miny[ci] > cell_hi_lat):
                        continue

                    try:
                        covered += geoms[ci].intersection(circle).area
                    except Exception:
                        continue

                    # 6e: early saturation exit
                    if covered >= circle_area:
                        covered = circle_area
                        break

                arr[ix, iy] = float(covered / circle_area)
                if arr[ix, iy] > 0.0:
                    n_nonzero += 1

            if ix % tqdm_every == 0:
                col_bar.update(tqdm_every)

    # -- Step 7: explicit cleanup so memory does not compound -----------------
    del geoms, tree, prepared_geoms, poly_bounds
    del poly_minx, poly_miny, poly_maxx, poly_maxy

    return n_nonzero



# ---------------------------------------------------------------------------
# _PolyRaster
# ---------------------------------------------------------------------------

class _PolyRaster:
    """
    A precomputed regular lon/lat grid of polygon-fraction values.

    Built once at stream init (see StreamConfig._build_poly_rasters()) by
    running the exact same _ua_fetch_candidates + _ua_compute_fraction
    pipeline that the live batch path uses, but over a dense grid of
    synthetic query points covering the bounding box of all selected LST
    partitions.

    Grid design
    -----------
    Resolution: raster_resolution_m (default 15 m — half the 30 m LST pixel
    spacing so every LST pixel maps to the nearest precomputed point within
    ~7.5 m, well inside the acceptable margin for any radius >= 100 m).

    The grid is stored as a dict of numpy float32 arrays, one per layer key.
    Array indices correspond to (lon_idx, lat_idx) integer pairs derived from:

        lon_idx = round((lon - lon0) / step_lon)
        lat_idx = round((lat - lat0) / step_lat)

    Lookup is O(1): snap the query coordinate to the nearest grid index,
    clamp to bounds, index the array.

    Temporal layers (UA)
    --------------------
    UA has four survey years (2006, 2012, 2018, 2021).  For each
    (luc_code, ua_year) combination the raster holds one layer.  At query
    time the caller passes the LST row timestamp and _PolyRaster selects the
    layer whose ua_year is the last_previous year <= the LST year.

    Static layers (WIS)
    -------------------
    WIS has a single timestamp; layer key is the attribute value string alone.
    """

    UA_YEARS = [2006, 2012, 2018, 2021]

    def __init__(
        self,
        lon0: float,
        lat0: float,
        step_lon: float,
        step_lat: float,
        n_lon: int,
        n_lat: int,
        resolution_m: float,
    ):
        self.lon0         = lon0
        self.lat0         = lat0
        self.step_lon     = step_lon
        self.step_lat     = step_lat
        self.n_lon        = n_lon
        self.n_lat        = n_lat
        self.resolution_m = resolution_m
        # layers: {layer_key: np.ndarray shape (n_lon, n_lat) float32}
        self._layers: Dict[str, np.ndarray] = {}
        self._coverage: Dict[str, int] = {}   # layer_key -> filled cell count

    # ------------------------------------------------------------------
    # Grid coordinate helpers
    # ------------------------------------------------------------------
    def _snap(self, lon: float, lat: float) -> Tuple[int, int]:
        """Snap a WGS-84 coordinate to the nearest grid index pair."""
        ix = int(round((lon - self.lon0) / self.step_lon))
        iy = int(round((lat - self.lat0) / self.step_lat))
        ix = max(0, min(self.n_lon - 1, ix))
        iy = max(0, min(self.n_lat - 1, iy))
        return ix, iy

    def grid_lon(self, ix: int) -> float:
        return self.lon0 + ix * self.step_lon

    def grid_lat(self, iy: int) -> float:
        return self.lat0 + iy * self.step_lat

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------
    def add_layer(self, key: str) -> np.ndarray:
        """Create and register a new layer filled with NaN."""
        arr = np.full((self.n_lon, self.n_lat), np.nan, dtype=np.float32)
        self._layers[key] = arr
        self._coverage[key] = 0
        return arr

    def set_cell(self, key: str, ix: int, iy: int, value: float) -> None:
        self._layers[key][ix, iy] = value
        self._coverage[key] += 1

    def has_layer(self, key: str) -> bool:
        return key in self._layers

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def lookup(self, lon: float, lat: float, layer_key: str) -> Optional[float]:
        """
        Return the precomputed fraction for (lon, lat) from the given layer.
        Returns None if the layer is missing or the cell was never filled.
        """
        arr = self._layers.get(layer_key)
        if arr is None:
            return None
        ix, iy = self._snap(lon, lat)
        v = float(arr[ix, iy])
        return None if math.isnan(v) else v

    def lookup_ua_last_previous(
        self, lon: float, lat: float, luc_code: str, lst_year: int
    ) -> Optional[float]:
        """
        Return the UA fraction for the last_previous survey year <= lst_year.
        Layer keys are formatted as "{luc_code}:{ua_year}".
        Returns None if no layer is available for any year <= lst_year.
        """
        best_year = None
        for y in self.UA_YEARS:
            if y <= lst_year:
                key = f"{luc_code}:{y}"
                if self.has_layer(key):
                    best_year = y
        if best_year is None:
            return None
        return self.lookup(lon, lat, f"{luc_code}:{best_year}")

    def lookup_wis(
        self, lon: float, lat: float, attr_val: str
    ) -> Optional[float]:
        """Return the WIS fraction for a given attribute value (static)."""
        return self.lookup(lon, lat, f"wis:{attr_val}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def log_summary(self) -> None:
        total_cells = self.n_lon * self.n_lat
        log.info(
            "PolyRaster: %.0f m grid, %d x %d cells (%.2f x %.2f deg), "
            "%d layers, bbox lon=[%.4f,%.4f] lat=[%.4f,%.4f]",
            self.resolution_m, self.n_lon, self.n_lat,
            self.n_lon * self.step_lon, self.n_lat * self.step_lat,
            len(self._layers),
            self.lon0, self.lon0 + (self.n_lon - 1) * self.step_lon,
            self.lat0, self.lat0 + (self.n_lat - 1) * self.step_lat,
        )
        for key, count in self._coverage.items():
            pct = 100.0 * count / total_cells if total_cells else 0
            log.debug("PolyRaster layer '%s': %d / %d cells filled (%.1f%%)",
                      key, count, total_cells, pct)