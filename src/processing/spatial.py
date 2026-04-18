"""
ingest/spatial.py
=================
Spatial utility helpers: raster block streaming, polygon filtering,
H3 indexing, and Ghent boundary caches.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

import h3
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.transform
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shapely_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gathering"))
from ghent_polygon import get_ghent_outers, get_ghent_convex_hull

from config import H3_RES, BLOCK_LIMIT

log = logging.getLogger("spatial")

# ============================================================
# Polygon caches (loaded on-demand)
# ============================================================

_GHENT_POLYGON_EXACT:       Optional[Polygon] = None
_GHENT_POLYGON_CONVEX_HULL: Optional[Polygon] = None


def get_ghent_exact_polygon() -> Polygon:
    """Load and cache the exact Ghent polygon."""
    global _GHENT_POLYGON_EXACT
    if _GHENT_POLYGON_EXACT is None:
        _GHENT_POLYGON_EXACT = Polygon(get_ghent_outers())
    return _GHENT_POLYGON_EXACT


def get_ghent_convex_hull_polygon() -> Polygon:
    """Load and cache the Ghent convex hull polygon."""
    global _GHENT_POLYGON_CONVEX_HULL
    if _GHENT_POLYGON_CONVEX_HULL is None:
        _GHENT_POLYGON_CONVEX_HULL = Polygon(get_ghent_convex_hull())
    return _GHENT_POLYGON_CONVEX_HULL


# ============================================================
# Raster block streaming
# ============================================================

def iter_raster_blocks(
    tif_path:      Path,
    value_dtype=   np.float32,
    skip_zeros:    bool = True,
    already_wgs84: bool = True,
    transformer:   Optional[Transformer] = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream a GeoTIFF one block at a time, yielding DataFrames with columns:
        longitude, latitude, value

    Memory usage is bounded to one raster block regardless of file size.

    Parameters
    ----------
    tif_path      : path to the GeoTIFF
    value_dtype   : numpy dtype to cast raster values to
    skip_zeros    : drop pixels where value == 0
    already_wgs84 : if True, pixel centres are used directly as lon/lat
    transformer   : pyproj.Transformer for reprojection (when not WGS-84)
    """
    with rasterio.open(tif_path) as src:
        nodata     = src.nodata
        block_iter = list(src.block_windows(1))
        if BLOCK_LIMIT is not None:
            block_iter = block_iter[:BLOCK_LIMIT]

        for _, window in block_iter:
            data      = src.read(1, window=window).astype(value_dtype)
            transform = src.window_transform(window)
            h, w      = data.shape

            row_idx, col_idx = np.mgrid[0:h, 0:w]
            xs, ys = rasterio.transform.xy(
                transform, row_idx.ravel(), col_idx.ravel(), offset="center"
            )
            xs   = np.asarray(xs,   dtype=np.float64)
            ys   = np.asarray(ys,   dtype=np.float64)
            vals = data.ravel()

            mask = ~np.isnan(vals)
            if nodata is not None:
                mask &= vals != nodata
            if skip_zeros:
                mask &= vals != 0.0

            if not mask.any():
                continue

            xs, ys, vals = xs[mask], ys[mask], vals[mask]

            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)   # → (lon, lat)

            yield pd.DataFrame({"longitude": xs, "latitude": ys, "value": vals})


def iter_raster_blocks_masked(
    tif_path:      Path,
    mask_shapes:   Polygon | list[Polygon],
    value_dtype=   np.float32,
    skip_zeros:    bool = True,
    already_wgs84: bool = True,
    transformer:   Optional[Transformer] = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream a GeoTIFF clipped to mask_shapes using rasterio.mask (GDAL-optimized).
    Yields DataFrames with columns: longitude, latitude, value.

    Much faster than iter_raster_blocks() + per-row polygon filtering because
    raster-polygon clipping happens at GDAL level, not via Shapely point checks.

    Automatically handles CRS mismatch: if mask_shapes are in a different CRS
    than the raster, transforms the shapes to the raster's CRS before masking.

    Parameters
    ----------
    tif_path      : path to the GeoTIFF
    mask_shapes   : Shapely Polygon or list of Polygons to clip to (assumed WGS-84)
    value_dtype   : numpy dtype to cast raster values to
    skip_zeros    : drop pixels where value == 0
    already_wgs84 : if True, pixel centres are used directly as lon/lat
    transformer   : pyproj.Transformer for reprojection (raster CRS → WGS-84)
    """
    if not isinstance(mask_shapes, list):
        mask_shapes = [mask_shapes]

    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        raster_crs = src.crs

        # If raster CRS differs from WGS-84, transform mask shapes to match raster CRS
        mask_shapes_for_clip = mask_shapes
        if raster_crs and str(raster_crs) != "EPSG:4326":
            log.debug(f"Raster {tif_path.name} CRS: {raster_crs} — transforming mask shapes from WGS-84")
            # Polygon is in WGS-84; transform to raster CRS for masking
            wgs84_to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            mask_shapes_for_clip = [
                shapely_transform(wgs84_to_raster.transform, poly)
                for poly in mask_shapes
            ]
        elif not raster_crs:
            log.warning(f"Raster {tif_path.name} has no CRS metadata — assuming WGS-84")

        # Use rasterio.mask to efficiently clip raster to polygon bounds at GDAL level.
        # Returns array and updated transform.
        try:
            masked_data, out_transform = rasterio.mask.mask(
                src, mask_shapes_for_clip, crop=True, nodata=nodata, indexes=1
            )
        except ValueError:
            # No intersection between raster and mask shapes
            log.warning(f"No intersection between {tif_path.name} and mask shapes")
            return

        data = masked_data.astype(value_dtype) if masked_data.ndim == 2 else masked_data[0].astype(value_dtype)
        h, w = data.shape

        row_idx, col_idx = np.mgrid[0:h, 0:w]
        xs, ys = rasterio.transform.xy(
            out_transform, row_idx.ravel(), col_idx.ravel(), offset="center"
        )
        xs   = np.asarray(xs,   dtype=np.float64)
        ys   = np.asarray(ys,   dtype=np.float64)
        vals = data.ravel()

        mask = ~np.isnan(vals)
        if nodata is not None:
            mask &= vals != nodata
        if skip_zeros:
            mask &= vals != 0.0

        if not mask.any():
            return

        xs, ys, vals = xs[mask], ys[mask], vals[mask]

        if transformer is not None:
            xs, ys = transformer.transform(xs, ys)   # → (lon, lat)

        yield pd.DataFrame({"longitude": xs, "latitude": ys, "value": vals})


# ============================================================
# Polygon filtering
# ============================================================

def filter_by_polygon(
    df:      pd.DataFrame,
    polygon: Polygon,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """
    Filter a DataFrame to keep only rows with (lon, lat) points inside the polygon.
    Uses AABB pre-filter before the expensive point-in-polygon test.
    """
    if len(df) == 0:
        return df

    bounds_minx, bounds_miny, bounds_maxx, bounds_maxy = polygon.bounds
    aabb_mask = (
        (df[lon_col] >= bounds_minx) & (df[lon_col] <= bounds_maxx) &
        (df[lat_col] >= bounds_miny) & (df[lat_col] <= bounds_maxy)
    )
    df_aabb = df[aabb_mask]

    if len(df_aabb) == 0:
        return df_aabb

    mask = np.array(
        [polygon.contains(Point(row[lon_col], row[lat_col]))
         for _, row in df_aabb.iterrows()],
        dtype=bool,
    )
    return df_aabb[mask]


# ============================================================
# H3 indexing
# ============================================================

def add_h3(df: pd.DataFrame, resolution: int = H3_RES) -> pd.DataFrame:
    """Add a tile_id (H3 cell string) column.  Vectorised via numpy + list-comp."""
    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()
    df["tile_id"] = [
        h3.latlng_to_cell(float(lat), float(lon), resolution)
        for lat, lon in zip(lats, lons)
    ]
    return df


# ============================================================
# Histogram helper
# ============================================================

def histogram(values: np.ndarray, edges: list[float]) -> dict:
    """Compute a fixed-bin histogram, ignoring NaN values."""
    clean = values[~np.isnan(values)]
    counts, _ = np.histogram(clean, bins=edges)
    return {"edges": edges, "counts": counts.tolist()}
