"""
features.py  -  /src/stream/features.py
========================================
Feature descriptors, registry, and the 2x2 spatiotemporal factory framework.

Public API:
  FeatureRow                                  - namedtuple view of one LST row for custom callables
  FeatureRegistry                             - container for all registered feature descriptors
  nearest()                                   - factory: nearest point in a dataset
  aggregate_in_radius()                       - factory: aggregated points within radius
  urban_atlas_luc_fraction()                  - factory: UA polygon fraction (single LUC code)
  urban_atlas_classifications_fractions()     - factory: UA polygon fractions (semantic classifications)
  wis_fraction()                              - factory: WIS polygon fraction

The 2x2 temporalxspatial framework
-----------------------------------
Spatial:
  nearest(dataset, columns, temporal, ...)
      -> the single closest point in space (optionally filtered by time)
  aggregate_in_radius(dataset, radius_m, columns, agg, temporal, ...)
      -> COUNT / AVG / SUM / MIN / MAX of all points within radius_m metres
  urban_atlas_luc_fraction(luc_code, radius_m, ua_year, ...)
      -> fraction [0, 1] of a circle's area occupied by polygons of luc_code
  urban_atlas_classifications_fractions(classification_map, radius_m, ua_year, ...)
      -> dict of fractions for each semantic classification

Temporal (applies to non-static datasets):
  "last_previous"   -> most recent observation with ts <= driving_ts
  "nearest"         -> observation with smallest |ts - driving_ts|
  "none"            -> no temporal filter (static datasets)
"""

from __future__ import annotations

import logging
import time
from collections import namedtuple
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from connections import Connections
from poly_raster import _PolyRaster
from queries import (
    batch_nearest, batch_radius,
    batch_urban_atlas_luc_fraction, batch_wis_fraction,
    query_nearest, query_radius,
    query_urban_atlas_luc_fraction, query_wis_fraction,
)

log = logging.getLogger("stream")


# ---------------------------------------------------------------------------
# FeatureRow
# ---------------------------------------------------------------------------
FeatureRow = namedtuple(
    "FeatureRow",
    ["longitude", "latitude", "aster_lst", "modis_lst", "ndvi",
     "image_id", "timestamp", "partition_key", "tile_id"],
)


# ---------------------------------------------------------------------------
# _FeatureDescriptor
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
    __slots__ = (
        "name", "prefix", "_batch_fn", "_row_fn",
        # Raster precomputation spec — set by urban_atlas_luc_fraction(),
        # urban_atlas_classifications_fractions(), and wis_fraction() factories;
        # absent on all other descriptors.
        "_raster_ref", "_raster_type",
        "_luc_code", "_ua_year",           # UA single-code fields
        "_classification_map",             # UA multi-code (classifications) field
        "_attr_col", "_attr_val",          # WIS fields
        "_radius_m",                       # shared
    )

    def __init__(
        self,
        name:     str,
        prefix:   str,
        batch_fn: Optional[Callable] = None,
        row_fn:   Optional[Callable] = None,
    ):
        self.name      = name
        self.prefix    = prefix
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
            log.debug("batch_feature: computing '%s' for %d rows", desc.name, len(df))
            t0 = time.perf_counter()
            try:
                result = desc.compute_batch(df, conns)
                log.info("batch_feature '%s': %.3fs -> cols %s",
                         desc.name, time.perf_counter() - t0, list(result.columns))
                parts.append(result)
            except Exception as exc:
                log.error("batch_feature '%s': FAILED in %.3fs - %s",
                          desc.name, time.perf_counter() - t0, exc, exc_info=True)
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
# Feature factories — the 2x2 framework
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
    temporal   : "none"          - ignore timestamps (static datasets)
                 "last_previous" - most recent observation with ts <= driving_ts
                 "nearest"       - observation closest in time to driving_ts
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


def urban_atlas_luc_fraction(
    luc_code: str,
    radius_m: float,
    ua_year: Optional[int] = None,
    prefix: str = "",
) -> _FeatureDescriptor:
    """
    Factory: fraction of a circle's area covered by Urban Atlas polygons of
    *luc_code*.

    Parameters
    ----------
    luc_code  : Urban Atlas land-use classification code, e.g. "11100"
                (continuous urban fabric), "14100" (green urban areas).
    radius_m  : radius of the query circle in metres.
    ua_year   : restrict to a specific survey year (2006 / 2012 / 2018 / 2021).
                Default None = last_previous survey year relative to each LST
                row's timestamp, which is the correct temporal join for most
                use cases.
    prefix    : optional column name prefix in the output.

    Output column
    -------------
    ``{prefix}luc{luc_code}_{radius_m}m_frac``  - float in [0.0, 1.0]

    Performance
    -----------
    When StreamConfig is initialised with raster_resolution_m > 0 (default
    15 m), a _PolyRaster is precomputed at stream() start covering the exact
    bounding box of the selected LST partitions.  All batch lookups then cost
    O(1) per row (numpy array index).  The slow Shapely path is the fallback
    if no raster is available.
    """
    out_col  = f"luc{luc_code}_{int(radius_m)}m_frac"
    _prefix  = prefix or "ua_"

    # _raster is set to a _PolyRaster instance by StreamConfig.stream() before
    # the first batch runs.  The mutable list acts as a closure cell so the
    # batch_fn always sees the current value.
    _raster_ref: List[Optional[_PolyRaster]] = [None]

    def _batch(df: pd.DataFrame, conns: Connections) -> pd.DataFrame:
        return batch_urban_atlas_luc_fraction(
            df, conns, luc_code, radius_m, ua_year, out_col,
            raster=_raster_ref[0],
        )

    def _row(row: FeatureRow, conns: Connections) -> dict:
        frac = query_urban_atlas_luc_fraction(
            conns, row.longitude, row.latitude,
            radius_m=radius_m, luc_code=luc_code, ua_year=ua_year,
        )
        return {out_col: frac}

    desc = _FeatureDescriptor(
        name=f"ua_luc_{luc_code}_{int(radius_m)}m",
        prefix=_prefix,
        batch_fn=_batch,
        row_fn=_row,
    )
    desc._raster_ref   = _raster_ref
    desc._raster_type  = "ua"
    desc._luc_code     = luc_code
    desc._radius_m     = radius_m
    desc._ua_year      = ua_year
    return desc


def urban_atlas_classifications_fractions(
    classification_map: Dict[str, List[str]],
    radius_m: float,
    ua_year: Optional[int] = None,
    prefix: str = "",
) -> _FeatureDescriptor:
    """
    Factory: aggregate Urban Atlas fractions by semantic classification.
    
    Groups multiple LUC codes into meaningful categories and aggregates
    their fractions (area coverage) into a single feature per classification.
    
    Parameters
    ----------
    classification_map : Dict[str, List[str]]
        {classification_name: [luc_code1, luc_code2, ...]}
        Example:
            {
                "artificial": ["11100", "11210", "11220"],
                "vegetation": ["31000", "32000"],
                "water": ["50000"],
            }
    radius_m : float
        Query radius in metres.
    ua_year : int, optional
        Restrict to Urban Atlas survey year (default: None = last_previous)
    prefix : str
        Column name prefix in the output (default: "ua_")
        
    Output columns
    ---------------
    {prefix}{classification}_{radius_m}m_frac for each classification
    Each column is a float in [0.0, 1.0] representing the fraction of the
    circle's area covered by that classification.

    Performance
    -----------
    When StreamConfig is initialised with raster_resolution_m > 0 (default
    15 m), a _PolyRaster is precomputed at stream() start for every LUC code
    in the classification_map.  All batch lookups then cost O(1) per row
    (numpy array index) and require no SpatiaLite round-trips.  The slow
    Shapely path is the fallback if no raster is available.
    """
    if not isinstance(classification_map, dict):
        raise TypeError("classification_map must be a dict")
    
    # Normalise all LUC codes to strings once at factory time
    norm_map: Dict[str, List[str]] = {
        cls: [str(c) for c in codes]
        for cls, codes in classification_map.items()
    }

    # Validate: each LUC code appears in at most one classification
    reverse_map: Dict[str, str] = {}
    for classification, luc_codes in norm_map.items():
        for luc_code_str in luc_codes:
            if luc_code_str in reverse_map:
                raise ValueError(
                    f"LUC code '{luc_code_str}' appears in multiple classifications"
                )
            reverse_map[luc_code_str] = classification
    
    _prefix = prefix or "ua_"

    # _raster_ref is a mutable cell wired by _build_poly_rasters before the
    # first batch.  When populated, batch lookups cost O(1) per row.
    _raster_ref: List[Optional[_PolyRaster]] = [None]

    def _batch(df: pd.DataFrame, conns: Connections) -> pd.DataFrame:
        raster = _raster_ref[0]
        result_cols: Dict[str, any] = {}

        lons = df["longitude"].to_numpy(dtype=float)
        lats = df["latitude"].to_numpy(dtype=float)
        tss  = df["timestamp"].to_numpy(dtype=str)
        n    = len(df)

        for classification, luc_codes in norm_map.items():
            total_frac = np.zeros(n, dtype=np.float32)

            for luc_code in luc_codes:
                if raster is not None:
                    # Fast path: O(1) raster lookup per row
                    if ua_year is not None:
                        layer_key = f"{luc_code}:{ua_year}"
                        code_frac = np.array(
                            [raster.lookup(lons[i], lats[i], layer_key) or 0.0
                             for i in range(n)],
                            dtype=np.float32,
                        )
                    else:
                        code_frac = np.array(
                            [raster.lookup_ua_last_previous(
                                 lons[i], lats[i], luc_code, int(tss[i][:4])) or 0.0
                             for i in range(n)],
                            dtype=np.float32,
                        )
                else:
                    # Slow path: per-tile SpatiaLite + Shapely queries
                    frac_col = f"luc{luc_code}_{int(radius_m)}m_frac"
                    batch = batch_urban_atlas_luc_fraction(
                        df, conns, luc_code, radius_m, ua_year,
                        frac_col, raster=None,
                    )
                    code_frac = (
                        batch[frac_col].to_numpy(dtype=np.float32)
                        if frac_col in batch.columns
                        else np.zeros(n, dtype=np.float32)
                    )

                total_frac += code_frac

            # Cap aggregate at 1.0 (polygons of different codes may overlap)
            total_frac = np.minimum(total_frac, 1.0)
            col_name = f"{_prefix}{classification}_{int(radius_m)}m_frac"
            result_cols[col_name] = total_frac

        return pd.DataFrame(result_cols, index=df.index)
    
    def _row(row: FeatureRow, conns: Connections) -> dict:
        result = {}
        
        for classification, luc_codes in norm_map.items():
            total_frac = 0.0
            
            for luc_code in luc_codes:
                frac = query_urban_atlas_luc_fraction(
                    conns, row.longitude, row.latitude,
                    radius_m=radius_m, luc_code=luc_code, ua_year=ua_year,
                )
                if frac is not None:
                    total_frac += frac
            
            # Cap at 1.0
            total_frac = min(total_frac, 1.0)
            col_name = f"{_prefix}{classification}_{int(radius_m)}m_frac"
            result[col_name] = total_frac
        
        return result
    
    desc = _FeatureDescriptor(
        name=f"ua_classifications_{int(radius_m)}m",
        prefix=_prefix,
        batch_fn=_batch,
        row_fn=_row,
    )
    desc._raster_ref        = _raster_ref
    desc._raster_type       = "ua_classifications"
    desc._classification_map = norm_map
    desc._radius_m          = radius_m
    desc._ua_year           = ua_year
    return desc


def wis_fraction(
    attr_col: str,
    attr_val: str,
    radius_m: float,
    prefix: str = "",
) -> _FeatureDescriptor:
    """
    Factory: fraction of a circle's area covered by WIS polygons whose
    *attr_col* equals *attr_val*.

    WIS (Ghent Road Information System) stores road surface polygons with two
    categorical attributes:
        bestemming     TEXT   road surface purpose (primary split dimension)
        materiaalsoort TEXT   surface material (may be null)

    Parameters
    ----------
    attr_col : column to filter on — "bestemming" or "materiaalsoort".
    attr_val : the attribute value to match.
    radius_m : radius of the query circle in metres.
    prefix   : optional column name prefix in the output.

    Output column
    -------------
    ``{prefix}wis_{attr_val}_{radius_m}m_frac``  - float in [0.0, 1.0]

    Temporal behaviour
    ------------------
    WIS has a single survey timestamp (static dataset).  No temporal argument
    is needed; every LST row receives the same underlying polygon layer.

    Performance
    -----------
    Same raster-precomputation path as urban_atlas_luc_fraction.  When a
    _PolyRaster is available, lookups are O(1).
    """
    safe_val = attr_val.replace(" ", "_").replace("/", "_")
    out_col  = f"wis_{safe_val}_{int(radius_m)}m_frac"
    _prefix  = prefix or "wis_"

    _raster_ref: List[Optional[_PolyRaster]] = [None]

    def _batch(df: pd.DataFrame, conns: Connections) -> pd.DataFrame:
        return batch_wis_fraction(
            df, conns, attr_col, attr_val, radius_m, out_col,
            raster=_raster_ref[0],
        )

    def _row(row: FeatureRow, conns: Connections) -> dict:
        frac = query_wis_fraction(
            conns, row.longitude, row.latitude,
            radius_m=radius_m, attr_col=attr_col, attr_val=attr_val,
        )
        return {out_col: frac}

    desc = _FeatureDescriptor(
        name=f"wis_{attr_col}_{safe_val}_{int(radius_m)}m",
        prefix=_prefix,
        batch_fn=_batch,
        row_fn=_row,
    )
    desc._raster_ref   = _raster_ref
    desc._raster_type  = "wis"
    desc._attr_col     = attr_col
    desc._attr_val     = attr_val
    desc._radius_m     = radius_m
    return desc