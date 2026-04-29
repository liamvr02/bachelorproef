"""
queries.py  -  /src/stream/queries.py
======================================
Row-level and batch spatial/temporal query helpers.

Public API (usable in custom feature callables):
  query_nearest               - single nearest point, one row
  query_radius                - aggregated radius query, one row
  query_urban_atlas_luc_fraction - UA polygon fraction, one row
  query_wis_fraction          - WIS polygon fraction, one row

Internal batch helpers (used by feature factories):
  batch_nearest               - vectorised nearest, one DataFrame
  batch_radius                - vectorised radius aggregate, one DataFrame
  batch_urban_atlas_luc_fraction - vectorised UA fraction, one DataFrame
  batch_wis_fraction          - vectorised WIS fraction, one DataFrame
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from stream.connections import Connections
from stream.geo import (
    _LAT_DEG_PER_M, _LON_DEG_PER_M,
    _haversine_m, _spatialite_fetch_bbox,
    _temporal_clause, _temporal_order, _ts_epoch,
)
from stream.poly_raster import (
    _PolyRaster,
    _ua_fetch_candidates, _wis_fetch_candidates,
    _ua_make_circle, _ua_compute_fraction,
)

log = logging.getLogger("stream")


# ---------------------------------------------------------------------------
# Extra WHERE-fragment helper (attribute filters + planting-year temporal)
# ---------------------------------------------------------------------------

def _extra_where_fragment(
    attr_filter: Optional[Dict[str, str]],
    aanlegjaar_lte_scene: bool,
    scene_ts: Optional[str],
) -> str:
    """
    Build extra SQL WHERE fragments to be appended to existing radius queries.

    *attr_filter* applies equality predicates: ``{"beheerfase": "Jeugdfase"}``
    becomes ``AND beheerfase = 'Jeugdfase'``.  Values come from registry-time
    constants (not user input) but are still single-quote-escaped.

    *aanlegjaar_lte_scene* — when True, restricts trees rows to those already
    planted at or before the scene's year *and* with a known ``aanlegjaar``
    (i.e. excludes the ~73% of trees with NULL aanlegjaar, which would
    otherwise contaminate the staggered-DiD treatment dose).
    """
    parts: List[str] = []
    if attr_filter:
        for col, val in attr_filter.items():
            safe = str(val).replace("'", "''")
            parts.append(f"AND {col} = '{safe}'")
    if aanlegjaar_lte_scene:
        if scene_ts is None:
            raise ValueError("aanlegjaar_lte_scene requires scene_ts")
        scene_year = int(str(scene_ts)[:4])
        parts.append(
            f"AND aanlegjaar IS NOT NULL AND aanlegjaar <= {scene_year}"
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# LST Emissivity helpers
# ---------------------------------------------------------------------------

def _resolve_lst_columns(columns: List[str], emissivity_mode: str) -> List[str]:
    """
    Resolve LST column selection based on emissivity_mode.

    For dataset_id="lst", the table has 3 value columns: aster_lst, modis_lst, ndvi.
    This function maps the requested columns (usually ["temperature"] for backward
    compatibility) to the appropriate emissivity columns.

    Parameters
    ----------
    columns : list of column names requested (may include "temperature" for compat)
    emissivity_mode : "any", "fallback", "aster", "modis", "ndvi"

    Returns
    -------
    Updated column list with "temperature" replaced by emissivity column(s).
    """
    # Map backward-compatible "temperature" to emissivity columns
    result = []
    for col in columns:
        if col == "temperature":
            if emissivity_mode == "any":
                # Return all three so caller can pick first non-null
                result.extend(["aster_lst", "modis_lst", "ndvi"])
            elif emissivity_mode == "fallback":
                # ASTER > MODIS (exclude NDVI)
                result.extend(["aster_lst", "modis_lst"])
            elif emissivity_mode in ("aster", "modis", "ndvi"):
                result.append(f"{emissivity_mode}_lst" if emissivity_mode != "ndvi" else "ndvi")
            else:
                raise ValueError(f"Invalid emissivity_mode: {emissivity_mode}")
        else:
            result.append(col)
    return result


def _select_temperature_from_emissivity(
    result: dict,
    emissivity_mode: str,
) -> dict:
    """
    Post-process LST query result to select first non-null temperature value
    based on emissivity_mode.

    Modifies result dict in-place, replacing multi-column output with single
    "temperature" key.

    Parameters
    ----------
    result : dict from query_nearest containing aster_lst, modis_lst, and/or ndvi
    emissivity_mode : "any" or "fallback"

    Returns
    -------
    Modified result dict with "temperature" key containing first non-null value.
    """
    candidates = []
    if emissivity_mode == "any":
        candidates = [("aster_lst", "ASTER"), ("modis_lst", "MODIS"), ("ndvi", "NDVI")]
    elif emissivity_mode == "fallback":
        candidates = [("aster_lst", "ASTER"), ("modis_lst", "MODIS")]

    selected_temp = None
    for col_name, _ in candidates:
        if col_name in result and result[col_name] is not None:
            selected_temp = result[col_name]
            break

    # Remove individual emissivity columns and add unified "temperature" key
    for col_name, _ in candidates:
        result.pop(col_name, None)

    result["temperature"] = selected_temp
    return result


# ---------------------------------------------------------------------------
# Row-level helpers (public - usable in custom feature callables)
# ---------------------------------------------------------------------------

def query_nearest(
    conn: Connections,
    dataset_id: str,
    lon: float,
    lat: float,
    timestamp: str,
    columns: List[str],
    temporal: str = "none",
    radius_m: float = 500.0,
    emissivity_mode: str = "any",
) -> dict:
    """
    Find the single spatially nearest point in *dataset_id*, optionally
    filtered/ordered by time.

    Returns a dict of {column: value} for the nearest row, or {} if none found.

    This is the low-level building block used by the nearest() factory and
    available to custom feature callables.

    Parameters
    ----------
    conn           : Connections object to database(s)
    dataset_id     : Name of dataset to query (e.g., "lst", "dhm", "ndvi")
    lon, lat       : Coordinates for spatial query
    timestamp      : ISO timestamp for temporal filtering
    columns        : List of column names to retrieve
    temporal       : Temporal filtering mode ("none", "last_previous", "nearest")
    radius_m       : Search radius in metres
    emissivity_mode: For dataset_id="lst" only. How to select LST value:
                     "any"      -> first non-null in order: ASTER > MODIS > NDVI
                     "fallback" -> ASTER > MODIS (NDVI never used as substitute)
                     "aster"/"modis"/"ndvi" -> use only that column (may be null)
                     (Ignored for non-LST datasets)
    """
    meta    = conn._meta[dataset_id]
    store   = meta["store"]
    table   = meta["db_table"]
    db_file = meta["db_file"]
    ts_col  = meta.get("timestamp_column", "timestamp")
    has_ts  = meta["temporal_behavior"] != "static"

    # ========== Handle LST emissivity column selection ==========
    if dataset_id == "lst":
        columns = _resolve_lst_columns(columns, emissivity_mode)
    else:
        emissivity_mode = "any"  # No emissivity handling for non-LST datasets

    cols_sql = ", ".join(columns)
    t_where  = _temporal_clause(temporal, ts_col, timestamp) if has_ts else ""
    t_order  = _temporal_order(temporal, ts_col, timestamp)  if has_ts else "1"

    if store == "spatialite":
        db = conn.spatialite(db_file)
        dlat = radius_m * _LAT_DEG_PER_M
        dlon = radius_m * _LON_DEG_PER_M

        ts_col_select = f", {ts_col}" if has_ts else ""
        ts_where = f"AND {ts_col} <= '{timestamp}'" if (has_ts and temporal == "last_previous") else ""
        extra_cols = "".join(f", {c}" for c in columns)

        rows = _spatialite_fetch_bbox(
            db, table, lon, lat, dlat, dlon,
            extra_cols=f"{ts_col_select}{extra_cols}",
            extra_where=ts_where,
        )
        if not rows:
            return {}

        # Row layout: (id=0, _lon=1, _lat=2, [ts_col=3 if has_ts,] *columns)
        lons_arr = np.array([r[1] for r in rows], dtype=float)
        lats_arr = np.array([r[2] for r in rows], dtype=float)
        dists    = _haversine_m(lon, lat, lons_arr, lats_arr)

        mask = dists <= radius_m
        if not mask.any():
            return {}

        col_offset = 3 + (1 if has_ts else 0)  # skip id, _lon, _lat, [ts]

        if has_ts and temporal == "nearest":
            driving_epoch = _ts_epoch(timestamp)
            td = np.array([abs(_ts_epoch(r[3]) - driving_epoch) for r in rows], dtype=float)
            td[~mask] = np.inf
            dists_m = dists.copy(); dists_m[~mask] = np.inf
            best_i = int(np.lexsort((dists_m, td))[0])
        else:
            dists_m = dists.copy(); dists_m[~mask] = np.inf
            best_i = int(np.argmin(dists_m))

        best_row = rows[best_i]
        result = {columns[j]: best_row[col_offset + j] for j in range(len(columns))}

        # ========== Post-process for LST emissivity modes ==========
        if dataset_id == "lst" and emissivity_mode in ("any", "fallback"):
            result = _select_temperature_from_emissivity(result, emissivity_mode)

        return result

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
        result = {desc[i][0]: row[i] for i in range(len(desc))}

        # ========== Post-process for LST emissivity modes ==========
        if dataset_id == "lst" and emissivity_mode in ("any", "fallback"):
            result = _select_temperature_from_emissivity(result, emissivity_mode)

        return result


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
    attr_filter: Optional[Dict[str, str]] = None,
    aanlegjaar_lte_scene: bool = False,
) -> dict:
    """
    Aggregate all points within *radius_m* metres of (lon, lat).

    *agg* is one of: "count", "avg", "sum", "min", "max".
    Returns {f"{col}_{agg}": value, "count": n} or {"count": 0} if empty.

    *attr_filter* — optional equality predicates appended to the WHERE clause,
    e.g. ``{"beheerfase": "Jeugdfase"}`` to count only juvenile trees.

    *aanlegjaar_lte_scene* — when True (trees-only), restrict to rows with
    ``aanlegjaar IS NOT NULL AND aanlegjaar <= year(timestamp)``.  This is
    the staggered-DiD treatment-dose filter: count trees that were already
    planted at the time of the LST observation and have a known planting year.

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
    extra_where = _extra_where_fragment(
        attr_filter, aanlegjaar_lte_scene, timestamp,
    )
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

        ts_col_select  = f", {ts_col}" if has_ts else ""
        ts_where = f"AND {ts_col} <= '{timestamp}'" if (has_ts and temporal == "last_previous") else ""
        col_select_str = "".join(f", {c}" for c in columns) if columns else ""

        rows = _spatialite_fetch_bbox(
            db, table, lon, lat, dlat, dlon,
            extra_cols=f"{ts_col_select}{col_select_str}",
            extra_where=f"{ts_where} {extra_where}".strip(),
        )
        if not rows:
            return {"count": 0}

        lons_arr = np.array([r[1] for r in rows], dtype=float)
        lats_arr = np.array([r[2] for r in rows], dtype=float)
        dists    = _haversine_m(lon, lat, lons_arr, lats_arr)
        mask     = dists <= radius_m
        n_in     = int(mask.sum())
        if n_in == 0:
            return {"count": 0}

        result: dict = {"count": n_in}
        col_offset = 3 + (1 if has_ts else 0)  # id, _lon, _lat, [ts,] *columns
        for j, col in enumerate(columns):
            if agg_up == "COUNT":
                break
            vals = np.array([r[col_offset + j] for r in rows], dtype=float)[mask]
            if   agg_up == "AVG": result[f"{col}_{agg}"] = float(vals.mean())
            elif agg_up == "SUM": result[f"{col}_{agg}"] = float(vals.sum())
            elif agg_up == "MIN": result[f"{col}_{agg}"] = float(vals.min())
            elif agg_up == "MAX": result[f"{col}_{agg}"] = float(vals.max())
        return result

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
            {extra_where}
        """
        row = db.execute(sql).fetchone()
        if row is None:
            return {"count": 0}
        desc = db.description
        return {desc[i][0]: row[i] for i in range(len(desc))}


def query_urban_atlas_luc_fraction(
    conn: Connections,
    lon: float,
    lat: float,
    radius_m: float,
    luc_code: str,
    ua_year: Optional[int] = None,
) -> float:
    """
    Return the fraction [0.0, 1.0] of a circle's area occupied by Urban Atlas
    polygons whose luc_code matches *luc_code*.

    The circle is centred at (lon, lat) with radius *radius_m* metres.
    If *ua_year* is given, only polygons from that survey year are considered.

    Algorithm
    ---------
    1. SpatiaLite R-tree pre-filter - retrieve candidate polygon WKB blobs.
       No GEOS functions called inside SQLite at this step.
    2. Shapely (independent GEOS context) - intersect each candidate with
       a degree-space circle and accumulate area.
    3. Return sum(intersection areas) / circle area.

    This is the low-level building block used by urban_atlas_luc_fraction() and
    available to custom feature callables.
    """
    db     = conn.spatialite("spatial.db")
    blobs  = _ua_fetch_candidates(db, lon, lat, radius_m, luc_code, ua_year)
    if not blobs:
        return 0.0
    circle = _ua_make_circle(lon, lat, radius_m)
    return _ua_compute_fraction(blobs, circle)


def query_wis_fraction(
    conn: Connections,
    lon: float,
    lat: float,
    radius_m: float,
    attr_col: str,
    attr_val: str,
) -> float:
    """
    Return the fraction [0.0, 1.0] of a circle's area covered by WIS polygons
    whose *attr_col* equals *attr_val*.

    Identical algorithm to query_urban_atlas_luc_fraction - R-tree pre-filter
    in SpatiaLite then exact Shapely intersection - but targets the wis table.
    WIS is a single-timestamp (static) dataset; there is no temporal argument.

    This is the row-level building block available to custom feature callables.
    """
    db     = conn.spatialite("spatial.db")
    blobs  = _wis_fetch_candidates(db, lon, lat, radius_m, attr_col, attr_val)
    if not blobs:
        return 0.0
    circle = _ua_make_circle(lon, lat, radius_m)   # same degree-space ellipse
    return _ua_compute_fraction(blobs, circle)


# ---------------------------------------------------------------------------
# Batch helpers (one SQL round-trip per feature per batch)
# ---------------------------------------------------------------------------

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

    SpatiaLite path:
        Loads all input coordinates into a temporary table, then issues one
        JOIN against the R-tree spatial index.  All candidates are returned
        and the winner per row_idx is picked in Python (minimum distance,
        with temporal tiebreak when requested).  One SQL round-trip per batch.

    DuckDB path:
        Builds a CTE VALUES list and uses ROW_NUMBER() OVER (PARTITION BY
        row_idx ORDER BY distance) to pick the winner in one query.

    Returns a DataFrame indexed like df with one column per requested feature
    column (None where no point was found within radius_m).
    """
    meta    = conns._meta[dataset_id]
    store   = meta["store"]
    table   = meta["db_table"]
    db_file = meta["db_file"]
    has_ts  = meta["temporal_behavior"] != "static"

    dlat = radius_m * _LAT_DEG_PER_M
    dlon = radius_m * _LON_DEG_PER_M

    if store == "spatialite":
        db = conns.spatialite(db_file)

        if has_ts and temporal == "last_previous":
            ts_where_tpl = "AND {ts_col} <= '{{ts}}'"
        else:
            ts_where_tpl = ""

        ts_col_select  = f", {meta.get('timestamp_column', 'timestamp')}" if has_ts else ""
        col_select_str = "".join(f", {c}" for c in columns)

        lons_in     = df["longitude"].to_numpy(dtype=float)
        lats_in     = df["latitude"].to_numpy(dtype=float)
        tss_in      = df["timestamp"].to_numpy(dtype=str)
        tile_ids_in = df["tile_id"].to_numpy(dtype=str)

        # One bbox query per unique (tile_id, date) - same cache key logic as
        # batch_radius.  All rows in a tile share the same candidate set.
        cache: Dict[str, Optional[tuple]] = {}
        col_arrays = {c: [None] * len(df) for c in columns}

        n_queries   = 0
        t_query_sum = 0.0

        for i in range(len(df)):
            date_key  = tss_in[i][:10] if has_ts else "static"
            cache_key = f"{tile_ids_in[i]}:{date_key}"

            if cache_key not in cache:
                lon_i = lons_in[i]
                lat_i = lats_in[i]
                ts_w  = ts_where_tpl.format(ts_col=meta.get('timestamp_column','timestamp'),
                                            ts=tss_in[i]) if ts_where_tpl else ""

                t_q = time.perf_counter()
                rows = _spatialite_fetch_bbox(
                    db, table, lon_i, lat_i, dlat, dlon,
                    extra_cols=f"{ts_col_select}{col_select_str}",
                    extra_where=ts_w,
                )
                t_q = time.perf_counter() - t_q
                n_queries   += 1
                t_query_sum += t_q

                # After first query, log its cost so a slow environment is
                # visible immediately rather than after the full batch.
                if n_queries == 1:
                    log.debug("batch_nearest[%s]: first tile query -> %d candidates "
                              "in %.3fs", dataset_id, len(rows), t_q)
                # Periodic summary every 100 unique tiles
                elif n_queries % 100 == 0:
                    log.debug("batch_nearest[%s]: %d unique tiles queried, "
                              "avg %.3fs/tile, elapsed %.1fs",
                              dataset_id, n_queries,
                              t_query_sum / n_queries, t_query_sum)

                if not rows:
                    cache[cache_key] = None
                else:
                    cand_lons = np.array([r[1] for r in rows], dtype=float)
                    cand_lats = np.array([r[2] for r in rows], dtype=float)
                    dists     = _haversine_m(lon_i, lat_i, cand_lons, cand_lats)
                    mask      = dists <= radius_m
                    if not mask.any():
                        cache[cache_key] = None
                    else:
                        ts_offset  = 3
                        col_offset = ts_offset + (1 if has_ts else 0)

                        if has_ts and temporal == "nearest":
                            driving_epoch = _ts_epoch(tss_in[i])
                            td = np.array([abs(_ts_epoch(r[ts_offset]) - driving_epoch)
                                           for r in rows], dtype=float)
                            td[~mask] = np.inf
                            dm = dists.copy(); dm[~mask] = np.inf
                            best_i = int(np.lexsort((dm, td))[0])
                        else:
                            dm = dists.copy(); dm[~mask] = np.inf
                            best_i = int(np.argmin(dm))

                        best_row = rows[best_i]
                        cache[cache_key] = tuple(best_row[col_offset + j]
                                                 for j in range(len(columns)))

            best = cache[cache_key]
            if best is not None:
                for j, c in enumerate(columns):
                    col_arrays[c][i] = best[j]

        log.debug("batch_nearest[%s]: done - %d unique tiles, %.3fs total, avg %.3fs/tile",
                  dataset_id, n_queries,
                  t_query_sum, t_query_sum / max(n_queries, 1))

        return pd.DataFrame(col_arrays, index=df.index)

    else:  # duckdb - tile-deduplicated: one small bounded query per unique tile.
        # The previous approach built a single VALUES CTE with all N rows and
        # ran ROW_NUMBER() OVER (PARTITION BY row_idx) across the full table.
        # DuckDB has no R-tree; the BETWEEN bbox filters use only zone-map
        # pruning, producing a huge intermediate set and 300-400 s per 10k batch.
        #
        # All pixels in the same tile_id are within the same ~174 m H3 cell, so
        # one representative-coordinate query per unique tile is correct.
        db = conns.duckdb(db_file)

        lons_in     = df["longitude"].to_numpy(dtype=float)
        lats_in     = df["latitude"].to_numpy(dtype=float)
        tss_in      = df["timestamp"].to_numpy(dtype=str)
        tile_ids_in = df["tile_id"].to_numpy(dtype=str)

        if has_ts and temporal == "last_previous":
            t_where_tpl = "AND timestamp <= '{ts}'"
            t_order_tpl = "timestamp DESC,"
        elif has_ts and temporal == "nearest":
            t_where_tpl = ""
            t_order_tpl = ("ABS(epoch(CAST(timestamp AS TIMESTAMP))"
                           " - epoch(CAST('{ts}' AS TIMESTAMP))),")
        else:
            t_where_tpl = ""
            t_order_tpl = ""

        cache: Dict[str, Optional[tuple]] = {}
        col_arrays = {c: [None] * len(df) for c in columns}

        seen_tile: Dict[str, int] = {}
        for i, tid in enumerate(tile_ids_in):
            if tid not in seen_tile:
                seen_tile[tid] = i

        n_queries = 0
        t_query_sum = 0.0
        for tile_id, i in seen_tile.items():
            date_key  = tss_in[i][:10] if has_ts else "static"
            cache_key = f"{tile_id}:{date_key}"
            if cache_key in cache:
                continue
            t_where = t_where_tpl.format(ts=tss_in[i]) if t_where_tpl else ""
            t_order = t_order_tpl.format(ts=tss_in[i]) if t_order_tpl else ""
            sql = f"""
                SELECT {", ".join(columns)},
                       sqrt(pow((longitude - {lons_in[i]}) * 111320, 2)
                          + pow((latitude  - {lats_in[i]}) * 111320
                              * cos(radians({lats_in[i]})), 2)) AS _dist
                FROM {table}
                WHERE longitude BETWEEN {lons_in[i] - dlon} AND {lons_in[i] + dlon}
                  AND latitude  BETWEEN {lats_in[i] - dlat} AND {lats_in[i] + dlat}
                  AND sqrt(pow((longitude - {lons_in[i]}) * 111320, 2)
                         + pow((latitude  - {lats_in[i]}) * 111320
                             * cos(radians({lats_in[i]})), 2)) <= {radius_m}
                {t_where}
                ORDER BY {t_order} _dist ASC
                LIMIT 1
            """
            t_q = time.perf_counter()
            row = db.execute(sql).fetchone()
            t_query_sum += time.perf_counter() - t_q
            n_queries   += 1
            cache[cache_key] = tuple(row[:len(columns)]) if row else None

        log.debug("batch_nearest[%s]: done - %d unique tiles, %.3fs total, avg %.3fs/tile",
                  dataset_id, n_queries, t_query_sum, t_query_sum / max(n_queries, 1))

        for i in range(len(df)):
            date_key  = tss_in[i][:10] if has_ts else "static"
            cache_key = f"{tile_ids_in[i]}:{date_key}"
            best = cache.get(cache_key)
            if best is not None:
                for j, c in enumerate(columns):
                    col_arrays[c][i] = best[j]

        return pd.DataFrame(col_arrays, index=df.index)


def batch_radius(
    df: pd.DataFrame,
    conns: Connections,
    dataset_id: str,
    columns: List[str],
    agg: str,
    temporal: str,
    radius_m: float,
    attr_filter: Optional[Dict[str, str]] = None,
    aanlegjaar_lte_scene: bool = False,
) -> pd.DataFrame:
    """
    Aggregate all points within *radius_m* metres for every row in *df*.

    The spatial filter is a degree-space bounding box centred on the exact
    pixel coordinate, followed by an exact Distance() check.  It is not
    restricted to any tile boundary, so feature points from neighbouring tiles
    are always included when they fall inside the search radius.

    One R-tree query is issued per unique (tile_id, date) pair.  This is
    correct and not just an optimisation: all pixels that share a tile_id
    are within the same ~200 m H3 cell, so the set of feature points captured
    by a circle of radius_m drawn from any pixel in the tile is the same for
    practical search radii (>= ~30 m).  The query always uses the exact pixel
    coordinates for the bounding box, not a tile centroid, so edge pixels that
    happen to share a tile with interior pixels use their own coordinates.

    Returns a DataFrame indexed like df with columns {col}_{agg} and "count".
    """
    meta    = conns._meta[dataset_id]
    store   = meta["store"]
    table   = meta["db_table"]
    db_file = meta["db_file"]
    has_ts  = meta["temporal_behavior"] != "static"

    agg_up  = agg.upper()
    agg_sql = ", ".join(
        f"{agg_up}(CAST(f.{c} AS REAL)) AS {c}_{agg}"
        for c in columns if agg_up != "COUNT"
    )
    out_cols   = ["count"] + [f"{c}_{agg}" for c in columns if agg_up != "COUNT"]
    select_sql = "COUNT(*) AS count" + (f", {agg_sql}" if agg_sql else "")

    dlat = radius_m * _LAT_DEG_PER_M
    dlon = radius_m * _LON_DEG_PER_M

    if has_ts and temporal in ("last_previous", "nearest"):
        t_where_tpl = "AND timestamp <= '{ts}'"
    else:
        t_where_tpl = ""

    lons     = df["longitude"].to_numpy(dtype=float)
    lats     = df["latitude"].to_numpy(dtype=float)
    tss      = df["timestamp"].to_numpy(dtype=str)
    tile_ids = df["tile_id"].to_numpy(dtype=str)

    cache: Dict[str, dict] = {}
    if aanlegjaar_lte_scene:
        # Result depends on the scene's calendar year — cache by (tile, year).
        cache_keys = np.array([
            f"{tid}:ay{ts[:4]}"
            for tid, ts in zip(tile_ids, tss)
        ])
    else:
        cache_keys = np.array([
            f"{tid}:{ts[:10] if has_ts else 'static'}"
            for tid, ts in zip(tile_ids, tss)
        ])

    for i, cache_key in enumerate(cache_keys):
        if cache_key in cache:
            continue
        lon = lons[i]
        lat = lats[i]
        ts  = tss[i]
        t_where = t_where_tpl.format(ts=ts) if t_where_tpl else ""
        extra_where = _extra_where_fragment(
            attr_filter, aanlegjaar_lte_scene, ts,
        )

        if store == "spatialite":
            db  = conns.spatialite(db_file)
            ts_col_name    = meta.get("timestamp_column", "timestamp")
            ts_col_select  = f", {ts_col_name}" if has_ts else ""
            col_select_str = "".join(f", {c}" for c in columns) if columns else ""

            rows = _spatialite_fetch_bbox(
                db, table, lon, lat, dlat, dlon,
                extra_cols=f"{ts_col_select}{col_select_str}",
                extra_where=f"{t_where} {extra_where}".strip(),
            )
            if not rows:
                cache[cache_key] = {"count": 0}
                continue

            cand_lons = np.array([r[1] for r in rows], dtype=float)
            cand_lats = np.array([r[2] for r in rows], dtype=float)
            dists     = _haversine_m(lon, lat, cand_lons, cand_lats)
            mask      = dists <= radius_m
            n_in      = int(mask.sum())
            if n_in == 0:
                cache[cache_key] = {"count": 0}
                continue

            row_result: dict = {"count": n_in}
            col_offset = 3 + (1 if has_ts else 0)
            for j, col in enumerate(columns):
                if agg_up == "COUNT":
                    break
                vals = np.array([r[col_offset + j] for r in rows], dtype=float)[mask]
                if   agg_up == "AVG": row_result[f"{col}_{agg}"] = float(vals.mean())
                elif agg_up == "SUM": row_result[f"{col}_{agg}"] = float(vals.sum())
                elif agg_up == "MIN": row_result[f"{col}_{agg}"] = float(vals.min())
                elif agg_up == "MAX": row_result[f"{col}_{agg}"] = float(vals.max())
            cache[cache_key] = row_result
        else:
            db  = conns.duckdb(db_file)
            sql = f"""
                SELECT {select_sql}
                FROM {table} f
                WHERE f.longitude BETWEEN {lon - dlon} AND {lon + dlon}
                  AND f.latitude  BETWEEN {lat - dlat} AND {lat + dlat}
                  AND sqrt(pow((f.longitude - {lon}) * 111320, 2)
                         + pow((f.latitude  - {lat}) * 111320
                             * cos(radians({lat})), 2)) <= {radius_m}
                {t_where}
                {extra_where}
            """
            
            db_row = db.execute(sql).fetchone()

            cache[cache_key] = (
                dict(zip(out_cols, db_row)) if db_row else {"count": 0}
            )

    # Vectorised broadcast: map each row's cache_key to its result in one pass
    n = len(df)
    col_arrays = {col: np.empty(n, dtype=object) for col in out_cols}
    for i in range(n):
        row_result = cache[cache_keys[i]]
        for col in out_cols:
            col_arrays[col][i] = row_result.get(col)

    return pd.DataFrame(col_arrays, index=df.index)


def batch_urban_atlas_luc_fraction(
    df: pd.DataFrame,
    conns: Connections,
    luc_code: str,
    radius_m: float,
    ua_year: Optional[int],
    out_col: str,
    raster: Optional["_PolyRaster"] = None,
) -> pd.DataFrame:
    """
    Compute urban_atlas luc_code fraction for every row in *df* in bulk.

    Fast path (raster available)
    ----------------------------
    If a precomputed _PolyRaster is supplied, each row is answered by a single
    array lookup: snap (lon, lat) to the nearest precomputed grid point and
    read the value.  For temporal features (ua_year=None, multiple surveys)
    the last_previous UA year relative to the LST row timestamp is selected.

    Slow path (no raster / fallback)
    ---------------------------------
    Iterates unique tile_ids (one Shapely R-tree query each) and broadcasts
    results.  Used on the first batch before raster build completes, or when
    the raster is disabled.

    Returns a single-column DataFrame named *out_col*, indexed like *df*.
    """
    lons     = df["longitude"].to_numpy(dtype=float)
    lats     = df["latitude"].to_numpy(dtype=float)

    # ---- Fast path: O(n) raster lookup ----
    if raster is not None:
        if ua_year is not None:
            layer_key = f"{luc_code}:{ua_year}"
            values = np.array(
                [raster.lookup(lons[i], lats[i], layer_key) or 0.0
                 for i in range(len(df))],
                dtype=float,
            )
        else:
            tss = df["timestamp"].to_numpy(dtype=str)
            values = np.array(
                [raster.lookup_ua_last_previous(
                     lons[i], lats[i], luc_code, int(tss[i][:4])) or 0.0
                 for i in range(len(df))],
                dtype=float,
            )
        return pd.DataFrame({out_col: values}, index=df.index)

    # ---- Slow path: per-tile Shapely queries ----
    db       = conns.spatialite("spatial.db")
    tile_ids = df["tile_id"].to_numpy(dtype=str)

    cache: Dict[str, float] = {}
    seen_tile: Dict[str, int] = {}
    for i, tile_id in enumerate(tile_ids):
        if tile_id not in seen_tile:
            seen_tile[tile_id] = i

    for tile_id, i in tqdm(seen_tile.items(),
                           desc=f"  ua {luc_code}",
                           unit="tile", position=2, leave=False,
                           dynamic_ncols=True):
        blobs = _ua_fetch_candidates(db, lons[i], lats[i], radius_m, luc_code, ua_year)
        if not blobs:
            cache[tile_id] = 0.0
        else:
            circle = _ua_make_circle(lons[i], lats[i], radius_m)
            cache[tile_id] = _ua_compute_fraction(blobs, circle)

    values = np.array([cache[tid] for tid in tile_ids], dtype=float)
    return pd.DataFrame({out_col: values}, index=df.index)


def batch_wis_fraction(
    df: pd.DataFrame,
    conns: Connections,
    attr_col: str,
    attr_val: str,
    radius_m: float,
    out_col: str,
    raster: Optional["_PolyRaster"] = None,
) -> pd.DataFrame:
    """
    Compute WIS polygon fraction for every row in *df* in bulk.

    Mirrors batch_urban_atlas_luc_fraction exactly.  WIS is static (no
    temporal component); the raster layer key is "wis:{attr_val}".

    Fast path: O(n) raster lookup when a precomputed _PolyRaster is supplied.
    Slow path: per-tile Shapely queries (fallback / first batch).

    Returns a single-column DataFrame named *out_col*, indexed like *df*.
    """
    lons = df["longitude"].to_numpy(dtype=float)
    lats = df["latitude"].to_numpy(dtype=float)

    # ---- Fast path: O(n) raster lookup ----
    if raster is not None:
        values = np.array(
            [raster.lookup_wis(lons[i], lats[i], attr_val) or 0.0
             for i in range(len(df))],
            dtype=float,
        )
        return pd.DataFrame({out_col: values}, index=df.index)

    # ---- Slow path: per-tile Shapely queries ----
    db       = conns.spatialite("spatial.db")
    tile_ids = df["tile_id"].to_numpy(dtype=str)

    cache: Dict[str, float] = {}
    seen_tile: Dict[str, int] = {}
    for i, tile_id in enumerate(tile_ids):
        if tile_id not in seen_tile:
            seen_tile[tile_id] = i

    for tile_id, i in tqdm(seen_tile.items(),
                           desc=f"  wis {attr_val}",
                           unit="tile", position=2, leave=False,
                           dynamic_ncols=True):
        blobs = _wis_fetch_candidates(db, lons[i], lats[i], radius_m, attr_col, attr_val)
        if not blobs:
            cache[tile_id] = 0.0
        else:
            circle = _ua_make_circle(lons[i], lats[i], radius_m)
            cache[tile_id] = _ua_compute_fraction(blobs, circle)

    values = np.array([cache[tid] for tid in tile_ids], dtype=float)
    return pd.DataFrame({out_col: values}, index=df.index)