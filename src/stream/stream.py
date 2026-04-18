"""
stream.py  -  /src/stream/stream.py
====================================
Spatiotemporal LST streaming layer.

Quick start
-----------
    from stream import StreamConfig, FeatureRegistry, nearest, aggregate_in_radius
    from stream import urban_atlas_luc_fraction
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

    # Urban Atlas land-use fraction within a radius
    reg.add(urban_atlas_luc_fraction("11100", radius_m=100))  # continuous urban fabric
    reg.add(urban_atlas_luc_fraction("14100", radius_m=100))  # green urban areas

    # Custom feature using framework building-blocks (row-level API)
    from stream import query_nearest, query_urban_atlas_luc_fraction
    def my_feature(row, connections):
        elev = query_nearest(connections, "dhm2", row.longitude, row.latitude,
                             row.timestamp, columns=["elevation"],
                             temporal="last_previous")
        frac = query_urban_atlas_luc_fraction(connections, row.longitude, row.latitude,
                                              radius_m=200, luc_code="11100")
        return {"dhm2_elev_log": math.log1p(elev.get("elevation", 0)),
                "urban_frac_200m": frac}
    reg.add_custom(my_feature, name="custom_urban_elev")

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

BATCH path (framework features - nearest, aggregate_in_radius,
            urban_atlas_luc_fraction):
  Each batch's coordinates are bulk-loaded into a SpatiaLite temporary table.
  A single spatial JOIN per feature replaces N per-row queries.
  For aggregate features, results are deduplicated by tile_id so rows sharing
  a tile share one query result (valid because all pixels in an H3 cell at
  resolution 9 are within ~200 m of each other).

ROW path (custom callables added via add_custom):
  Custom features receive one FeatureRow at a time plus a Connections object.
  They may call query_nearest(), query_radius(), and
  query_urban_atlas_luc_fraction() to reuse framework logic.
  Results are accumulated into pre-allocated column arrays (not list-of-dicts)
  and assembled into a DataFrame at batch end.

The 2x2 temporalxspatial framework
-----------------------------------
Spatial:
  nearest(dataset, columns, temporal, ...)
      -> the single closest point in space (optionally filtered by time)
  aggregate_in_radius(dataset, radius_m, columns, agg, temporal, ...)
      -> COUNT / AVG / SUM / MIN / MAX of all points within radius_m metres
  urban_atlas_luc_fraction(luc_code, radius_m, ua_year, ...)
      -> fraction [0, 1] of a circle's area occupied by polygons of luc_code

Temporal (applies to non-static datasets):
  "last_previous"   -> most recent observation with ts <= driving_ts
  "nearest"         -> observation with smallest |ts - driving_ts|
  "none"            -> no temporal filter (static datasets)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from pathlib import Path
from typing import (
    Dict, Generator, List, Optional, Tuple
)

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Sub-module imports
# ---------------------------------------------------------------------------
from connections import Connections
from distribution import DistributionTarget
from features import (
    FeatureRow, FeatureRegistry,
    _FeatureDescriptor,
    nearest, aggregate_in_radius,
    urban_atlas_luc_fraction, wis_fraction,
)
from geo import (
    DEFAULT_PREPARED, _LAT_DEG_PER_M, _LON_DEG_PER_M,
    GHENT_LON_MIN, GHENT_LON_MAX, GHENT_LAT_MIN, GHENT_LAT_MAX,
)
from logging_config import configure_logging
from poly_raster import (
    _PolyRaster,
    _ua_fetch_all_in_bbox, _wis_fetch_all_in_bbox,
    _rasterise_layer,
)
from queries import (
    query_nearest, query_radius,
    query_urban_atlas_luc_fraction, query_wis_fraction,
    batch_nearest, batch_radius,
    batch_urban_atlas_luc_fraction, batch_wis_fraction,
)

log = logging.getLogger("stream")


# ---------------------------------------------------------------------------
# StreamConfig
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
    lst_emissivity_mode: How to resolve LST from 3 emissivity columns (ASTER/MODIS/NDVI).
                    "any" (default): first non-null in order ASTER > MODIS > NDVI
                    "fallback": ASTER > MODIS (NDVI excluded)
                    "aster"/"modis"/"ndvi": use only that column
    lst_null_handling: "skip" (default) or "impute". Controls row filtering when LST is null.

    Distribution Targeting
    ----------------------
    Supports weighted sampling based on target distributions across dimensions:
      - temperature: LST values in degrees C
      - timestamp: observation dates (ISO format)
      - longitude / latitude: geographic coordinates

    Examples
    --------
    Temperature-only (backwards compatible):
        cfg = StreamConfig(Path("prepared_stream_data"))
        cfg.set_distribution({20: 0.4, 25: 0.4, 30: 0.2})

    With emissivity control:
        cfg = StreamConfig(
            Path("prepared_stream_data"),
            lst_emissivity_mode="fallback",  # ASTER > MODIS only
            lst_null_handling="impute"
        )

    Multi-dimensional (temperature + geographic location):
        cfg = StreamConfig(Path("prepared_stream_data"))
        cfg.set_distribution({
            "temperature": ({20: 0.3, 25: 0.4, 30: 0.3}, bin_edges_temp),
            "longitude": ({3.2: 0.5, 3.3: 0.5}, bin_edges_lon),
        })

    Streaming:
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
        raster_resolution_m: float = 15.0,
        lst_emissivity_mode: str = "any",
        lst_null_handling: str = "skip",
    ):
        """
        Parameters
        ----------
        prepared_data        : Path to prepared_stream_data/ from ingest.py.
        batch_size           : Rows per yielded DataFrame batch.
        partition_keys       : Restrict to specific YYYY-MM months.  None = all.
        raster_resolution_m  : Grid spacing for the precomputed polygon-fraction
                               raster (Urban Atlas and WIS).  Default 15 m =
                               half the 30 m LST pixel spacing, so every LST
                               pixel maps to a precomputed point within ~7.5 m —
                               well inside the acceptable margin for any query
                               radius >= 100 m.  Set 0 to disable precomputation
                               and always use the live Shapely path.
        lst_emissivity_mode  : How to select LST value from 3 emissivity sources:
                               "any" (default): first non-null (ASTER > MODIS > NDVI)
                               "fallback": ASTER > MODIS, never NDVI
                               "aster": return aster_lst only (null if missing)
                               "modis": return modis_lst only (null if missing)
                               "ndvi": return ndvi only (null if missing)
        lst_null_handling    : How to handle rows with null LST value:
                               "skip" (default): drop rows where selected LST is null
                               "impute": use fallback column if primary is null
                                        (only ASTER/MODIS can fallback to each other)
        """
        self.prepared            = Path(prepared_data)
        self.batch_size          = batch_size
        self._partitions         = partition_keys   # None -> all
        self.raster_resolution_m = raster_resolution_m
        self.lst_emissivity_mode = lst_emissivity_mode
        self.lst_null_handling   = lst_null_handling
        self._distribution: Optional[DistributionTarget] = None
        self._catalog_meta: Optional[dict] = None
        self._bin_edges_dict: Dict[str, List[float]] = {}  # temperature, timestamp, longitude, latitude
        self._partition_stats: Optional[List[dict]] = None

    def set_distribution(self, target: Dict) -> "StreamConfig":
        """
        Define a target distribution for weighted sampling.

        Supports both simple temperature-only and multi-dimensional targeting.

        Parameters
        ----------
        target : dict
            For simple temperature-only (backwards compatible):
                {15: 0.2, 20: 0.3, 25: 0.3, 30: 0.2}
            For multi-dimensional:
                {
                    "temperature": ({15: 0.2, 20: 0.3, 25: 0.3, 30: 0.2}, bin_edges),
                    "timestamp": ({...}, bin_edges),
                    ...
                }
        """
        self._distribution = DistributionTarget(target)
        return self

    def _resolve_lst_temperature(self, df: pd.DataFrame) -> np.ndarray:
        """
        Resolve LST temperature from 3 emissivity sources (aster_lst, modis_lst, ndvi).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: aster_lst, modis_lst, ndvi

        Returns
        -------
        np.ndarray
            Temperature column (1D array of floats, with NaN for unresolvable rows).
            Behavior determined by self.lst_emissivity_mode and self.lst_null_handling.

        Mode semantics
        -----------
        "any"        : first non-null in order [aster_lst, modis_lst, ndvi]
        "fallback"   : first non-null in order [aster_lst, modis_lst] (never NDVI)
        "aster"      : aster_lst only (NaN if missing)
        "modis"      : modis_lst only (NaN if missing)
        "ndvi"       : ndvi only (NaN if missing)

        Null handling after resolution
        "skip"       : rows with null LST remain NaN (caller may filter)
        "impute"     : fallback to next source if primary is null (ASTER<->MODIS only)
        """
        aster = df["aster_lst"].to_numpy()
        modis = df["modis_lst"].to_numpy()
        ndvi = df["ndvi"].to_numpy()

        mode = self.lst_emissivity_mode.lower()
        handling = self.lst_null_handling.lower()

        if mode == "any":
            # First non-null in [aster, modis, ndvi]
            temp = np.where(pd.notna(aster), aster,
                   np.where(pd.notna(modis), modis, ndvi))

        elif mode == "fallback":
            # First non-null in [aster, modis] only (never NDVI)
            temp = np.where(pd.notna(aster), aster, modis)

        elif mode == "aster":
            temp = aster.copy()

        elif mode == "modis":
            temp = modis.copy()

        elif mode == "ndvi":
            temp = ndvi.copy()

        else:
            raise ValueError(f"Unknown lst_emissivity_mode: {mode}")

        # Apply null handling (impute: fallback to next source)
        if handling == "impute" and mode in ("aster", "modis"):
            if mode == "aster":
                # Fallback ASTER -> MODIS (never NDVI)
                temp = np.where(pd.notna(temp), temp, modis)
            elif mode == "modis":
                # Fallback MODIS -> ASTER (never NDVI)
                temp = np.where(pd.notna(temp), temp, aster)

        return temp

    def _load_catalog(self) -> None:
        """Read catalog.duckdb once at stream start."""
        cat_path = self.prepared / "catalog.duckdb"
        if not cat_path.exists():
            raise FileNotFoundError(
                f"Catalog not found: {cat_path}\n"
                "Run ingest.py first to build the feature store."
            )
        log.info("catalog: loading from %s", cat_path)
        t0 = time.perf_counter()
        conn = duckdb.connect(str(cat_path), read_only=True)
        log.debug("catalog: duckdb connection opened in %.3fs", time.perf_counter() - t0)

        # Dataset metadata
        t1 = time.perf_counter()
        rows = conn.execute("SELECT * FROM dataset_metadata").fetchall()
        cols = [d[0] for d in conn.description]
        self._catalog_meta = {}
        for r in rows:
            row_dict = dict(zip(cols, r))
            self._catalog_meta[row_dict["dataset_id"]] = row_dict
        log.debug("catalog: dataset_metadata loaded (%d datasets) in %.3fs",
                  len(self._catalog_meta), time.perf_counter() - t1)

        # Histogram config - load bin edges for all dimensions (stored as VARCHAR[] arrays)
        t1 = time.perf_counter()
        rows = conn.execute("SELECT dataset_id, bin_edges FROM histogram_config").fetchall()
        for dataset_id, edges_array in rows:
            if edges_array:
                # Parse numeric dimensions (temperature, longitude, latitude) to float
                # Keep timestamp as strings (e.g., "2000-Q1")
                if dataset_id in ("temperature", "longitude", "latitude"):
                    self._bin_edges_dict[dataset_id] = [float(v) for v in edges_array]
                else:  # timestamp or other string-based dimensions
                    self._bin_edges_dict[dataset_id] = list(edges_array)
        if "temperature" in self._bin_edges_dict:
            log.debug("catalog: loaded %d temperature bins", len(self._bin_edges_dict["temperature"]))
        if "timestamp" in self._bin_edges_dict:
            log.debug("catalog: loaded %d timestamp bins", len(self._bin_edges_dict["timestamp"]))
        if "longitude" in self._bin_edges_dict:
            log.debug("catalog: loaded %d longitude bins", len(self._bin_edges_dict["longitude"]))
        if "latitude" in self._bin_edges_dict:
            log.debug("catalog: loaded %d latitude bins", len(self._bin_edges_dict["latitude"]))
        log.debug("catalog: histogram_config loaded in %.3fs", time.perf_counter() - t1)

        # Partition list
        t1 = time.perf_counter()
        rows = conn.execute(
            "SELECT DISTINCT partition_key FROM partition_statistics "
            "WHERE dataset_id = 'lst' ORDER BY partition_key"
        ).fetchall()
        self._partition_keys = [r[0] for r in rows]
        log.debug("catalog: %d distinct partition_keys loaded in %.3fs",
                  len(self._partition_keys), time.perf_counter() - t1)

        # Keep the connection open - _select_partitions may query it
        self._catalog_conn = conn
        log.info("catalog: total load time %.3fs", time.perf_counter() - t0)

    def _build_poly_rasters(
        self,
        registry: "FeatureRegistry",
    ) -> Optional["_PolyRaster"]:
        """
        Precompute polygon-fraction features (Urban Atlas, WIS) onto a regular
        lon/lat grid and wire the result into every registered descriptor that
        carries a _raster_ref.

        The raster is built **once per stream() call**, independently of which
        LST partitions were selected.  It always covers the full Ghent spatial
        extent (GHENT_LON/LAT_MIN/MAX from geo.py) so that the precomputed
        grid is valid for any combination of partitions without rebuilding.

        Algorithm
        ---------
        1.  Use the fixed Ghent bounding box as the raster extent.  This
            decouples raster build cost from partition selection: a run that
            streams only summer months gets the same raster as a full run,
            and no LST table scan is needed to determine the extent.

        2.  Construct a _PolyRaster grid at self.raster_resolution_m spacing,
            with a 1-cell border pad to avoid edge effects when LST pixels
            land on the boundary.

        3.  For each unique (luc_code, radius_m, ua_year) / (attr_col,
            attr_val, radius_m) combination found in the registered
            descriptors, rasterise the SpatiaLite polygons onto the grid.
            UA descriptors with ua_year=None expand to **all four survey
            years** (2006, 2012, 2018, 2021); at query time lookup_ua_
            last_previous() selects the appropriate year per LST row.
            Duplicate (luc_code, radius_m, year) triples produced by
            multiple descriptors that share a luc_code/radius are de-
            duplicated before rasterisation — each unique layer is computed
            exactly once.

        4.  Attach the completed raster to each descriptor's _raster_ref cell.

        Returns the shared _PolyRaster, or None if precomputation is disabled
        (raster_resolution_m == 0) or no raster-capable descriptors exist.
        """
        if self.raster_resolution_m <= 0:
            log.info("raster: precomputation disabled (raster_resolution_m=0)")
            return None

        raster_descs = [
            d for d in registry._descriptors
            if hasattr(d, "_raster_ref")
        ]
        if not raster_descs:
            log.info("raster: no raster-capable descriptors registered — skipping")
            return None

        # ---- Step 1: use the fixed Ghent spatial extent ----
        # We deliberately do NOT query the LST table here.  The raster must
        # cover the full city regardless of which partitions (months/years)
        # are selected for this particular stream() run.
        lon_min, lon_max = GHENT_LON_MIN, GHENT_LON_MAX
        lat_min, lat_max = GHENT_LAT_MIN, GHENT_LAT_MAX
        log.info(
            "raster: using fixed Ghent bbox lon=[%.4f, %.4f] lat=[%.4f, %.4f]",
            lon_min, lon_max, lat_min, lat_max,
        )

        # ---- Step 2: build grid ----
        res_m    = self.raster_resolution_m
        step_lon = res_m * _LON_DEG_PER_M
        step_lat = res_m * _LAT_DEG_PER_M
        pad      = 1   # 1 cell border

        lon0  = lon_min - pad * step_lon
        lat0  = lat_min - pad * step_lat
        n_lon = int(math.ceil((lon_max - lon_min) / step_lon)) + 2 * pad + 1
        n_lat = int(math.ceil((lat_max - lat_min) / step_lat)) + 2 * pad + 1

        raster = _PolyRaster(
            lon0=lon0, lat0=lat0,
            step_lon=step_lon, step_lat=step_lat,
            n_lon=n_lon, n_lat=n_lat,
            resolution_m=res_m,
        )
        raster.log_summary()

        # ---- Step 3: identify unique layer specs ----
        # Use sets/dicts so that multiple descriptors sharing the same
        # (luc_code, radius_m) or (attr_col, attr_val, radius_m) don't
        # cause redundant rasterisation passes.
        #
        # UA: key=(luc_code, radius_m) → set of concrete ua_years to compute.
        #   ua_year=None  → expand to all four survey years so that
        #                   lookup_ua_last_previous() works at query time.
        #   ua_year=<int> → add only that year.
        #
        # Both "ua" (single-code) and "ua_classifications" (multi-code)
        # descriptors contribute their LUC codes to the same dict so that
        # each unique (luc_code, radius_m, year) triple is rasterised
        # exactly once and shared across all descriptors that need it.
        ua_years_to_compute: Dict[Tuple[str, float], set] = {}

        wis_specs: Dict[Tuple, List] = {}

        def _register_ua_luc(luc_code: str, radius_m: float, ua_year) -> None:
            key = (luc_code, radius_m)
            if key not in ua_years_to_compute:
                ua_years_to_compute[key] = set()
            if ua_year is None:
                ua_years_to_compute[key].update(_PolyRaster.UA_YEARS)
            else:
                ua_years_to_compute[key].add(ua_year)

        for desc in raster_descs:
            if desc._raster_type == "ua":
                _register_ua_luc(desc._luc_code, desc._radius_m, desc._ua_year)
            elif desc._raster_type == "ua_classifications":
                for luc_codes in desc._classification_map.values():
                    for luc_code in luc_codes:
                        _register_ua_luc(luc_code, desc._radius_m, desc._ua_year)
            elif desc._raster_type == "wis":
                key = (desc._attr_col, desc._attr_val, desc._radius_m)
                wis_specs.setdefault(key, []).append(desc)

        # Flatten to ordered lists for the rasterisation loop
        ua_layer_list: List[Tuple[str, float, int]] = []
        for (luc_code, radius_m), years in ua_years_to_compute.items():
            for y in sorted(years):          # deterministic order
                ua_layer_list.append((luc_code, radius_m, y))

        wis_layer_list = [(ac, av, rm) for (ac, av, rm) in wis_specs]

        all_layers = (
            [("ua", x) for x in ua_layer_list] +
            [("wis", x) for x in wis_layer_list]
        )

        # Open SpatiaLite for polygon fetching
        db = sqlite3.connect(str(self.prepared / "spatial.db"))
        db.enable_load_extension(True)
        for lib in ["mod_spatialite", "mod_spatialite.so",
                    "mod_spatialite.dylib",
                    "/usr/lib/x86_64-linux-gnu/mod_spatialite.so"]:
            try:
                db.load_extension(lib)
                break
            except sqlite3.OperationalError:
                continue

        total_cells    = n_lon * n_lat
        t_raster_start = time.perf_counter()

        log.info(
            "raster: precomputing %d UA layer(s) + %d WIS layer(s) "
            "over %d x %d = %d grid cells at %.0f m resolution",
            len(ua_layer_list), len(wis_layer_list),
            n_lon, n_lat, total_cells, res_m,
        )

        with tqdm(total=len(all_layers), desc="Raster precompute",
                  unit="layer", position=0, dynamic_ncols=True) as layer_bar:
            for layer_type, spec in all_layers:
                if layer_type == "ua":
                    luc_code, radius_m, year = spec
                    layer_key = f"{luc_code}:{year}"
                    arr = raster.add_layer(layer_key)
                    t_layer = time.perf_counter()

                    dlat_r = radius_m * _LAT_DEG_PER_M
                    dlon_r = radius_m * _LON_DEG_PER_M
                    all_blobs = _ua_fetch_all_in_bbox(
                        db, luc_code, year,
                        raster.grid_lon(0) - dlon_r, raster.grid_lat(0) - dlat_r,
                        raster.grid_lon(n_lon - 1) + dlon_r,
                        raster.grid_lat(n_lat - 1) + dlat_r,
                    )
                    log.info("raster: UA layer '%s' — %d polygons in bbox",
                             layer_key, len(all_blobs))

                    n_nonzero = _rasterise_layer(
                        arr, raster, all_blobs, radius_m,
                        desc=f"  UA {luc_code} {year}",
                    )
                    raster._coverage[layer_key] = n_nonzero
                    elapsed = time.perf_counter() - t_layer
                    log.info(
                        "raster: UA layer '%s' (r=%.0fm) done — "
                        "%d x %d cells, %d non-zero, %.1fs (%.0f cells/s)",
                        layer_key, radius_m, n_lon, n_lat, n_nonzero, elapsed,
                        (n_lon * n_lat) / elapsed if elapsed > 0 else 0,
                    )
                    layer_bar.update(1)

                elif layer_type == "wis":
                    attr_col, attr_val, radius_m = spec
                    layer_key = f"wis:{attr_val}"
                    arr = raster.add_layer(layer_key)
                    t_layer = time.perf_counter()

                    dlat_r = radius_m * _LAT_DEG_PER_M
                    dlon_r = radius_m * _LON_DEG_PER_M
                    all_blobs = _wis_fetch_all_in_bbox(
                        db, attr_col, attr_val,
                        raster.grid_lon(0) - dlon_r, raster.grid_lat(0) - dlat_r,
                        raster.grid_lon(n_lon - 1) + dlon_r,
                        raster.grid_lat(n_lat - 1) + dlat_r,
                    )
                    log.info("raster: WIS layer '%s' — %d polygons in bbox",
                             layer_key, len(all_blobs))

                    n_nonzero = _rasterise_layer(
                        arr, raster, all_blobs, radius_m,
                        desc=f"  WIS {attr_val}",
                    )
                    raster._coverage[layer_key] = n_nonzero
                    elapsed = time.perf_counter() - t_layer
                    log.info(
                        "raster: WIS layer '%s' (r=%.0fm) done — "
                        "%d x %d cells, %d non-zero, %.1fs (%.0f cells/s)",
                        layer_key, radius_m, n_lon, n_lat, n_nonzero, elapsed,
                        (n_lon * n_lat) / elapsed if elapsed > 0 else 0,
                    )
                    layer_bar.update(1)

        db.close()

        total_elapsed = time.perf_counter() - t_raster_start
        log.info(
            "raster: precomputation complete — %d layers, %d total cells, "
            "%.1fs total (%.0f cells/s per layer avg)",
            len(all_layers), total_cells * len(all_layers),
            total_elapsed,
            (total_cells * len(all_layers)) / total_elapsed if total_elapsed > 0 else 0,
        )
        raster.log_summary()

        # ---- Step 4: wire raster into all descriptor _raster_ref cells ----
        for desc in raster_descs:
            desc._raster_ref[0] = raster
            log.debug("raster: wired into descriptor '%s'", desc.name)

        return raster

    def _select_partitions(self) -> List[Tuple[str, float]]:
        """
        Return list of (partition_key, weight) sorted by weight descending.

        When no distribution is set: all partition_keys with weight 1.0.
        When a distribution is set: score each partition using the selected
        dimensions (temperature, timestamp, coordinates). Score is computed
        inside DuckDB as a product of dimension weights.
        """
        all_keys = getattr(self, "_partition_keys", [])

        if self._partitions:
            pk_set   = set(self._partitions)
            all_keys = [pk for pk in all_keys if pk in pk_set]

        if not all_keys:
            return []

        if self._distribution is None or not self._distribution.dimensions:
            return [(pk, 1.0) for pk in sorted(all_keys)]

        # Build SQL to score partitions on requested dimensions
        pk_filter = ", ".join(f"'{pk}'" for pk in all_keys)
        dimensions_to_score = list(self._distribution.dimensions.keys())

        # Build SELECT clause to extract histogram arrays (stored as BIGINT[])
        select_cols = ["partition_key"]
        if "temperature" in dimensions_to_score:
            select_cols.append(
                "CAST(histogram_counts AS DOUBLE[]) AS temp_counts"
            )
        if "timestamp" in dimensions_to_score:
            select_cols.append(
                "CAST(timestamp_histogram_counts AS DOUBLE[]) AS ts_counts"
            )
        if "longitude" in dimensions_to_score:
            select_cols.append(
                "CAST(longitude_histogram_counts AS DOUBLE[]) AS lon_counts"
            )
        if "latitude" in dimensions_to_score:
            select_cols.append(
                "CAST(latitude_histogram_counts AS DOUBLE[]) AS lat_counts"
            )

        select_clause = ", ".join(select_cols)

        # Build WHERE clause to ensure all requested histogram arrays are present
        where_clauses = ["dataset_id = 'lst'", f"partition_key IN ({pk_filter})"]
        if "temperature" in dimensions_to_score:
            where_clauses.append("histogram_counts IS NOT NULL")
        if "timestamp" in dimensions_to_score:
            where_clauses.append("timestamp_histogram_counts IS NOT NULL")
        if "longitude" in dimensions_to_score:
            where_clauses.append("longitude_histogram_counts IS NOT NULL")
        if "latitude" in dimensions_to_score:
            where_clauses.append("latitude_histogram_counts IS NOT NULL")

        where_clause = " AND ".join(where_clauses)

        # Build weight expressions for each dimension
        weight_exprs = []

        if "temperature" in dimensions_to_score:
            temp_dim = self._distribution.dimensions["temperature"]
            temp_edges = temp_dim.bin_edges
            n_bins = len(temp_edges) - 1
            target_vec = [0.0] * n_bins
            for edge, desired_prop in temp_dim.target.items():
                bin_idx = min(
                    range(n_bins),
                    key=lambda i: abs(temp_edges[i] - edge),
                )
                target_vec[bin_idx] += desired_prop
            target_json = json.dumps(target_vec)
            weight_exprs.append(f"""
                list_aggregate(
                    list_transform(
                        generate_series(0, len(temp_counts) - 1),
                        i -> CASE
                            WHEN list_sum(temp_counts) = 0 THEN 0.0
                            ELSE least(
                                temp_counts[i + 1]::DOUBLE / list_sum(temp_counts),
                                ({target_json}::DOUBLE[])[i + 1]
                            )
                        END
                    ),
                    'sum'
                )
            """)

        if "timestamp" in dimensions_to_score:
            ts_dim = self._distribution.dimensions["timestamp"]
            ts_edges = ts_dim.bin_edges
            n_bins = len(ts_edges) - 1
            target_vec = [0.0] * n_bins
            for edge, desired_prop in ts_dim.target.items():
                bin_idx = min(
                    range(n_bins),
                    key=lambda i: abs(ts_edges[i] - edge),
                )
                target_vec[bin_idx] += desired_prop
            target_json = json.dumps(target_vec)
            weight_exprs.append(f"""
                list_aggregate(
                    list_transform(
                        generate_series(0, len(ts_counts) - 1),
                        i -> CASE
                            WHEN list_sum(ts_counts) = 0 THEN 0.0
                            ELSE least(
                                ts_counts[i + 1]::DOUBLE / list_sum(ts_counts),
                                ({target_json}::DOUBLE[])[i + 1]
                            )
                        END
                    ),
                    'sum'
                )
            """)

        if "longitude" in dimensions_to_score:
            lon_dim = self._distribution.dimensions["longitude"]
            lon_edges = lon_dim.bin_edges
            n_bins = len(lon_edges) - 1
            target_vec = [0.0] * n_bins
            for edge, desired_prop in lon_dim.target.items():
                bin_idx = min(
                    range(n_bins),
                    key=lambda i: abs(lon_edges[i] - edge),
                )
                target_vec[bin_idx] += desired_prop
            target_json = json.dumps(target_vec)
            weight_exprs.append(f"""
                list_aggregate(
                    list_transform(
                        generate_series(0, len(lon_counts) - 1),
                        i -> CASE
                            WHEN list_sum(lon_counts) = 0 THEN 0.0
                            ELSE least(
                                lon_counts[i + 1]::DOUBLE / list_sum(lon_counts),
                                ({target_json}::DOUBLE[])[i + 1]
                            )
                        END
                    ),
                    'sum'
                )
            """)

        if "latitude" in dimensions_to_score:
            lat_dim = self._distribution.dimensions["latitude"]
            lat_edges = lat_dim.bin_edges
            n_bins = len(lat_edges) - 1
            target_vec = [0.0] * n_bins
            for edge, desired_prop in lat_dim.target.items():
                bin_idx = min(
                    range(n_bins),
                    key=lambda i: abs(lat_edges[i] - edge),
                )
                target_vec[bin_idx] += desired_prop
            target_json = json.dumps(target_vec)
            weight_exprs.append(f"""
                list_aggregate(
                    list_transform(
                        generate_series(0, len(lat_counts) - 1),
                        i -> CASE
                            WHEN list_sum(lat_counts) = 0 THEN 0.0
                            ELSE least(
                                lat_counts[i + 1]::DOUBLE / list_sum(lat_counts),
                                ({target_json}::DOUBLE[])[i + 1]
                            )
                        END
                    ),
                    'sum'
                )
            """)

        # Build the product of weights (one weight expression per dimension)
        if len(weight_exprs) == 1:
            weight_expr = weight_exprs[0]
        else:
            # Product of all dimension weights
            weight_expr = " * ".join(f"({expr})" for expr in weight_exprs)

        sql = f"""
            WITH parsed AS (
                SELECT {select_clause}
                FROM partition_statistics
                WHERE {where_clause}
            ),
            scored AS (
                SELECT
                    partition_key,
                    {weight_expr} AS weight
                FROM parsed
            )
            SELECT partition_key, weight
            FROM scored
            WHERE weight > 0
            ORDER BY weight DESC
        """

        t0 = time.perf_counter()
        rows = self._catalog_conn.execute(sql).fetchall()
        log.info("catalog: partition scoring done in %.3fs (%d partitions) "
                 "using dimensions: %s",
                 time.perf_counter() - t0, len(rows), dimensions_to_score)
        return [(r[0], r[1]) for r in rows]

    def stream(
        self,
        registry: Optional[FeatureRegistry] = None,
        batch_size: Optional[int] = None,
        max_rows: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Stream LST data with engineered features, one DataFrame batch at a time.

        Parameters
        ----------
        registry   : FeatureRegistry with registered features.
                     Pass None (or an empty registry) to stream raw LST only.
        batch_size : override the batch_size set in __init__.
        max_rows   : stop after yielding this many rows total.  Applies both
                     to full-dataset streaming and distribution-weighted runs.
                     The last batch may be smaller than batch_size.

        Yields
        ------
        pd.DataFrame, shape (<= batch_size, 10 + n_features)
        Columns: longitude, latitude, temperature,
                 aster_lst, modis_lst, ndvi,
                 image_id, timestamp, partition_key, tile_id,
                 [feature columns...]
        Note: 'temperature' is resolved from aster_lst/modis_lst/ndvi using
              lst_emissivity_mode and lst_null_handling settings.

        Progress
        --------
        Two tqdm bars are shown:
          Outer - one tick per partition (247 total), postfix shows rows
                  yielded, throughput, and current partition key.
          Inner - one tick per batch within the current partition, postfix
                  shows per-feature timing once the first batch completes.

        Early stopping
        --------------
        Stops immediately once max_rows is reached.  When a distribution
        target is set, partitions are ordered by weight descending so the
        highest-value data comes first - set max_rows to collect a well-
        distributed sample without streaming the full dataset.
        """
        if registry is None:
            registry = FeatureRegistry()

        bs = batch_size or self.batch_size
        log.info("stream: starting - batch_size=%d, max_rows=%s, features=%d "
                 "(batch=%d row=%d)",
                 bs, max_rows, len(registry),
                 len(registry._batch_descriptors),
                 len(registry._row_descriptors))

        t0 = time.perf_counter()
        self._load_catalog()
        partitions = self._select_partitions()
        log.info("stream: catalog ready, %d partitions selected in %.3fs",
                 len(partitions), time.perf_counter() - t0)

        if not partitions:
            log.warning("stream: no partitions selected - nothing to stream")
            return

        lst_meta = self._catalog_meta.get("lst", {})
        lst_db   = self.prepared / lst_meta.get("db_file", "lst.duckdb")

        log.info("stream: opening LST database %s", lst_db)
        t0 = time.perf_counter()
        lst_conn      = duckdb.connect(str(lst_db), read_only=True)
        log.info("stream: LST database opened in %.3fs", time.perf_counter() - t0)

        log.debug("stream: initialising feature Connections object")
        feature_conns = Connections(self.prepared, self._catalog_meta)

        # Force SpatiaLite to initialise before any Shapely geometry call.
        log.info("stream: pre-loading SpatiaLite connection ...")
        t0 = time.perf_counter()
        try:
            feature_conns.spatialite("spatial.db")
            log.info("stream: SpatiaLite connection ready in %.3fs",
                     time.perf_counter() - t0)
        except Exception as exc:
            log.error("stream: SpatiaLite failed to load - %s", exc, exc_info=True)
            raise

        # ---- Precompute polygon rasters (UA + WIS) before the batch loop ----
        if registry is not None:
            self._build_poly_rasters(registry)

        has_batch = len(registry._batch_descriptors) > 0
        has_row   = len(registry._row_descriptors)   > 0

        # Session-level counters
        session_rows   = 0
        session_t0     = time.perf_counter()
        stop_early     = False

        try:
            part_bar = tqdm(
                partitions,
                desc="Partitions",
                unit="partition",
                position=0,
                dynamic_ncols=True,
            )
            for part_idx, (partition_key, weight) in enumerate(part_bar):
                if stop_early:
                    break

                part_t0   = time.perf_counter()
                part_rows = 0

                part_bar.set_postfix({
                    "pk":      partition_key,
                    "weight":  f"{weight:.3f}",
                    "yielded": f"{session_rows:,}",
                }, refresh=True)

                log.info("partition %s [%d/%d] (w=%.3f): executing LST cursor",
                         partition_key, part_idx + 1, len(partitions), weight)

                # Estimate batch count BEFORE opening the streaming cursor.
                # Executing any query on lst_conn while the streaming cursor is
                # open replaces the active DuckDB result set, causing fetchmany
                # to return [] immediately.  The COUNT must complete first.
                est_batches = None
                try:
                    n_part = lst_conn.execute(
                        "SELECT COUNT(*) FROM lst WHERE partition_key = ?",
                        [partition_key]
                    ).fetchone()[0]
                    est_batches = max(1, (n_part + bs - 1) // bs)
                    log.debug("partition %s: ~%d rows, ~%d batches",
                              partition_key, n_part, est_batches)
                except Exception:
                    pass

                cursor = lst_conn.execute(
                    "SELECT longitude, latitude, aster_lst, modis_lst, ndvi, "
                    "       image_id, timestamp, partition_key, tile_id "
                    "FROM lst "
                    "WHERE partition_key = ?",
                    [partition_key],
                )
                log.debug("partition %s: cursor ready in %.3fs",
                          partition_key, time.perf_counter() - part_t0)

                batch_num = 0
                first_batch_feat_time: Optional[float] = None

                batch_bar = tqdm(
                    total=est_batches,
                    desc=f"  {partition_key}",
                    unit="batch",
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                )

                try:
                    while True:
                        # Respect max_rows before fetching
                        if max_rows is not None:
                            remaining = max_rows - session_rows
                            if remaining <= 0:
                                stop_early = True
                                break
                            fetch_n = min(bs, remaining)
                        else:
                            fetch_n = bs

                        t_fetch = time.perf_counter()
                        raw_rows = cursor.fetchmany(fetch_n)
                        fetch_ms = (time.perf_counter() - t_fetch) * 1000
                        log.debug("partition %s batch %d: fetchmany(%d) -> %d rows "
                                  "in %.0fms",
                                  partition_key, batch_num, fetch_n,
                                  len(raw_rows), fetch_ms)

                        if not raw_rows:
                            break

                        base_df = pd.DataFrame(raw_rows, columns=FeatureRow._fields)

                        # Resolve LST temperature from 3 emissivity sources based on mode
                        temperature = self._resolve_lst_temperature(base_df)
                        base_df.insert(2, "temperature", temperature)

                        if not has_batch and not has_row:
                            yield base_df
                            n_yielded = len(raw_rows)
                        else:
                            out_cols_data: Dict[str, np.ndarray] = {
                                col: base_df[col].to_numpy()
                                for col in base_df.columns
                            }

                            t_feat = time.perf_counter()

                            if has_batch:
                                batch_feat_df = registry.compute_batch_features(
                                    base_df, feature_conns
                                )
                                for col in batch_feat_df.columns:
                                    out_cols_data[col] = batch_feat_df[col].to_numpy()

                            if has_row:
                                row_feat_df = registry.compute_row_features(
                                    raw_rows, feature_conns
                                )
                                for col in row_feat_df.columns:
                                    out_cols_data[col] = row_feat_df[col].to_numpy()

                            feat_elapsed = time.perf_counter() - t_feat
                            if first_batch_feat_time is None:
                                first_batch_feat_time = feat_elapsed
                                log.info("partition %s batch %d: first feature batch "
                                         "in %.2fs - cols: %s",
                                         partition_key, batch_num, feat_elapsed,
                                         [c for c in out_cols_data
                                          if c not in FeatureRow._fields])

                            result_df = pd.DataFrame(out_cols_data)
                            yield result_df
                            n_yielded = len(result_df)

                        session_rows += n_yielded
                        part_rows    += n_yielded
                        batch_num    += 1

                        elapsed    = time.perf_counter() - session_t0
                        throughput = session_rows / elapsed if elapsed > 0 else 0

                        batch_bar.update(1)
                        batch_bar.set_postfix({
                            "rows":   f"{part_rows:,}",
                            "r/s":    f"{throughput:,.0f}",
                            "feat_s": f"{first_batch_feat_time:.1f}s"
                                      if first_batch_feat_time else "...",
                        }, refresh=False)
                        part_bar.set_postfix({
                            "pk":      partition_key,
                            "weight":  f"{weight:.3f}",
                            "yielded": f"{session_rows:,}",
                            "r/s":     f"{throughput:,.0f}",
                        }, refresh=False)

                        if max_rows is not None and session_rows >= max_rows:
                            stop_early = True
                            break

                finally:
                    batch_bar.close()

                part_elapsed = time.perf_counter() - part_t0
                part_rps     = part_rows / part_elapsed if part_elapsed > 0 else 0
                log.info("partition %s: done - %d rows in %.1fs (%.0f rows/s)",
                         partition_key, part_rows, part_elapsed, part_rps)

                if stop_early:
                    log.info("stream: early stop after %d rows (max_rows=%s)",
                             session_rows, max_rows)
                    break

        finally:
            part_bar.close()
            session_elapsed = time.perf_counter() - session_t0
            session_rps     = session_rows / session_elapsed if session_elapsed > 0 else 0
            log.info("stream: session complete - %d rows in %.1fs (%.0f rows/s)",
                     session_rows, session_elapsed, session_rps)
            lst_conn.close()
            feature_conns.close()
            if hasattr(self, "_catalog_conn") and self._catalog_conn:
                try:
                    self._catalog_conn.close()
                except Exception:
                    pass
                self._catalog_conn = None

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
                   Passed directly to stream() so early stopping happens
                   at fetch time, not after feature computation.
        """
        chunks = []
        for batch_df in self.stream(registry, batch_size=batch_size,
                                    max_rows=max_rows):
            chunks.append(batch_df)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()