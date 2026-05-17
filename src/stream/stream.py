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
    reg.add(nearest("dhm",         columns=["elevation"],  temporal="last_previous"))
    reg.add(nearest("ndvi",        columns=["ndvi"],       temporal="nearest"))
    reg.add(aggregate_in_radius("trees", radius_m=50,      columns=["hoogte"],
                                 agg="count",              temporal="none"))

    # Urban Atlas land-use fraction within a radius
    reg.add(urban_atlas_luc_fraction("11100", radius_m=100))  # continuous urban fabric
    reg.add(urban_atlas_luc_fraction("14100", radius_m=100))  # green urban areas

    # Custom feature using framework building-blocks (row-level API)
    from stream import query_nearest, query_urban_atlas_luc_fraction
    def my_feature(row, connections):
        elev = query_nearest(connections, "dhm", row.longitude, row.latitude,
                             row.timestamp, columns=["elevation"],
                             temporal="last_previous")
        frac = query_urban_atlas_luc_fraction(connections, row.longitude, row.latitude,
                                              radius_m=200, luc_code="11100")
        return {"dhm_elev_log": math.log1p(elev.get("elevation", 0)),
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
import hashlib
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
from stream.config import DIMENSION_CATALOG, get_dimension_edges
from stream.connections import Connections
from stream.distribution import DistributionTarget
from stream.features import (
    FeatureRow, FeatureRegistry,
    _FeatureDescriptor,
    nearest, aggregate_in_radius,
    urban_atlas_luc_fraction, wis_fraction,
)
from stream.geo import (
    DEFAULT_PREPARED, _LAT_DEG_PER_M, _LON_DEG_PER_M,
    GHENT_LON_MIN, GHENT_LON_MAX, GHENT_LAT_MIN, GHENT_LAT_MAX,
)
from stream.logging_config import configure_logging
from stream.poly_raster import (
    _PolyRaster,
    _ua_fetch_all_in_bbox, _wis_fetch_all_in_bbox,
    _rasterise_layer, _rasterise_layer_fft,
)
from stream.queries import (
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
    Supports weighted partition sampling across any dimension registered in
    config.DIMENSION_CATALOG.  Pass {} as the target dict for any dimension
    to request a uniform (flat) distribution over all its bins.

    Available dimensions (see config.DIMENSION_CATALOG for the full list):
      - temperature    LST value in °C
      - timestamp      quarterly label "YYYY-Q{1..4}" (coarse year scoring)
      - year           calendar year (2000–2025)
      - month_of_year  month 1–12
      - day_of_month   day within month 1–31
      - day_of_year    day within year 1–366 (12 monthly-breakpoint bins)
      - hour_of_day    fractional UTC hour 0.0–24.0
      - longitude / latitude  geographic coordinates

    Examples
    --------
    Temperature-only (backwards compatible):
        cfg = StreamConfig(Path("prepared_stream_data"))
        cfg.set_distribution({20: 0.4, 25: 0.4, 30: 0.2})

    Even distribution across year, month, and hour:
        from config import get_dimension_edges
        cfg.set_distribution({
            "year":          ({}, get_dimension_edges("year")),
            "month_of_year": ({}, get_dimension_edges("month_of_year")),
            "hour_of_day":   ({}, get_dimension_edges("hour_of_day")),
        })

    Mixed: skewed temperature + uniform hour-of-day:
        cfg.set_distribution({
            "temperature": ({20: 0.3, 25: 0.5, 30: 0.2}, get_dimension_edges("temperature")),
            "hour_of_day": ({},                            get_dimension_edges("hour_of_day")),
        })

    Streaming:
        reg = FeatureRegistry()
        reg.add(nearest("dhm", ["elevation"], temporal="last_previous"))
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
        raster_fft: bool = True,
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
        raster_fft           : Use FFT convolution to rasterise polygon layers
                               (default True).  FFT cost is dominated by one
                               2-D convolution regardless of polygon count, so
                               it is faster than the vector path for any class
                               with more than ~20 polygons in the grid extent.
                               Set False to force the Shapely vector path
                               (useful for debugging or very sparse UA classes).
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
        self.raster_fft          = raster_fft
        self.lst_emissivity_mode = lst_emissivity_mode
        self.lst_null_handling   = lst_null_handling
        self._distribution: Optional[DistributionTarget] = None
        self._catalog_meta: Optional[dict] = None
        # Bin edges keyed by dimension name, loaded from catalog histogram_config.
        # Numeric dimensions store list[float]; string dimensions store list[str].
        self._bin_edges_dict: Dict[str, list] = {}
        self._partition_stats: Optional[List[dict]] = None
        # Raster cache lives two levels above stream.py: /src/stream_cache/rasters/
        self._raster_cache_dir: Path = (
            Path(__file__).resolve().parent.parent / "stream_cache" / "rasters"
        )

    def set_distribution(self, target: Dict) -> "StreamConfig":
        """
        Define a target distribution for weighted partition sampling.

        Parameters
        ----------
        target : dict — two accepted formats:

            Multi-dimension (recommended):
                {
                    "year":          ({}, year_edges),   # {} = uniform
                    "month_of_year": ({}, month_edges),
                    "hour_of_day":   ({}, hour_edges),
                    "temperature":   ({20: 0.3, 25: 0.5, 30: 0.2}, temp_edges),
                }
            Any dimension name in config.DIMENSION_CATALOG is valid.
            An empty target dict {} means "uniform over all bins".

            Simple (backwards-compatible, temperature only):
                {15: 0.2, 20: 0.3, 25: 0.5}
        """
        self._distribution = DistributionTarget(
            target,
            valid_dimensions=set(DIMENSION_CATALOG.keys()),
        )
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

        # Histogram config — load bin edges for every dimension stored in the catalog.
        # Numeric dimensions (float edges) are stored as VARCHAR[] and parsed back to float.
        # String dimensions (e.g. quarterly timestamp labels) are kept as str.
        # The 'numeric' boolean column (added in the updated catalog schema) drives parsing.
        t1 = time.perf_counter()
        rows = conn.execute(
            "SELECT dataset_id, bin_edges, numeric FROM histogram_config"
        ).fetchall()
        for dim_name, edges_array, is_numeric in rows:
            if edges_array:
                self._bin_edges_dict[dim_name] = (
                    [float(v) for v in edges_array] if is_numeric
                    else list(edges_array)
                )
        log.debug(
            "catalog: histogram_config loaded — %d dimension(s): %s  (%.3fs)",
            len(self._bin_edges_dict),
            list(self._bin_edges_dict.keys()),
            time.perf_counter() - t1,
        )

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

    # ------------------------------------------------------------------
    # Raster cache helpers  (DuckDB, per layer)
    # ------------------------------------------------------------------
    # Cache file: /src/stream_cache/rasters/raster_cache.duckdb
    #
    # Schema
    # ------
    # raster_grid (one row per unique grid configuration):
    #   grid_key    VARCHAR PK   — 16-char SHA-256 of grid geometry params
    #   resolution_m DOUBLE
    #   lon0, lat0, step_lon, step_lat  DOUBLE
    #   n_lon, n_lat  INTEGER
    #
    # raster_layer (one row per computed layer):
    #   grid_key    VARCHAR      — FK → raster_grid
    #   layer_key   VARCHAR      — e.g. "11100:2006" or "wis:rijbaan"
    #   radius_m    DOUBLE       — query radius used for this layer
    #   n_nonzero   INTEGER      — non-zero cell count (for logging)
    #   array_blob  BLOB         — raw float32 bytes, shape (n_lon * n_lat,)
    #   PRIMARY KEY (grid_key, layer_key)
    #
    # A layer is identified solely by (grid_key, layer_key).  The grid_key
    # encodes resolution + bbox + derived grid geometry; the layer_key encodes
    # luc_code + survey year (or WIS attribute value).  radius_m is stored
    # separately because the same polygon class can be queried at different
    # radii — each (grid_key, layer_key) pair therefore corresponds to exactly
    # one (class, radius, grid) triple.
    #
    # This means a layer computed for one feature grouping (e.g.
    # classifications_fractions at r=100m) is reusable when a second run
    # requests the same class at the same radius — even if the overall layer
    # list differs.

    @property
    def _cache_db_path(self) -> Path:
        return self._raster_cache_dir / "raster_cache.duckdb"

    def _open_cache_db(self) -> "duckdb.DuckDBPyConnection":
        """
        Open (creating if necessary) the raster cache DuckDB and ensure
        the schema exists.  Returns an open read-write connection.
        """
        self._raster_cache_dir.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(self._cache_db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raster_grid (
                grid_key     VARCHAR PRIMARY KEY,
                resolution_m DOUBLE  NOT NULL,
                lon0         DOUBLE  NOT NULL,
                lat0         DOUBLE  NOT NULL,
                step_lon     DOUBLE  NOT NULL,
                step_lat     DOUBLE  NOT NULL,
                n_lon        INTEGER NOT NULL,
                n_lat        INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raster_layer (
                grid_key   VARCHAR NOT NULL,
                layer_key  VARCHAR NOT NULL,
                radius_m   DOUBLE  NOT NULL,
                n_nonzero  INTEGER NOT NULL,
                array_blob BLOB    NOT NULL,
                PRIMARY KEY (grid_key, layer_key)
            )
        """)
        return conn

    def _grid_key(
        self,
        lon0: float, lat0: float,
        step_lon: float, step_lat: float,
        n_lon: int, n_lat: int,
        resolution_m: float,
    ) -> str:
        """
        16-char SHA-256 of the grid geometry parameters.  Two grids with
        identical geometry produce the same key regardless of what layers
        are requested — this is what allows per-layer reuse across runs.
        """
        spec = {
            "resolution_m": resolution_m,
            "lon0":     round(lon0,     10),
            "lat0":     round(lat0,     10),
            "step_lon": round(step_lon, 10),
            "step_lat": round(step_lat, 10),
            "n_lon":    n_lon,
            "n_lat":    n_lat,
        }
        return hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _load_layer_from_cache(
        self,
        conn: "duckdb.DuckDBPyConnection",
        grid_key: str,
        layer_key: str,
        n_lon: int,
        n_lat: int,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Try to load one layer from the cache DB.

        Returns (array, n_nonzero) on hit, None on miss or any error.
        The array is always shape (n_lon, n_lat) float32.
        """
        try:
            row = conn.execute(
                "SELECT array_blob, n_nonzero FROM raster_layer "
                "WHERE grid_key = ? AND layer_key = ?",
                [grid_key, layer_key],
            ).fetchone()
            if row is None:
                return None
            blob, n_nonzero = row
            arr = np.frombuffer(bytes(blob), dtype=np.float32).reshape(n_lon, n_lat).copy()
            return arr, int(n_nonzero)
        except Exception as exc:
            log.warning("raster cache: load error for %s/%s: %s",
                        grid_key, layer_key, exc)
            return None

    def _save_layer_to_cache(
        self,
        conn: "duckdb.DuckDBPyConnection",
        grid_key: str,
        layer_key: str,
        radius_m: float,
        arr: np.ndarray,
        n_nonzero: int,
        lon0: float, lat0: float,
        step_lon: float, step_lat: float,
        n_lon: int, n_lat: int,
        resolution_m: float,
    ) -> None:
        """
        Persist one layer to the cache DB immediately after rasterisation.
        Upserts both the grid row (idempotent) and the layer row.
        """
        try:
            # Ensure grid row exists
            conn.execute("""
                INSERT OR IGNORE INTO raster_grid
                    (grid_key, resolution_m, lon0, lat0, step_lon, step_lat, n_lon, n_lat)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [grid_key, resolution_m, lon0, lat0, step_lon, step_lat, n_lon, n_lat])

            blob = arr.astype(np.float32).tobytes()
            conn.execute("""
                INSERT OR REPLACE INTO raster_layer
                    (grid_key, layer_key, radius_m, n_nonzero, array_blob)
                VALUES (?, ?, ?, ?, ?)
            """, [grid_key, layer_key, radius_m, n_nonzero, blob])
        except Exception as exc:
            log.warning("raster cache: save error for %s/%s: %s",
                        grid_key, layer_key, exc)

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

        # ---- Step 4: open cache DB and compute the grid key ----
        gkey = self._grid_key(
            lon0, lat0, step_lon, step_lat, n_lon, n_lat, res_m,
        )
        log.info("raster cache: grid_key=%s  db=%s", gkey, self._cache_db_path)

        cache_conn = None
        try:
            cache_conn = self._open_cache_db()
        except Exception as exc:
            log.warning("raster cache: could not open cache DB (%s) "
                        "— will compute without caching", exc)

        # ---- Step 5: open SpatiaLite for polygon fetching ----
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

        # Idempotent B-tree indexes on attribute columns used by raster
        # precompute.  Without these, COUNT/SELECT WHERE bestemming=? does a
        # full-table scan and a single dense WIS class can take >40 minutes.
        for stmt in (
            "CREATE INDEX IF NOT EXISTS idx_wis_bestemming     ON wis(bestemming)",
            "CREATE INDEX IF NOT EXISTS idx_wis_materiaalsoort ON wis(materiaalsoort)",
            "CREATE INDEX IF NOT EXISTS idx_ua_luc_code        ON urban_atlas(luc_code)",
            "CREATE INDEX IF NOT EXISTS idx_ua_year            ON urban_atlas(ua_year)",
        ):
            try:
                db.execute(stmt)
            except sqlite3.OperationalError as exc:
                log.debug("raster: index ensure skipped (%s): %s", stmt, exc)
        db.commit()

        total_cells    = n_lon * n_lat
        t_raster_start = time.perf_counter()
        n_cache_hits   = 0
        n_computed     = 0

        log.info(
            "raster: %d UA layer(s) + %d WIS layer(s), "
            "%d x %d = %d grid cells at %.0f m resolution",
            len(ua_layer_list), len(wis_layer_list),
            n_lon, n_lat, total_cells, res_m,
        )

        # ---- Step 6: per-layer cache-check → rasterise → save ----
        n_total_layers = len(all_layers)
        with tqdm(total=n_total_layers, desc="Raster layers",
                  unit="layer", position=0, dynamic_ncols=True) as layer_bar:
            for layer_idx, (layer_type, spec) in enumerate(all_layers, start=1):
                tag = f"[{layer_idx}/{n_total_layers}]"
                if layer_type == "ua":
                    luc_code, layer_radius_m, year = spec
                    layer_key = f"{luc_code}:{year}:r{int(layer_radius_m)}m"
                elif layer_type == "wis":
                    attr_col, attr_val, layer_radius_m = spec
                    layer_key = f"wis:{attr_val}:r{int(layer_radius_m)}m"
                else:
                    layer_bar.update(1)
                    continue

                # Prefix the cache key with the rasteriser variant so that
                # switching raster_fft forces a recompute rather than a
                # silent hit from an entry built with the other path.
                cache_layer_key = ("fft:" if self.raster_fft else "vec:") + layer_key

                # --- cache hit? ---
                if cache_conn is not None:
                    t_cache_lookup = time.perf_counter()
                    hit = self._load_layer_from_cache(
                        cache_conn, gkey, cache_layer_key, n_lon, n_lat,
                    )
                    t_cache_lookup = time.perf_counter() - t_cache_lookup
                    if hit is not None:
                        arr, n_nonzero = hit
                        raster._layers[layer_key]   = arr
                        raster._coverage[layer_key] = n_nonzero
                        n_cache_hits += 1
                        log.debug("raster %s cache HIT  %s  (%d non-zero, %.2fs)",
                                 tag, cache_layer_key, n_nonzero, t_cache_lookup)
                        layer_bar.set_postfix(
                            hit=n_cache_hits, computed=n_computed, refresh=False
                        )
                        layer_bar.update(1)
                        continue
                    else:
                        log.debug("raster %s cache MISS %s (lookup %.2fs) — computing",
                                  tag, cache_layer_key, t_cache_lookup)

                # --- cache miss: rasterise ---
                arr = raster.add_layer(layer_key)
                t_layer = time.perf_counter()

                dlat_r = layer_radius_m * _LAT_DEG_PER_M
                dlon_r = layer_radius_m * _LON_DEG_PER_M

                t_fetch = time.perf_counter()
                if layer_type == "ua":
                    all_blobs = _ua_fetch_all_in_bbox(
                        db, luc_code, year,
                        raster.grid_lon(0)       - dlon_r,
                        raster.grid_lat(0)       - dlat_r,
                        raster.grid_lon(n_lon-1) + dlon_r,
                        raster.grid_lat(n_lat-1) + dlat_r,
                    )
                    tqdm_desc = f"  UA {luc_code} {year} r={int(layer_radius_m)}m"
                else:
                    all_blobs = _wis_fetch_all_in_bbox(
                        db, attr_col, attr_val,
                        raster.grid_lon(0)       - dlon_r,
                        raster.grid_lat(0)       - dlat_r,
                        raster.grid_lon(n_lon-1) + dlon_r,
                        raster.grid_lat(n_lat-1) + dlat_r,
                    )
                    tqdm_desc = f"  WIS {attr_val} r={int(layer_radius_m)}m"
                t_fetch = time.perf_counter() - t_fetch

                total_blob_bytes = sum(len(b) for b in all_blobs)
                log.debug(
                    "raster %s %s r=%dm: fetched %d polygon(s) (%.1f MB WKB) in %.2fs",
                    tag, layer_key, int(layer_radius_m),
                    len(all_blobs), total_blob_bytes / 1e6, t_fetch,
                )

                t_rast = time.perf_counter()
                _rasterise = (
                    _rasterise_layer_fft if self.raster_fft else _rasterise_layer
                )
                n_nonzero = _rasterise(
                    arr, raster, all_blobs, layer_radius_m,
                    desc=tqdm_desc,
                )
                t_rast = time.perf_counter() - t_rast
                raster._coverage[layer_key] = n_nonzero
                elapsed = time.perf_counter() - t_layer
                n_computed += 1
                pct_nonzero = 100.0 * n_nonzero / total_cells if total_cells else 0
                log.info(
                    "raster %s %s done: %d non-zero cells (%.1f%%), "
                    "fetch %.2fs, rasterise %.2fs, total %.2fs (%.0f cells/s)",
                    tag, layer_key, n_nonzero, pct_nonzero,
                    t_fetch, t_rast, elapsed,
                    total_cells / elapsed if elapsed > 0 else 0,
                )

                # --- write to cache immediately ---
                if cache_conn is not None:
                    t_cache_save = time.perf_counter()
                    self._save_layer_to_cache(
                        cache_conn, gkey, cache_layer_key, layer_radius_m,
                        arr, n_nonzero,
                        lon0, lat0, step_lon, step_lat, n_lon, n_lat, res_m,
                    )
                    t_cache_save = time.perf_counter() - t_cache_save
                    log.debug("raster %s %s cached in %.2fs",
                              tag, cache_layer_key, t_cache_save)

                layer_bar.set_postfix(
                    hit=n_cache_hits, computed=n_computed, refresh=False
                )
                layer_bar.update(1)

        db.close()
        if cache_conn is not None:
            cache_conn.close()

        total_elapsed = time.perf_counter() - t_raster_start
        log.info(
            "raster: done — %d hit(s) from cache, %d computed, "
            "%d total cells, %.1fs",
            n_cache_hits, n_computed, total_cells, total_elapsed,
        )
        raster.log_summary()

        # ---- Step 7: wire raster into all descriptor _raster_ref cells ----
        for desc in raster_descs:
            desc._raster_ref[0] = raster
            log.debug("raster: wired into descriptor '%s'", desc.name)

        return raster

    def _select_partitions(self) -> List[Tuple[str, float]]:
        """
        Return list of (partition_key, weight) sorted by weight descending.

        When no distribution is set all partitions are returned with weight 1.0.

        When a distribution is set the method:
        1.  Builds a single DuckDB query that SELECTs the histogram arrays for
            every requested dimension from partition_statistics.
        2.  Computes a per-partition weight as the product of per-dimension
            overlap scores (sum of min(actual, target) per bin).
        3.  Returns partitions ordered by weight descending so that the
            highest-value data is streamed first.

        The query is constructed data-driven from DIMENSION_CATALOG so adding
        a new dimension requires no changes here — only entries in config.py
        and catalog.py.

        Weight expression (per dimension)
        ----------------------------------
        For a histogram array `counts` and a target vector `T` (same length):

            weight = SUM_i  min(counts[i] / total, T[i])

        This is the standard histogram-overlap metric: 1.0 when the partition
        matches the target perfectly, 0.0 when there is no overlap at all.
        The overall partition weight is the product over all active dimensions.
        """
        all_keys = getattr(self, "_partition_keys", [])

        if self._partitions:
            pk_set   = set(self._partitions)
            all_keys = [pk for pk in all_keys if pk in pk_set]

        if not all_keys:
            return []

        if self._distribution is None or not self._distribution.dimensions:
            return [(pk, 1.0) for pk in sorted(all_keys)]

        dimensions_to_score = list(self._distribution.dimensions.keys())
        pk_filter           = ", ".join(f"'{pk}'" for pk in all_keys)

        # ----------------------------------------------------------------
        # Build the SQL query from DIMENSION_CATALOG in one pass.
        # Each active dimension contributes three things:
        #   • A SELECT column alias (histogram array cast to DOUBLE[])
        #   • A WHERE IS NOT NULL guard
        #   • A weight expression (list_aggregate over list_transform)
        # ----------------------------------------------------------------
        select_cols   = ["partition_key"]
        where_clauses = ["dataset_id = 'lst'", f"partition_key IN ({pk_filter})"]
        weight_exprs  = []

        for dim in dimensions_to_score:
            dim_meta = DIMENSION_CATALOG.get(dim)
            if dim_meta is None:
                log.warning("_select_partitions: unknown dimension '%s' — skipping", dim)
                continue

            db_col    = dim_meta["col"]        # column name in partition_statistics
            sql_alias = dim_meta["sql_alias"]  # short alias for this query

            # SELECT: cast the stored BIGINT[] to DOUBLE[] for arithmetic
            select_cols.append(f"CAST({db_col} AS DOUBLE[]) AS {sql_alias}")

            # WHERE: skip partitions where this histogram is NULL
            where_clauses.append(f"{db_col} IS NOT NULL")

            # Weight expression: proportion-overlap per bin, summed
            dim_target = self._distribution.dimensions[dim]
            bin_edges  = dim_target.bin_edges
            n_bins     = len(bin_edges) - 1

            # Build the dense target vector aligned to the histogram bins
            target_vec = [0.0] * n_bins
            for edge_val, desired_prop in dim_target.target.items():
                bin_idx = dim_target._find_bin(edge_val)
                if 0 <= bin_idx < n_bins:
                    target_vec[bin_idx] += desired_prop

            target_json = json.dumps(target_vec)

            weight_exprs.append(f"""
                list_aggregate(
                    list_transform(
                        generate_series(0, len({sql_alias}) - 1),
                        i -> CASE
                            WHEN list_sum({sql_alias}) = 0 THEN 0.0
                            ELSE least(
                                {sql_alias}[i + 1] / list_sum({sql_alias}),
                                ({target_json}::DOUBLE[])[i + 1]
                            )
                        END
                    ),
                    'sum'
                )""")

        if not weight_exprs:
            return [(pk, 1.0) for pk in sorted(all_keys)]

        # Build the product of per-dimension weight expressions.
        if len(weight_exprs) == 1:
            weight_expr = weight_exprs[0]
        else:
            weight_expr = " * ".join(f"({e})" for e in weight_exprs)

        where_clause = " AND ".join(where_clauses)

        # partition_statistics has one row per (partition_key, tile_id).
        # Before scoring we must sum histogram arrays across all tiles of a
        # partition; otherwise every tile is a separate result row and
        # ORDER BY weight DESC locks onto the single highest-tile-count month,
        # streaming it until max_rows is hit without ever visiting another month.
        #
        # Aggregation pattern (one CTE per dimension):
        #   UNNEST each tile's BIGINT[] into (partition_key, bin_idx, count),
        #   SUM counts by (partition_key, bin_idx),
        #   re-aggregate into a DOUBLE[] ordered by bin_idx.
        # The resulting arrays feed the weight_exprs unchanged.

        agg_ctes  = []   # one per dimension
        agg_joins = []   # JOIN onto the distinct partition_key list

        for dim in dimensions_to_score:
            dim_meta = DIMENSION_CATALOG.get(dim)
            if dim_meta is None:
                continue
            db_col    = dim_meta["col"]
            sql_alias = dim_meta["sql_alias"]
            cte_name  = f"_agg_{sql_alias}"

            agg_ctes.append(f"""
        {cte_name} AS (
            SELECT
                partition_key,
                array_agg(bin_sum::DOUBLE ORDER BY bin_idx) AS {sql_alias}
            FROM (
                SELECT
                    partition_key,
                    idx - 1            AS bin_idx,
                    SUM({db_col}[idx]) AS bin_sum
                FROM partition_statistics,
                     generate_series(1, len({db_col})) AS t(idx)
                WHERE dataset_id = 'lst'
                  AND partition_key IN ({pk_filter})
                  AND {db_col} IS NOT NULL
                GROUP BY partition_key, idx
            )
            GROUP BY partition_key
        )""")
            agg_joins.append(
                f"LEFT JOIN {cte_name} USING (partition_key)"
            )

        agg_cte_block = ",\n".join(agg_ctes)
        joins_block   = "\n            ".join(agg_joins)

        sql = f"""
            WITH
            _pks AS (
                SELECT DISTINCT partition_key
                FROM partition_statistics
                WHERE dataset_id = 'lst'
                  AND partition_key IN ({pk_filter})
            ),
            {agg_cte_block},
            _scored AS (
                SELECT _pks.partition_key, {weight_expr} AS weight
                FROM _pks
                {joins_block}
            )
            SELECT partition_key, weight
            FROM _scored
            WHERE weight > 0
            ORDER BY weight DESC
        """
        
        # Check how many partitions the catalog actually loaded
        log.info("_select_partitions: %d partition_keys, scoring dims: %s",
                len(all_keys), dimensions_to_score)

        # Check what columns actually exist in partition_statistics
        cols = self._catalog_conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'partition_statistics'"
        ).fetchall()
        log.info("_select_partitions: partition_statistics columns = %s",
                [c[0] for c in cols])

        t0 = time.perf_counter()
        rows = self._catalog_conn.execute(sql).fetchall()

        t0 = time.perf_counter()
        rows = self._catalog_conn.execute(sql).fetchall()
        log.info(
            "catalog: partition scoring done in %.3fs (%d partitions) "
            "dimensions: %s",
            time.perf_counter() - t0, len(rows), dimensions_to_score,
        )
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

        Early stopping and row quotas
        ------------------------------
        When max_rows is set without a distribution, rows are streamed
        sequentially and stopped once max_rows is reached.

        When max_rows is set WITH a distribution target, each partition
        receives a row_quota proportional to its weight:

            quota_i = floor(weight_i / sum(weights) * max_rows)

        Leftover rows (from floor rounding) are assigned to the
        highest-weight partitions first.  Each partition cursor is issued
        with "LIMIT quota_i" so it never reads more than its share.
        This is the mechanism that actually enforces the target distribution
        rather than just ordering partitions.

        Without max_rows the distribution only orders partitions; all rows
        from every partition are yielded in weight-descending order.
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

        # ---- Compute per-partition row quotas --------------------------------
        # When max_rows is set and a distribution is active, enforce the
        # distribution by limiting each partition to its proportional share.
        # Without a distribution (all weights == 1.0) fall back to the original
        # behaviour: stream sequentially and stop at max_rows.
        has_distribution = (
            self._distribution is not None
            and bool(self._distribution.dimensions)
        )
        if max_rows is not None and has_distribution:
            total_weight = sum(w for _, w in partitions)
            if total_weight <= 0:
                total_weight = 1.0
            # Floor allocation
            raw_quotas    = [int((w / total_weight) * max_rows) for _, w in partitions]
            allocated     = sum(raw_quotas)
            leftover      = max_rows - allocated
            # Distribute leftover to highest-weight partitions first
            order         = sorted(range(len(partitions)), key=lambda i: partitions[i][1], reverse=True)
            for i in range(leftover):
                raw_quotas[order[i % len(order)]] += 1
            # Map partition_key → quota; skip partitions with quota 0
            partition_quotas = {
                pk: q for (pk, _), q in zip(partitions, raw_quotas) if q > 0
            }
            log.info(
                "stream: quota mode — %d partitions, %d total allocated rows "
                "(target %d), non-zero: %d",
                len(partitions), sum(partition_quotas.values()),
                max_rows, len(partition_quotas),
            )
        else:
            # Sequential mode: no per-partition caps
            partition_quotas = None

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

                # Skip partitions that received zero quota allocation
                if partition_quotas is not None and partition_key not in partition_quotas:
                    continue

                part_t0   = time.perf_counter()
                part_rows = 0

                # Row cap for this partition:
                #   quota mode → fixed allocation
                #   sequential mode → remainder of global max_rows budget
                if partition_quotas is not None:
                    part_limit = partition_quotas[partition_key]
                elif max_rows is not None:
                    part_limit = max_rows - session_rows
                    if part_limit <= 0:
                        stop_early = True
                        break
                else:
                    part_limit = None   # no cap — read the full partition

                part_bar.set_postfix({
                    "pk":      partition_key,
                    "weight":  f"{weight:.3f}",
                    "quota":   f"{part_limit:,}" if part_limit is not None else "all",
                    "yielded": f"{session_rows:,}",
                }, refresh=True)

                log.debug("partition %s [%d/%d] (w=%.3f, limit=%s): opening cursor",
                          partition_key, part_idx + 1, len(partitions), weight,
                          part_limit if part_limit is not None else "∞")

                # Build the cursor SQL — add LIMIT when a cap is set so DuckDB
                # reads only the required rows from disk.
                limit_sql = f" LIMIT {part_limit}" if part_limit is not None else ""
                cursor = lst_conn.execute(
                    "SELECT longitude, latitude, aster_lst, modis_lst, ndvi, "
                    "       image_id, timestamp, partition_key, tile_id, "
                    "       year, month_of_year, day_of_month, day_of_year, hour_of_day "
                    f"FROM lst WHERE partition_key = ?{limit_sql}",
                    [partition_key],
                )
                log.debug("partition %s: cursor ready in %.3fs",
                          partition_key, time.perf_counter() - part_t0)

                batch_num = 0
                first_batch_feat_time: Optional[float] = None

                while True:
                    # In quota mode the LIMIT on the cursor already caps how
                    # many rows this partition contributes; we just read until
                    # the cursor is exhausted.  In sequential mode we still
                    # need to honour the global max_rows budget.
                    if partition_quotas is None and max_rows is not None:
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
                            log.debug("partition %s batch %d: first feature batch "
                                        "in %.2fs - cols: %s",
                                        partition_key, batch_num, feat_elapsed,
                                        [c for c in out_cols_data
                                        if c not in FeatureRow._fields])
                        


                        try:
                            result_df = pd.DataFrame(out_cols_data)
                        except:
                            log.error("partition %s batch %d: error creating result DataFrame",
                                        partition_key, batch_num, exc_info=True)
                            log.error(f"out_cols_data={out_cols_data}")
                            raise
                        yield result_df
                        n_yielded = len(result_df)

                    session_rows += n_yielded
                    part_rows    += n_yielded
                    batch_num    += 1

                    elapsed    = time.perf_counter() - session_t0
                    throughput = session_rows / elapsed if elapsed > 0 else 0

                    part_bar.set_postfix({
                        "pk":      partition_key,
                        "weight":  f"{weight:.3f}",
                        "yielded": f"{session_rows:,}",
                        "r/s":     f"{throughput:,.0f}",
                        "feat_s":  f"{first_batch_feat_time:.1f}s"
                                    if first_batch_feat_time else "...",
                    }, refresh=False)

                    # In sequential mode (no quotas), stop as soon as the
                    # global max_rows budget is exhausted.
                    # In quota mode the LIMIT on the cursor handles capping;
                    # stop_early is only set if somehow session_rows overshoots.
                    if max_rows is not None and session_rows >= max_rows:
                        stop_early = True
                        break

                part_elapsed = time.perf_counter() - part_t0
                part_rps     = part_rows / part_elapsed if part_elapsed > 0 else 0
                log.debug("partition %s: done - %d rows in %.1fs (%.0f rows/s)",
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