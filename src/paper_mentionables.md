# Paper Mentionables — Technologies & Strategies

Collected from `/processing/` and `/stream/`. Each entry notes what is used, how it is applied, and why that choice was made.

---

## Data Storage & Querying

### DuckDB (`lst.duckdb`, `dhm.duckdb`, `trees.duckdb`, `catalog.duckdb`)
- **What**: Embedded analytical database (columnar, OLAP-optimised).
- **How**: All point-data tables (LST pixels, DHM elevation points, tree inventory, catalog histograms) are stored as DuckDB files. At ingest time rows are appended in batches; at stream time a read-only cursor issues `fetchmany()` so the full 725 M-row LST table is never loaded into memory.
- **Why**: Columnar storage + zone-map pruning on sorted columns makes per-partition scans (e.g. `WHERE partition_key = ?`) extremely fast without a traditional row-level index. DuckDB also allows arbitrary analytical SQL including array functions (`list_aggregate`, `list_transform`, `generate_series`) used by the distribution-scoring query in `_select_partitions`.

### SpatiaLite (`spatial.db` via `mod_spatialite`)
- **What**: SQLite extension providing spatial data types, R-tree spatial indexes, and geometry functions (WKB, `BuildMbr`, `SpatialIndex`).
- **How**: Urban Atlas land-use polygons (four survey years: 2006, 2012, 2018, 2021) and WIS road-surface polygons are stored in SpatiaLite. Spatial queries use the built-in R-tree (`SELECT id FROM SpatialIndex WHERE f_table_name = '...' AND search_frame = BuildMbr(...)`) as a bounding-box pre-filter, then exact intersection is computed in Shapely/GEOS outside SQLite.
- **Why**: SpatiaLite R-tree pre-filters avoid loading the full polygon table into Python for every query. Separating the GEOS computation to Shapely avoids the SQLite extension overhead for complex intersection math and allows WKB caching.

### Idempotent B-tree Indexes at Stream Start
- **What**: `CREATE INDEX IF NOT EXISTS` statements on `wis.bestemming`, `wis.materiaalsoort`, `urban_atlas.luc_code`, `urban_atlas.ua_year` are executed each time the stream opens the SpatiaLite file.
- **Why**: Without these, `WHERE bestemming = ?` during raster precomputation performs a full-table scan; a single dense WIS class (e.g. "Rijbaan") can take >40 minutes. The index creation is idempotent so it costs nothing if the index already exists.

---

## Spatial Indexing & Geometry

### H3 Hierarchical Spatial Index (Uber H3)
- **What**: Hexagonal global hierarchical indexing system. Resolution 9 used as the primary tile key (~0.17 km² per cell, ~174 m diameter).
- **How**: Every LST pixel is assigned a `tile_id` (H3 r9 cell) at ingest time via `h3.latlng_to_cell`. Coarser resolutions (r8 ~860 m, r7 ~2.3 km) are stored as additional columns. All feature batch queries use `tile_id` as a cache key so that pixels sharing a tile share one query result.
- **Why**: H3 cells cluster nearby pixels, allowing heavy spatial queries (radius aggregates, nearest-point lookups) to be deduplicated: one SQL query per unique `tile_id` instead of one per pixel. At resolution 9 all pixels in a cell are within ~200 m, which is within acceptable error for any feature radius >= 50 m.

### Rectangular Grid Tiles (Belgian Lambert 1972, EPSG:31370)
- **What**: Rectilinear 1 km x 1 km and 2 km x 2 km tiles computed in the Belgian national projection.
- **How**: Pixel coordinates are projected from WGS-84 to Belgian Lambert via `pyproj.Transformer` at ingest time; tile keys are `floor(x/size)_floor(y/size)` strings.
- **Why**: Alternative grouping keys for analyses that benefit from regular rectangular cells (e.g. zone-level aggregation compatible with Belgian statistical datasets).

### NGI Kaartbladversnijding Tile (Optional)
- **What**: Tile keys from the Belgian National Geographic Institute topographic map sheet grid.
- **How**: A spatial join (`geopandas.sjoin`) assigns each pixel to the NGI sheet polygon at ingest time.
- **Why**: Provides compatibility with NGI raster products (DHM, orthofotos) that are published per map sheet.

### Axis-Aligned Bounding Box (AABB) Pre-filter
- **What**: Point-in-polygon checks are preceded by an AABB test using the polygon's `.bounds`.
- **How**: In `filter_by_polygon`: bounding-box test filters out points clearly outside the polygon before the expensive Shapely `contains()` call.
- **Why**: Reduces the number of expensive GEOS predicate calls significantly for large rasters.

---

## Raster Processing

### Block-Streamed GeoTIFF Reading (`rasterio`)
- **What**: GeoTIFFs are read one internal block at a time via `rasterio.block_windows`.
- **How**: `iter_raster_blocks` and `iter_raster_blocks_masked` yield `pd.DataFrame` chunks; memory usage is bounded to one raster block regardless of file size. The masked variant uses `rasterio.mask.mask` (GDAL-level polygon clipping) instead of per-pixel Shapely checks.
- **Why**: The LST GeoTIFFs can be large; loading the full array and running a Python-level point-in-polygon check per pixel would be prohibitively slow and memory-intensive. GDAL-level masking is orders of magnitude faster.

### CRS Reprojection at Clip Time
- **What**: If a raster's CRS differs from WGS-84, the clip polygon is transformed to the raster's native CRS before masking.
- **How**: `pyproj.Transformer.from_crs("EPSG:4326", raster_crs)` + `shapely.ops.transform`.
- **Why**: Avoids reprojecting the raster itself (expensive and lossy); reprojecting the small clip polygon is cheaper and exact.

### Multi-Emissivity LST Schema (ASTER / MODIS / NDVI)
- **What**: Three separate emissivity-correction algorithms for Landsat-derived LST are stored as three nullable columns (`aster_lst`, `modis_lst`, `ndvi`). A configurable resolver selects or combines them at stream time.
- **How**: At ingest the three products are outer-joined on (longitude, latitude) for each acquisition scene. At stream time `_resolve_lst_temperature` applies a mode strategy: `"any"` (first non-null: ASTER -> MODIS -> NDVI), `"fallback"` (ASTER -> MODIS only), or single-product modes. `"impute"` null-handling backfills within ASTER/MODIS.
- **Why**: Different Landsat scenes may have only ASTER, only MODIS, or only NDVI coverage. Storing all three and resolving at query time preserves the full dataset while allowing the user to control which emissivity assumption is applied.

---

## Feature Engineering Framework

### 2x2 Spatiotemporal Feature Framework
- **What**: A factory system combining two spatial strategies (nearest point, radius aggregate) with two temporal strategies (last_previous, nearest) and a special polygon-fraction path.
- **How**: `nearest()`, `aggregate_in_radius()`, `urban_atlas_luc_fraction()`, `wis_fraction()`, `urban_atlas_classifications_fractions()` are factory functions returning `_FeatureDescriptor` objects registered in a `FeatureRegistry`. Each descriptor provides a `compute_batch()` method (bulk SQL) and a `compute_row()` method (per-row fallback / custom callables).
- **Why**: Separating the descriptor from the execution logic allows the streaming engine to choose the most efficient path: batch SQL for all registered features in a single loop, row-level for custom callables that cannot be expressed in SQL.

### Tile-Level Query Deduplication
- **What**: Batch spatial queries are issued once per unique `(tile_id, date)` key, not once per row.
- **How**: In `batch_nearest` and `batch_radius`, a `cache: Dict[str, result]` maps `tile_id:date` -> pre-computed result. Each batch iterates rows and assigns them the cached result for their tile.
- **Why**: Pixels in the same H3 r9 cell are within ~200 m; for any reasonable feature radius (50-500 m) their neighbourhood is identical. Deduplication reduces SQL round-trips by 100-1000x for typical batch sizes.

### Precomputed Polygon-Fraction Raster (`_PolyRaster`)
- **What**: A regular lon/lat grid (default 15 m spacing) of precomputed coverage fractions for every Urban Atlas LUC code and WIS road-surface class.
- **How**: Built once at the start of each `stream()` call, covering the fixed Ghent bounding box. At query time a lookup snaps (lon, lat) to the nearest grid point in O(1).
- **Why**: The slow Shapely path (R-tree pre-filter -> WKB decode -> polygon intersection) costs ~1-5 ms per tile. At 725 M rows this would dominate total runtime. Precomputing eliminates GEOS entirely from the batch hot path.

### Vector Rasteriser (`_rasterise_layer`)
- **What**: Fills one raster layer by computing covered-area/circle-area for every grid cell using a Shapely STRtree and PreparedGeometry.
- **How**: (1) Decode and optionally simplify polygons once; (2) build an STRtree; (3) pre-classify polygons as "large" (wrap in `PreparedGeometry`) or "small" (skip prep); (4) per-column: numpy bbox skip if no polygon overlaps; per-cell: STRtree bbox query -> column-set filter -> build translated ellipse only if candidates exist -> `contains` fast-path -> exact `intersection`; (5) early saturation exit when covered area reaches circle area.
- **Why**: Most cells in Ghent are either fully covered (urban fabric) or fully empty (parks, water). The early saturation exit and the per-column numpy skip together avoid the majority of GEOS calls.

### FFT Convolution Rasteriser (`_rasterise_layer_fft`)
- **What**: Fills one raster layer via 2-D FFT convolution of a high-resolution binary polygon mask with a fractional-coverage disk kernel.
- **How**: (1) Rasterise all polygons into a binary mask at `supersample x output` resolution (default 4x, ~3.75 m sub-pixels) using `rasterio.features.rasterize`; (2) build an analytical disk kernel (4x4 sub-sampling per kernel cell for accuracy); (3) convolve with `scipy.signal.oaconvolve` (overlap-add, memory-bounded); (4) sample at output grid points (integer stride lookup, no interpolation).
- **Why**: Used for dense polygon classes such as WIS road surfaces, where every output cell sees hundreds to thousands of overlapping polygons. The vector path cost grows as O(cells x candidates); FFT cost is O(N log N) in the mask size and is independent of polygon count. For WIS "Rijbaan" the FFT path completes in seconds versus hours for the vector path.
- **Accuracy**: At supersample=4 and radius=100 m, RMS fraction error is ~0.3%.

### Shapely Polygon Simplification Before Rasterisation
- **What**: Each decoded polygon is simplified with `simplify(0.00005, preserve_topology=True)` (~5 m tolerance) before being added to the STRtree.
- **Why**: Detailed road polygons (UA class 12220) can have thousands of vertices; simplification reduces vertex count while keeping fraction error well below 1%, roughly halving GEOS intersection cost on complex geometries.

### Raster Cache (DuckDB, per-layer content-addressed blobs)
- **What**: A persistent DuckDB database (`stream_cache/rasters/raster_cache.duckdb`) storing precomputed raster layers as raw float32 blobs, keyed by a 16-char SHA-256 of grid geometry parameters plus a layer key.
- **How**: On stream open, each required layer is checked in the cache. Hit -> deserialise blob and skip rasterisation. Miss -> rasterise -> serialise float32 array to `bytes()` -> insert with `INSERT OR REPLACE`.
- **Why**: Rasterisation of all layers for a typical Ghent run takes 10-30 minutes. Caching eliminates this cost on subsequent runs with the same grid and feature set. The content-addressed key ensures that changing the resolution or bounding box invalidates the cache correctly without requiring manual clearing.

---

## Partition Statistics & Distribution Sampling

### Per-Partition Histogram Catalog (`catalog.duckdb`)
- **What**: For every `(dataset_id, partition_key, tile_id)` combination, the catalog stores histogram count arrays for all registered dimensions (temperature, year, month, day-of-year, hour, longitude, latitude, and a quarterly timestamp label).
- **How**: Written by `write_catalog()` at ingest time. Loaded once at stream start. Consumed by `_select_partitions` to score partitions for distribution-targeted streaming.
- **Why**: Eliminates the need to scan LST rows to determine their value distribution; per-partition histograms allow the streaming engine to score and rank partitions entirely from metadata.

### Histogram-Overlap Partition Scoring
- **What**: A DuckDB query computes a per-partition weight as the product of per-dimension histogram-overlap scores: `SUM_i min(counts[i]/total, target[i])`.
- **How**: A dynamically-constructed SQL query (built from `DIMENSION_CATALOG`) uses `list_transform`, `list_aggregate`, and `generate_series` to compute the overlap in-database. Per-tile histogram arrays are first summed across all tiles of a partition (via unnest CTEs) so the scoring is per-partition, not per-tile.
- **Why**: The histogram-overlap metric (standard in distribution matching) gives a score of 1.0 when the partition perfectly matches the target and 0.0 when there is no overlap. Scoring in-database avoids transferring large arrays to Python and scales to multiple dimensions.

### Proportional Row-Quota Allocation
- **What**: When `max_rows` is set alongside a distribution target, each partition receives a row quota proportional to its overlap weight; remainders are distributed to the highest-weight partitions first.
- **How**: `cursor.fetchmany(quota)` with a `LIMIT quota` SQL clause so DuckDB reads only the required rows from disk.
- **Why**: Enforces the target distribution at the data-read level rather than post-hoc sampling, avoiding unnecessary I/O.

### DIMENSION_CATALOG as Single Source of Truth
- **What**: A module-level `OrderedDict` in `config.py` that maps each dimension name to its column name in `partition_statistics`, its bin edges, its `numeric` flag, and a SQL alias.
- **How**: `catalog.py`, `stream.py`, and `distribution.py` all iterate this dict. Adding a new dimension requires a single entry here.
- **Why**: Avoids fragile ad-hoc column enumerations scattered across modules; the config drives schema, histogramming, scoring SQL generation, and validation from one place.

---

## Temporal Handling

### Pre-computed Time Components at Ingest
- **What**: Five integer/float columns derived from the acquisition timestamp are stored in the LST table: `year`, `month_of_year`, `day_of_month`, `day_of_year`, `hour_of_day`.
- **Why**: Avoids string-parsing at query time during catalog histogramming and streaming. Each component is independently targetable as a sampling dimension.

### `last_previous` Temporal Join
- **What**: When querying a non-static dataset (e.g. DHM which has two survey years), `temporal="last_previous"` selects the most recent observation with `timestamp <= driving_ts`.
- **How**: Appended as `AND timestamp <= '...' ORDER BY timestamp DESC` in SQL; equivalent logic applied in Python for the SpatiaLite path.
- **Why**: Correct for slowly-evolving data (DHM, Urban Atlas) where the most recent prior survey is the best available ground truth for a given LST observation.

### Urban Atlas Multi-Year Last-Previous Raster Lookup
- **What**: UA polygons exist for four survey years. When `ua_year=None`, the raster precompute builds layers for all four years; at query time `lookup_ua_last_previous()` picks the layer whose year is the last survey year <= the LST row's year.
- **Why**: Matches the temporal join semantics of `last_previous` without re-querying SpatiaLite per row.

---

## Causal Inference Support (Difference-in-Differences)

### `aanlegjaar_lte_scene` Treatment Dose Filter
- **What**: A flag on `aggregate_in_radius` that restricts tree-count queries to trees whose planting year (`aanlegjaar`) is known and <= the LST scene's year.
- **How**: Appended as `AND aanlegjaar IS NOT NULL AND aanlegjaar <= {scene_year}` in SQL. Cache key includes the scene year (`tile_id:ay{YYYY}`) so that the count correctly varies over time.
- **Why**: This is the staggered Difference-in-Differences (DiD) treatment dose: the count of trees already planted at observation time, excluding the ~73% of trees with unknown planting date. It provides the temporal variation in treatment intensity needed to identify the causal effect of greening on LST.

### `trees_count_planted_by` Factory
- **What**: Convenience factory wrapping `aggregate_in_radius` with `aanlegjaar_lte_scene=True`.
- **Why**: Makes the DiD treatment-dose feature declarative and self-documenting at the registry level.

---

## Coordinate Reference Systems

### WGS-84 (EPSG:4326) as the pipeline working CRS
All coordinates are stored and queried in WGS-84 decimal degrees throughout the pipeline. CRS conversion from source datasets happens once at ingest.

### Belgian Lambert 1972 (EPSG:31370)
Used for rectangular tile computation and DHM source data. `pyproj.Transformer` objects are created once at module import and reused.

### EPSG:3035 (LAEA Europe)
Listed in config as an additional supported CRS for source datasets.

### Degree-Space Circle Approximation
- **What**: Spatial queries use an ellipse in WGS-84 degree-space rather than a true metric circle.
- **How**: `_ua_make_circle` builds a unit circle, scales x by `radius_m * _LON_DEG_PER_M` and y by `radius_m * _LAT_DEG_PER_M` (where `_LON_DEG_PER_M` is latitude-corrected for Ghent), then translates.
- **Why**: Avoids reprojecting geometry to a metric CRS per query. Since both the polygon coverage numerator and circle area denominator use the same degree-space ellipse, the approximation cancels and the fraction is accurate.

### Haversine Distance
- **What**: Exact great-circle distance used for nearest-point selection and radius filtering in the SpatiaLite/Python path.
- **How**: Vectorised NumPy implementation of the haversine formula in `geo.py`.
- **Why**: The flat-Earth approximation (used in the DuckDB path as `sqrt(pow(dlon * 111320, 2) + pow(dlat * 111320 * cos(lat), 2))`) is sufficient for the bounding-box pre-filter but haversine is used for exact selection in the SpatiaLite path where precision matters.

---

## Streaming Architecture

### Generator-Based Streaming
- **What**: `StreamConfig.stream()` is a Python generator that yields `pd.DataFrame` batches.
- **How**: `cursor.fetchmany(batch_size)` pulls rows partition-by-partition; feature computation is applied in-place before yielding. The caller controls memory by consuming one batch at a time.
- **Why**: The 725 M-row LST table cannot fit in RAM; generator streaming allows model training via `model.partial_fit(batch)` without ever materialising the full dataset.

### Dual Feature-Computation Path (Batch vs Row)
- **What**: Framework features (nearest, radius, polygon-fraction) use a bulk SQL path (`compute_batch`); custom callables use a per-row path (`compute_row`).
- **How**: `FeatureRegistry` maintains separate `_batch_descriptors` and `_row_descriptors` lists. The stream loop calls `compute_batch_features` first (one SQL round-trip per descriptor), then `compute_row_features` (one Python call per row per descriptor).
- **Why**: Custom callables cannot be expressed as a single SQL JOIN, but framework features can be. Separating the paths avoids forcing all features into the slower per-row path.

### Pre-allocated Column Arrays for Row Features
- **What**: Row-level feature results are accumulated into pre-allocated `[None] * n` Python lists, not a list-of-dicts.
- **Why**: Avoids the overhead of building thousands of small dicts and merging them; pre-allocation gives predictable memory usage and allows a single `pd.DataFrame(col_arrays)` construction at batch end.

### SpatiaLite Shared Connection with Pre-init
- **What**: The `Connections` object holds one SpatiaLite connection per database file, opened once and reused across all batches.
- **How**: `feature_conns.spatialite("spatial.db")` is called before the batch loop begins, initialising the `mod_spatialite` extension. Subsequent calls return the cached connection.
- **Why**: Loading `mod_spatialite` (GEOS, Proj) has significant startup cost; re-opening per batch would dominate runtime.

### WKB Geometry Cache
- **What**: A module-level dict in `poly_raster.py` caches decoded Shapely geometries keyed by raw WKB bytes.
- **Why**: The same polygon blob may be encountered in multiple batch queries (e.g. a large park polygon appears in many neighbouring tiles). Caching avoids re-decoding the WKB and re-constructing the GEOS object.

---

## Performance & Scalability

### Zone-Map Pruning via Physical Sort Order
- **What**: The LST table is sorted by `(partition_key, tile_id)` as the final step of ingest.
- **How**: `CREATE TABLE lst_sorted AS SELECT * FROM lst ORDER BY partition_key, tile_id` then rename.
- **Why**: DuckDB uses zone-map (min/max metadata per row group) to skip row groups during filtered scans. Sorting by partition_key means all rows of a given month are contiguous; `WHERE partition_key = ?` skips the vast majority of row groups without reading them.

### `SET preserve_insertion_order = false` During Ingest
- **What**: DuckDB setting that allows out-of-order internal writes, trading insertion-order semantics for write throughput.
- **Why**: During ingest, insertion order is irrelevant (a global sort is applied at the end); disabling the guarantee removes a write bottleneck.

### Incremental Batch Flushing + Explicit Checkpointing
- **What**: Accumulated rows are flushed to DuckDB in batches of `APPEND_BATCH_ROWS` (1 M rows). An explicit `CHECKPOINT` is issued every 50 image-ID groups.
- **Why**: Prevents unbounded memory growth during ingest of large datasets; checkpointing ensures data is flushed to disk and reduces write-ahead log size.

### tqdm Progress Bars (Nested, Position-Controlled)
- **What**: `tqdm` progress bars at multiple nesting levels: partition loop (position=0), raster column loop (position=1), polygon fetch/decode (position=2).
- **Why**: Long-running rasterisation (hours for dense polygon classes) without progress feedback is unusable; nested position-controlled bars prevent interleaved output.

### Heartbeat Logging During Rasterisation
- **What**: A time-based log message is emitted every 5 seconds during the column rasterisation loop, reporting progress, ETA, and intersection statistics.
- **Why**: tqdm updates only on column completion; a single slow column (e.g. a dense urban area) can stall the bar for minutes. The heartbeat provides continuous feedback.

---

## Software Dependencies (paper-relevant)

| Library | Role |
|---|---|
| `duckdb` | Analytical embedded DB for LST, DHM, trees, catalog |
| `rasterio` | GeoTIFF reading, polygon masking, binary rasterisation |
| `geopandas` | NGI shapefile loading and spatial join |
| `shapely` | Polygon geometry, intersection, STRtree, WKB decode |
| `pyproj` | CRS transformations (WGS-84 / Lambert / LAEA) |
| `h3` | Hierarchical hexagonal spatial indexing |
| `scipy` | FFT convolution (`oaconvolve`) for dense rasterisation |
| `numpy` | Vectorised array ops throughout |
| `pandas` | Batch DataFrames |
| `tqdm` | Progress bars |
| `sqlite3` + `mod_spatialite` | SpatiaLite polygon storage and R-tree indexing |
