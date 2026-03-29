"""
Stream LST data with dynamically computed engineered features.

This module provides:
1. LSTStream: Generator that yields LST rows with appended features
2. FeatureTransformer: Extensible feature engineering pipeline
3. Built-in feature functions for common use cases

Uses memory-efficient DuckDB and Zarr backends to handle large datasets on limited RAM.

Usage:
    from lst_stream import LSTStream, register_feature
    
    # Register custom features
    @register_feature("my_feature", depends_on=["longitude", "latitude"])
    def compute_my_feature(row, context):
        return some_value
    
    # Create stream
    stream = LSTStream(batch_size=10000)
    for batch_df in stream.stream_batches():
        # Process batch with all computed features
        pass
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Generator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import duckdb
import zarr
import json
from tqdm import tqdm
from shapely.geometry import Point, Polygon


@dataclass
class FeatureSpec:
    """Specification for a feature in the pipeline."""
    name: str
    compute_fn: Callable
    depends_on: List[str] = field(default_factory=list)
    cache: bool = False
    description: str = ""
    columns: List[str] = field(default_factory=list)  # Column names for multi-column features


class FeatureTransformer:
    """
    Extensible feature engineering pipeline.
    
    Features can depend on:
    - Input row data (longitude, latitude, timestamp, etc.)
    - Prepared indexed data from DuckDB (DHM, trees, urban atlas)
    - Previously computed features
    """
    
    def __init__(self, prepared_data_path: Optional[Path] = None):
        """
        Initialize feature transformer with DuckDB connection.
        
        Args:
            prepared_data_path: Path to prepared stream data (created by prepare_all_for_streams.py)
        """
        self.prepared_data_path = prepared_data_path or (
            Path(__file__).parent.parent / "prepared_stream_data"
        )
        self._features: Dict[str, FeatureSpec] = {}
        self._feature_cache: Dict[str, Dict] = {}
        
        # DuckDB connection for spatial queries
        self.db_path = self.prepared_data_path / "stream_index.duckdb"
        self.conn = duckdb.connect(str(self.db_path), read_only=True)
        
        # Zarr arrays for DHM data (loaded on-demand by chunk)
        self._zarr_stores: Dict[str, Any] = {}  # Mix of Zarr groups and metadata dicts
        self._load_prepared_data()
        
    def _load_prepared_data(self) -> None:
        """Load prepared data metadata and establish connections."""
        tqdm.write("Connecting to prepared data...")
        
        # Verify DuckDB database exists
        if not self.db_path.exists():
            tqdm.write(f"[ERROR] Database not found: {self.db_path}")
            tqdm.write("[ERROR] Run prepare_all_for_streams.py first")
            return
        
        # Load Zarr stores for DHM (only metadata, data loaded on demand)
        zarr_files = list(self.prepared_data_path.glob("dhm_*.zarr"))
        for zarr_path in tqdm(zarr_files, desc="Loading DHM Zarr files", disable=len(zarr_files)==0):
            source_name = zarr_path.name.replace("dhm_", "").replace(".zarr", "")
            try:
                # Load Zarr group
                zarr_group = zarr.open_group(str(zarr_path), mode='r')
                self._zarr_stores[source_name] = zarr_group
                
                # Load metadata from JSON file
                metadata_file = self.prepared_data_path / f"dhm_{source_name}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    # Store metadata in a dict for easy access
                    self._zarr_stores[f"{source_name}_meta"] = metadata
                
                tqdm.write(f"  Loaded DHM Zarr: {source_name}")
            except Exception as e:
                tqdm.write(f"  Could not load Zarr {source_name}: {e}")
        
        # Verify DuckDB tables
        try:
            tables = self.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'"
            ).fetchall()
            table_names = [t[0] for t in tables]
            tqdm.write(f"  DuckDB tables: {', '.join(table_names)}")
        except Exception as e:
            tqdm.write(f"  Could not list tables: {e}")
    
    def register_feature(
        self,
        name: str,
        compute_fn: Callable,
        depends_on: Optional[List[str]] = None,
        cache: bool = False,
        description: str = "",
        columns: Optional[List[str]] = None
    ) -> None:
        """
        Register a feature in the pipeline.
        
        Args:
            name: Feature name (base name for output columns)
            compute_fn: Function(row, context) -> value or list of values that computes the feature
                - For single-column features: returns scalar value
                - For multi-column features: returns list/tuple with len(columns) values
            depends_on: List of column names this feature depends on
            cache: Whether to cache results
            description: Human-readable description
            columns: List of column names for multi-column features (e.g., ['height_mean', 'height_std'])
                If None or empty, uses name as single column name
        """
        if depends_on is None:
            depends_on = []
        if columns is None:
            columns = [name]  # Single-column feature uses the feature name
        
        self._features[name] = FeatureSpec(
            name=name,
            compute_fn=compute_fn,
            depends_on=depends_on,
            cache=cache,
            description=description,
            columns=columns
        )
        col_str = f" -> {columns}" if len(columns) > 1 else f""
        tqdm.write(f"  Registered feature: {name}{col_str}")
    
    def compute_features(
        self,
        row: pd.Series,
        computed_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute all registered features for a row.
        
        Args:
            row: LST data row
            computed_features: Dict to accumulate computed features
            
        Returns:
            Dict of column_name -> computed_value (flattens multi-column features)
        """
        if computed_features is None:
            computed_features = {}
        
        context = {
            'conn': self.conn,
            'zarr_stores': self._zarr_stores,
            'row': row
        }
        
        for feature_name, feature_spec in self._features.items():
            # Skip if any of the output columns already computed
            if any(col in computed_features for col in feature_spec.columns):
                continue
            
            try:
                value = feature_spec.compute_fn(row, context)
                
                # Unpack multi-column features
                if len(feature_spec.columns) == 1:
                    # Single-column feature: value is scalar
                    computed_features[feature_spec.columns[0]] = value
                else:
                    # Multi-column feature: value should be list/tuple
                    if isinstance(value, (list, tuple)):
                        for col_name, col_value in zip(feature_spec.columns, value):
                            computed_features[col_name] = col_value
                    else:
                        # If not a list/tuple, treat as missing
                        for col_name in feature_spec.columns:
                            computed_features[col_name] = np.nan
            except Exception as e:
                tqdm.write(f"  Error computing {feature_name}: {e}")
                for col_name in feature_spec.columns:
                    computed_features[col_name] = np.nan
        
        return computed_features
    
    def get_feature_info(self) -> pd.DataFrame:
        """Get information about all registered features."""
        info = []
        for name, spec in self._features.items():
            info.append({
                'feature': name,
                'description': spec.description,
                'depends_on': ', '.join(spec.depends_on) if spec.depends_on else 'none',
                'cached': spec.cache
            })
        return pd.DataFrame(info)
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()


class LSTStream:
    """
    Main stream class that yields LST data with computed features.
    
    This class:
    1. Loads LST parquet files
    2. Yields rows or batches with features appended
    3. Supports both lazy (per-row) and eager (batch) evaluation
    """
    
    def __init__(
        self,
        data_root: Optional[Path] = None,
        prepared_data_path: Optional[Path] = None,
        batch_size: int = 10000
    ):
        """
        Initialize LST stream.
        
        Args:
            data_root: Root data directory (default: /src/downloads/)
            prepared_data_path: Path to prepared stream data
            batch_size: Number of rows per batch in stream_batches()
        """
        self.data_root = data_root or (
            Path(__file__).parent.parent / "downloads"
        )
        self.lst_pixels_path = self.data_root / "lst_parquets" / "pixels"
        self.batch_size = batch_size
        
        self.feature_transformer = FeatureTransformer(prepared_data_path)
        self._lst_files = None
        
    @property
    def lst_files(self) -> List[Path]:
        """Get list of all LST parquet files."""
        if self._lst_files is None:
            self._lst_files = sorted(self.lst_pixels_path.glob("*.parquet"))
        return self._lst_files
    
    def stream_rows(self) -> Generator[Tuple[pd.Series, Dict[str, Any]], None, None]:
        """
        Stream individual rows with computed features.
        
        Yields:
            (row, features) tuples where features is computed feature dict
        """
        for file_path in tqdm(self.lst_files, desc="Processing LST files"):
            tqdm.write(f"  Loading {file_path.name}...")
            df = pd.read_parquet(file_path)
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {file_path.name}", leave=False):
                # Ensure timestamp is datetime (handle YYYYMMDD_HHMMSS format)
                if 'timestamp' in row and not isinstance(row['timestamp'], pd.Timestamp):
                    try:
                        row['timestamp'] = pd.to_datetime(row['timestamp'], format='%Y%m%d_%H%M%S')
                    except:
                        row['timestamp'] = pd.to_datetime(row['timestamp'])
                
                features = self.feature_transformer.compute_features(row)
                yield row, features
    
    def stream_batches(
        self,
        include_features: bool = True
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Stream data in batches with features appended.
        
        Each batch is a DataFrame with:
        - Original LST columns (longitude, latitude, value, etc.)
        - Computed feature columns
        
        Args:
            include_features: If True, compute and include features
            
        Yields:
            DataFrames of size batch_size with all columns (LST + features)
        """
        batch_rows = []
        batch_features = {}
        
        for file_path in tqdm(self.lst_files, desc="Processing LST files"):
            tqdm.write(f"\nProcessing {file_path.name}...")
            df = pd.read_parquet(file_path)
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {file_path.name}", leave=False):
                if 'timestamp' in row and not isinstance(row['timestamp'], pd.Timestamp):
                    try:
                        row['timestamp'] = pd.to_datetime(row['timestamp'], format='%Y%m%d_%H%M%S')
                    except:
                        row['timestamp'] = pd.to_datetime(row['timestamp'])
                
                batch_rows.append(row)
                
                if include_features:
                    features = self.feature_transformer.compute_features(row)
                    for feat_name, feat_value in features.items():
                        if feat_name not in batch_features:
                            batch_features[feat_name] = []
                        batch_features[feat_name].append(feat_value)
                
                # Yield batches
                if len(batch_rows) >= self.batch_size:
                    yield self._create_batch_dataframe(batch_rows, batch_features)
                    batch_rows = []
                    batch_features = {}
        
        # Yield remaining rows
        if batch_rows:
            yield self._create_batch_dataframe(batch_rows, batch_features)
    
    def _create_batch_dataframe(
        self,
        rows: List[pd.Series],
        features: Dict[str, List]
    ) -> pd.DataFrame:
        """Create batch DataFrame from rows and features."""
        df = pd.DataFrame(rows)
        
        # Add feature columns
        for feat_name, feat_values in features.items():
            df[feat_name] = feat_values
        
        return df
    
    def register_feature(
        self,
        name: str,
        compute_fn: Callable,
        depends_on: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Register a feature (delegates to FeatureTransformer).
        
        See FeatureTransformer.register_feature for details.
        """
        self.feature_transformer.register_feature(
            name, compute_fn, depends_on, **kwargs
        )
    
    def get_feature_info(self) -> pd.DataFrame:
        """Get information about registered features."""
        return self.feature_transformer.get_feature_info()
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'feature_transformer') and self.feature_transformer:
            self.feature_transformer.close()


# Built-in feature functions
def make_trees_within_radius(radius_m: float) -> Callable:
    """
    Factory for trees-within-radius features.
    
    Uses DuckDB spatial queries for efficient filtering.
    
    Args:
        radius_m: Search radius in meters (converted to degrees: 1° ≈ 111km)
        
    Returns:
        Feature function that counts trees within radius
    """
    def compute(row, context: Dict[str, Any]) -> int:
        lon, lat = row['longitude'], row['latitude']
        
        conn = context.get('conn')
        if conn is None:
            return 0
        
        # Convert radius from meters to degrees
        # At equator: 1° = ~111.32 km, adjust for latitude
        radius_deg = radius_m / 111320
        
        try:
            # Query DuckDB for trees within bounding box (fast index lookup)
            # Note: planting_timestamp may not exist in all datasets
            query = f"""
                SELECT COUNT(*) FROM trees 
                WHERE longitude BETWEEN {lon - radius_deg} AND {lon + radius_deg}
                    AND latitude BETWEEN {lat - radius_deg} AND {lat + radius_deg}
            """
            result = conn.execute(query).fetchone()
            return result[0] if result else 0
        except Exception as e:
            tqdm.write(f"  Error querying trees: {e}")
            return 0
    
    return compute


def make_height_statistics(radius_m: float = 50) -> Callable:
    """
    Factory for height-based features in neighborhood from DHM Zarr data.
    
    Returns multi-column feature with 4 primitive columns.
    
    Args:
        radius_m: Neighborhood search radius in meters
        
    Returns:
        Feature function returning tuple of (height_mean, height_std, height_max, height_min)
    """
    def compute(row, context: Dict[str, Any]) -> Tuple[float, float, float, float]:
        lon, lat = row['longitude'], row['latitude']
        
        zarr_stores = context.get('zarr_stores', {})
        if not zarr_stores:
            return (np.nan, np.nan, np.nan, np.nan)
        
        # Use first available DHM source
        source_names = [k for k in zarr_stores.keys() if not k.endswith('_meta')]
        if not source_names:
            return (np.nan, np.nan, np.nan, np.nan)
        
        source_name = source_names[0]
        zarr_group = zarr_stores[source_name]
        metadata = zarr_stores.get(f"{source_name}_meta", {})
        
        heights_array = zarr_group['heights']
        
        # Get metadata
        lon_min = metadata.get('lon_min')
        lat_min = metadata.get('lat_min')
        resolution = metadata.get('resolution')
        
        if lon_min is None or lat_min is None or resolution is None:
            return (np.nan, np.nan, np.nan, np.nan)
        
        # Calculate grid indices
        lon_idx = int(round((lon - lon_min) / resolution))
        lat_idx = int(round((lat - lat_min) / resolution))
        
        # Calculate grid radius in pixels
        grid_radius = int(radius_m / (resolution * 111320))  # Convert meters to grid cells
        
        # Load neighborhood from Zarr (only loads necessary chunk)
        try:
            # Clamp to array bounds
            lat_start = max(0, lat_idx - grid_radius)
            lat_end = min(heights_array.shape[0], lat_idx + grid_radius + 1)
            lon_start = max(0, lon_idx - grid_radius)
            lon_end = min(heights_array.shape[1], lon_idx + grid_radius + 1)
            
            neighborhood = heights_array[lat_start:lat_end, lon_start:lon_end]
            heights = neighborhood[~np.isnan(neighborhood)].flatten()
            
            if len(heights) == 0:
                return (np.nan, np.nan, np.nan, np.nan)
            
            return (
                float(heights.mean()),
                float(heights.std()),
                float(heights.max()),
                float(heights.min())
            )
        except Exception as e:
            tqdm.write(f"  Error accessing DHM data: {e}")
            return (np.nan, np.nan, np.nan, np.nan)
    
    return compute


def make_urban_atlas_features(year: int) -> Callable:
    """
    Factory for land use classification from Urban Atlas for a specific year.
    
    Args:
        year: Year to query (2006, 2012, 2018, or 2021)
        
    Returns:
        Feature function returning land use code if found
    """
    def compute(row, context: Dict[str, Any]) -> Optional[int]:
        lon, lat = row['longitude'], row['latitude']
        
        conn = context.get('conn')
        if conn is None:
            return None
        
        try:
            # Query DuckDB for urban atlas entry containing point
            query = f"""
                SELECT luc_code FROM urban_atlas 
                WHERE year = {year}
                    AND {lon} BETWEEN min_longitude AND max_longitude
                    AND {lat} BETWEEN min_latitude AND max_latitude
                LIMIT 1
            """
            result = conn.execute(query).fetchone()
            return int(result[0]) if result and result[0] is not None else None
        except Exception as e:
            tqdm.write(f"  Error querying urban atlas: {e}")
            return None
    
    return compute


# Convenience function for decorator-style registration
_global_stream: Optional[LSTStream] = None


def init_global_stream(data_root: Optional[Path] = None) -> LSTStream:
    """Initialize global stream instance for decorator support."""
    global _global_stream
    _global_stream = LSTStream(data_root=data_root)
    return _global_stream


def register_feature(
    name: str,
    depends_on: Optional[List[str]] = None,
    **kwargs
) -> Callable:
    """
    Decorator to register a feature function.
    
    Usage:
        @register_feature("my_feature", depends_on=["longitude", "latitude"])
        def compute_my_feature(row, context):
            return some_value
    """
    def decorator(func: Callable) -> Callable:
        if _global_stream is None:
            raise RuntimeError(
                "Global stream not initialized. Call init_global_stream() first."
            )
        _global_stream.register_feature(name, func, depends_on, **kwargs)
        return func
    
    return decorator


if __name__ == "__main__":
    # Import Ghent polygon for geographic filtering
    try:
        from gathering.ghent_polygon import get_ghent_convex_hull
        ghent_coords = get_ghent_convex_hull()
        ghent_polygon = Polygon(ghent_coords)
        tqdm.write("Loaded Ghent convex hull for geographic filtering")
    except Exception as e:
        tqdm.write(f"[WARNING] Could not load Ghent polygon: {e}")
        tqdm.write("[WARNING] Geographic filtering will be disabled")
        ghent_polygon = None
    
    # Load timestamp metadata to sample files across time range
    try:
        tiffs_queries_path = Path(__file__).parent.parent / "downloads" / "lst_parquets" / "tiffs_queries.parquet"
        tiffs_queries_df = pd.read_parquet(tiffs_queries_path)
        tqdm.write(f"Loaded {len(tiffs_queries_df)} LST files with timestamp metadata")
        
        # Extract year from timestamp for stratified sampling
        tiffs_queries_df['year'] = tiffs_queries_df['timestamp'].str[:4].astype(int)
        years_available = sorted(tiffs_queries_df['year'].unique())
        tqdm.write(f"Available years: {years_available[0]} to {years_available[-1]}")
        
        # Sample files evenly across the year range
        sample_indices = []
        for year in years_available[::max(1, len(years_available)//10)]:  # Sample ~10 years
            year_files = tiffs_queries_df[tiffs_queries_df['year'] == year]
            if len(year_files) > 0:
                sample_idx = year_files.sample(min(2, len(year_files)), random_state=42).index[0]
                sample_indices.append(sample_idx)
        
        selected_files = set(tiffs_queries_df.loc[sample_indices, 'parquet_file'].values)
        tqdm.write(f"Selected {len(selected_files)} LST files for sampling across timestamp range")
    except Exception as e:
        tqdm.write(f"[WARNING] Could not load timestamp metadata: {e}")
        tqdm.write("[WARNING] Will stream all files (may lack timestamp variation)")
        selected_files = None
    
    # Example usage
    tqdm.write("\n" + "="*60)
    tqdm.write("LST Stream with Feature Engineering")
    tqdm.write("="*60)
    tqdm.write("\nInitializing stream...")
    # Use very small batch size to force progression through multiple files for timestamp variation
    stream = LSTStream(batch_size=500)
    
    try:
        # Register built-in features
        tqdm.write("\nRegistering features...")
        stream.register_feature(
            "trees_within_100m",
            make_trees_within_radius(100),
            depends_on=["longitude", "latitude", "timestamp"],
            description="Count of trees within 100m radius",
            columns=["trees_within_100m"]
        )
        
        stream.register_feature(
            "height_neighborhood_50m",
            make_height_statistics(radius_m=50),
            depends_on=["longitude", "latitude"],
            description="Height statistics (mean, std, max, min) within 50m radius",
            columns=["height_mean", "height_std", "height_max", "height_min"]
        )
        
        stream.register_feature(
            "urban_atlas_2021",
            make_urban_atlas_features(2021),
            depends_on=["longitude", "latitude"],
            description="Land use code from Urban Atlas 2021",
            columns=["urban_atlas_2021"]
        )
        
        tqdm.write("\nRegistered features:")
        feature_info = stream.get_feature_info()
        print(feature_info)
        
        # Stream a sample of data with computed features
        tqdm.write("\n" + "="*60)
        tqdm.write("Streaming sample data with features...")
        tqdm.write("="*60)
        
        sample_batches = []
        batch_count = 0
        total_collected = 0
        target_rows = 15000
        
        # Track file changes by timestamp
        last_timestamp = None
        rows_from_current_file = 0
        max_rows_per_file = 500  # Only collect 500 rows per file to force multi-file variety
        files_count = 0
        
        # Geographic bounds
        lon_min, lon_max = 3.55, 3.85
        lat_min, lat_max = 50.95, 51.2
        
        for batch in stream.stream_batches(include_features=True):
            # Skip this file if we're doing selective sampling and it's not in selected set
            if selected_files is not None:
                # We can't easily check file boundaries in batch mode, so we'll process all
                # but the timestamp variation is ensured by the overall data collection
                pass
            
            # Apply all filters:
            # 1. Value != 0 and emissivity != "NDVI"
            # 2. Longitude and latitude bounds
            # 3. Inside Ghent polygon (if loaded)
            filtered_batch = batch[
                (batch['value'] != 0) &
                (batch['emissivity'] != 'NDVI') &
                (batch['longitude'] >= lon_min) &
                (batch['longitude'] <= lon_max) &
                (batch['latitude'] >= lat_min) &
                (batch['latitude'] <= lat_max)
            ].copy()
            
            # Apply polygon check if available
            if ghent_polygon is not None and len(filtered_batch) > 0:
                mask = filtered_batch.apply(
                    lambda row: ghent_polygon.contains(Point(row['longitude'], row['latitude'])),
                    axis=1
                )
                filtered_batch = filtered_batch[mask].copy()
            
            if len(filtered_batch) > 0:
                # Check if timestamp changed (indicates new file)
                current_timestamp = filtered_batch['timestamp'].iloc[0] if len(filtered_batch) > 0 else None
                if current_timestamp != last_timestamp:
                    files_count += 1
                    rows_from_current_file = 0
                    last_timestamp = current_timestamp
                
                # Limit rows from each file for variety
                take_from_batch = min(len(filtered_batch), max_rows_per_file - rows_from_current_file)
                if take_from_batch > 0:
                    sample_batches.append(filtered_batch.iloc[:take_from_batch].copy())
                    rows_from_current_file += take_from_batch
                    total_collected += take_from_batch
                    tqdm.write(f"  Batch {batch_count + 1}: {take_from_batch} rows from file {files_count} (total: {total_collected})")
                    batch_count += 1
            
            # Collect until we have enough rows and data from multiple files
            if total_collected >= target_rows and files_count >= 10:
                tqdm.write(f"  Reached target with data from {files_count} files")
                break
        
        # Combine sample batches
        if sample_batches:
            sample_df = pd.concat(sample_batches, ignore_index=True)
            
            # Trim to exact target if we collected more
            if len(sample_df) > target_rows:
                sample_df = sample_df[:target_rows].copy()
            
            # Save to CSV
            output_file = Path(__file__).parent.parent / "lst_sample_with_features.csv"
            sample_df.to_csv(output_file, index=False)
            
            tqdm.write(f"\n[SUCCESS] Sample data saved to: {output_file}")
            tqdm.write(f"  Total rows (after filtering): {len(sample_df)}")
            tqdm.write(f"  Total columns: {sample_df.shape[1]}")
            tqdm.write(f"  Collected from {batch_count} batches")
            tqdm.write(f"\nData overview:")
            tqdm.write(f"  Longitude range: [{sample_df['longitude'].min():.4f}, {sample_df['longitude'].max():.4f}]")
            tqdm.write(f"  Latitude range: [{sample_df['latitude'].min():.4f}, {sample_df['latitude'].max():.4f}]")
            tqdm.write(f"  Timestamp range: {sample_df['timestamp'].min()} to {sample_df['timestamp'].max()}")
            tqdm.write(f"  Value range: [{sample_df['value'].min():.4f}, {sample_df['value'].max():.4f}]")
            tqdm.write(f"\nFirst few rows:")
            print(sample_df.head())
        
        tqdm.write("\n" + "="*60)
        tqdm.write("[SUCCESS] Test complete")
        tqdm.write("="*60)
    finally:
        stream.close()
