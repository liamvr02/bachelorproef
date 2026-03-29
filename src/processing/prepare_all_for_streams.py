"""
Prepare and index source datasets for efficient streaming with fast spatial-temporal queries.

Uses memory-efficient formats to handle 8.4GB+ data on 8GB RAM:
1. DHM (height map) → Zarr (chunked array format, load on demand)
2. Trees (points) → DuckDB table with spatial index
3. Urban Atlas (polygons) → DuckDB table referencing Parquet files
4. LST metadata → Parquet

This avoids pickling and supports streaming queries without full memory load.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import duckdb
import zarr
import json
from datetime import datetime
from tqdm import tqdm
from tqdm.std import tqdm as tqdm_std


class DataPreparer:
    """Prepares source datasets for efficient streaming queries using DuckDB and Zarr."""
    
    def __init__(self, data_root: Path):
        """
        Initialize the data preparer.
        
        Args:
            data_root: Path to /src/downloads/ directory
        """
        self.data_root = Path(data_root)
        self.dhm_path = self.data_root / "DHM_parquets"
        self.trees_path = self.data_root / "trees"
        self.lst_path = self.data_root / "lst_parquets"
        self.urban_atlas_path = self.data_root / "urban_atlas_parquets"
        self.output_path = Path(__file__).parent.parent / "prepared_stream_data"
        self.output_path.mkdir(exist_ok=True)
        
        # DuckDB database for spatial queries
        self.db_path = self.output_path / "stream_index.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        
        # Install extensions
        try:
            self.conn.execute("INSTALL httpfs")
            self.conn.execute("LOAD httpfs")
        except:
            pass  # Optional extension
        
        try:
            self.conn.execute("INSTALL spatial")
            self.conn.execute("LOAD spatial")
        except:
            logger.warning("Spatial extension not available, will use manual distance queries")
        
    def prepare_dhm(self, chunk_size: int = 512) -> None:
        """
        Prepare DHM (height map) data as Zarr format.
        
        Zarr enables chunked access without loading entire dataset into memory.
        Uses vectorized NumPy operations for efficiency.
        Saves metadata as separate JSON files to avoid Zarr attr issues.
        
        Args:
            chunk_size: Size of Zarr chunks (512x512 is good for typical DHM files)
        """
        tqdm.write("Preparing DHM data to Zarr format...")
        
        dhm_files = list(self.dhm_path.glob("*.parquet"))
        if not dhm_files:
            tqdm.write("No DHM parquet files found")
            return
        
        # Process each DHM source separately
        for source_file in tqdm(dhm_files, desc="DHM sources"):
            source_name = source_file.stem
            tqdm.write(f"Processing {source_name}...")
            
            # Read DHM data
            df = pd.read_parquet(source_file)
            
            # Get bounds and infer resolution
            lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
            lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
            resolution = df['longitude'].diff().abs().median()  # Infer resolution
            
            # Create grid dimensions
            n_lon = int((lon_max - lon_min) / resolution) + 1
            n_lat = int((lat_max - lat_min) / resolution) + 1
            
            tqdm.write(f"  Grid size: {n_lon} x {n_lat}, resolution: {resolution:.6f}°")
            
            # Create output zarr path
            zarr_path = self.output_path / f"dhm_{source_name}.zarr"
            zarr_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Remove existing zarr directory if it exists
            import shutil
            if zarr_path.exists():
                shutil.rmtree(zarr_path)
            
            # Create Zarr group
            root = zarr.open_group(str(zarr_path), mode='w')
            
            # Create height array - initialized with NaN
            heights = root.create_array(
                'heights',
                shape=(n_lat, n_lon),
                chunks=(min(256, n_lat), min(256, n_lon)),
                dtype='float32',
                fill_value=np.nan
            )
            
            # Vectorized assignment: convert all coordinates at once
            lon_indices = ((df['longitude'] - lon_min) / resolution).round().astype(int)
            lat_indices = ((df['latitude'] - lat_min) / resolution).round().astype(int)
            
            # Filter valid indices
            valid = (lon_indices >= 0) & (lon_indices < n_lon) & \
                    (lat_indices >= 0) & (lat_indices < n_lat)
            
            if valid.sum() > 0:
                valid_lats = lat_indices[valid]
                valid_lons = lon_indices[valid]
                valid_values = df.loc[valid, 'value'].values
                
                # Write to array using fancy indexing
                heights[valid_lats, valid_lons] = valid_values
                tqdm.write(f"  Wrote {valid.sum()} height values")
            
            # Save metadata as JSON (separate file to avoid Zarr attr issues)
            metadata = {
                'lon_min': float(lon_min),
                'lon_max': float(lon_max),
                'lat_min': float(lat_min),
                'lat_max': float(lat_max),
                'resolution': float(resolution),
                'source': source_name,
                'grid_shape': [int(n_lat), int(n_lon)]
            }
            
            metadata_file = self.output_path / f"dhm_{source_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            tqdm.write(f"  Zarr saved to {zarr_path.name}")
        
    def prepare_urban_atlas(self) -> None:
        """
        Prepare Urban Atlas data in DuckDB for efficient spatial queries.
        Handles varying schemas across years by standardizing columns.
        """
        tqdm.write("Preparing Urban Atlas data...")
        
        ua_files = list(self.urban_atlas_path.glob("*.parquet"))
        if not ua_files:
            tqdm.write("No Urban Atlas parquet files found")
            return
        
        # Create initial table from first file
        self.conn.execute("""
            DROP TABLE IF EXISTS urban_atlas
        """)
        
        # Load first file to establish base structure
        first_file = sorted(ua_files)[0]
        self.conn.execute(f"""
            CREATE TABLE urban_atlas AS
            SELECT luc_code, 
                   min_longitude, max_longitude, 
                   min_latitude, max_latitude,
                   year, geometry_wkt
            FROM read_parquet('{first_file}')
        """)
        tqdm.write(f"  Loaded {first_file.name}")
        
        # Append data from remaining files, standardizing schemas
        for ua_file in tqdm(sorted(ua_files)[1:], desc="Urban Atlas files"):
            try:
                self.conn.execute(f"""
                    INSERT INTO urban_atlas
                    SELECT luc_code, 
                           min_longitude, max_longitude,
                           min_latitude, max_latitude,
                           year, geometry_wkt
                    FROM read_parquet('{ua_file}')
                """)
                tqdm.write(f"  Loaded {ua_file.name}")
            except Exception as e:
                tqdm.write(f"  Error loading {ua_file.name}: {e}")
                continue
        
        # Create spatial indices on bounding box columns
        try:
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ua_lon 
                ON urban_atlas (min_longitude, max_longitude)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ua_lat 
                ON urban_atlas (min_latitude, max_latitude)
            """)
        except:
            tqdm.write("Could not create spatial indices")
        
        # Index by year for temporal queries
        try:
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ua_year ON urban_atlas (year)
            """)
        except:
            pass
        
        # Get row count
        count = self.conn.execute("SELECT COUNT(*) FROM urban_atlas").fetchone()[0]
        tqdm.write(f"Urban Atlas table created ({count:,} rows)")
        
    def prepare_trees(self) -> None:
        """
        Prepare trees data in DuckDB with spatial indexing.
        
        Uses DuckDB's memory-mapped Parquet reading to avoid full memory load.
        Extracts coordinates from geo_point_2d field if present.
        """
        tqdm.write("Preparing trees data...")
        
        trees_files = list(self.trees_path.glob("*.csv"))
        if not trees_files:
            tqdm.write("No trees CSV file found")
            return
        
        trees_file = trees_files[0]
        
        # Read into DuckDB (can handle larger-than-memory files with streaming)
        self.conn.execute(f"""
            DROP TABLE IF EXISTS trees_raw
        """)
        
        self.conn.execute(f"""
            CREATE TABLE trees_raw AS
            SELECT * FROM read_csv('{trees_file}')
        """)
        
        # Check actual column names
        columns = self.conn.execute("PRAGMA table_info(trees_raw)").fetchall()
        col_names = [col[1] for col in columns]
        tqdm.write(f"Trees table columns: {col_names}")
        
        # Try to extract coordinates from geo_point_2d if it exists (format: "lat, lon")
        has_geo_point = 'geo_point_2d' in col_names
        
        if has_geo_point:
            # geo_point_2d is in "latitude, longitude" format (CSV standard)
            self.conn.execute("""
                DROP TABLE IF EXISTS trees
            """)
            
            self.conn.execute("""
                CREATE TABLE trees AS
                SELECT *,
                       CAST(TRIM(split_part(geo_point_2d, ',', 1)) AS DOUBLE) as latitude,
                       CAST(TRIM(split_part(geo_point_2d, ',', 2)) AS DOUBLE) as longitude
                FROM trees_raw
            """)
            
            tqdm.write("Extracted coordinates from geo_point_2d field")
            
            # Create spatial indices on extracted columns
            try:
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trees_lon ON trees (longitude)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trees_lat ON trees (latitude)")
            except Exception as e:
                logger.warning(f"Could not create spatial indices: {e}")
        else:
            # Look for direct lon/lat columns
            lon_col = next((c for c in col_names if 'lon' in c.lower() or 'x' in c.lower()), None)
            lat_col = next((c for c in col_names if 'lat' in c.lower() or 'y' in c.lower()), None)
            
            if lon_col and lat_col:
                tqdm.write(f"Found spatial columns: longitude={lon_col}, latitude={lat_col}")
                
                # Rename to standardized names
                self.conn.execute(f"""
                    DROP TABLE IF EXISTS trees
                """)
                
                # Use CREATE TABLE AS to rename columns
                col_list = ", ".join(col_names)
                self.conn.execute(f"""
                    CREATE TABLE trees AS
                    SELECT * FROM trees_raw
                """)
                
                # Create indices on actual spatial columns
                try:
                    self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_trees_lon ON trees ({lon_col})")
                    self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_trees_lat ON trees ({lat_col})")
                except Exception as e:
                    tqdm.write(f"Could not create spatial indices: {e}")
            else:
                tqdm.write(f"Could not identify longitude/latitude columns in trees data")
                
                # Just use the raw table without spatial indexing
                self.conn.execute(f"""
                    DROP TABLE IF EXISTS trees
                """)
                
                self.conn.execute(f"""
                    CREATE TABLE trees AS
                    SELECT * FROM trees_raw
                """)
        
        self.conn.execute("DROP TABLE IF EXISTS trees_raw")
        
        tree_count = self.conn.execute("SELECT COUNT(*) FROM trees").fetchone()[0]
        tqdm.write(f"Trees table created ({tree_count} trees)")
        
    def prepare_lst_metadata(self) -> None:
        """
        Prepare LST metadata in DuckDB for timestamp lookups.
        """
        tqdm.write("Preparing LST metadata...")
        
        tiff_queries_file = self.lst_path / "tiffs_queries.parquet"
        if not tiff_queries_file.exists():
            tqdm.write("LST tiffs_queries.parquet not found")
            return
        
        # Load into DuckDB
        self.conn.execute("""
            DROP TABLE IF EXISTS lst_metadata
        """)
        
        self.conn.execute(f"""
            CREATE TABLE lst_metadata AS
            SELECT * FROM read_parquet('{tiff_queries_file}')
        """)
        
        # Create timestamp index
        try:
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lst_timestamp ON lst_metadata (timestamp)
            """)
        except:
            pass
        
        tqdm.write("LST metadata table created")
        
    def validate_prepared_data(self) -> bool:
        """Validate that all required prepared datasets exist."""
        # Check DuckDB tables
        tables = self.conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
        table_names = [t[0] for t in tables]
        
        required_tables = ['trees', 'lst_metadata']
        missing = [t for t in required_tables if t not in table_names]
        
        # Check Zarr files
        zarr_files = list(self.output_path.glob("dhm_*.zarr"))
        
        if missing:
            tqdm.write(f"Missing tables: {missing}")
        
        if not zarr_files:
            tqdm.write("No DHM Zarr files found")
        
        if not missing and zarr_files:
            tqdm.write("All required prepared datasets found")
            return True
        
        return len(missing) == 0 and len(zarr_files) > 0
        
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()


def main():
    """Main execution function."""
    data_root = Path(__file__).parent.parent / "downloads"
    
    if not data_root.exists():
        tqdm.write(f"Data root not found: {data_root}")
        return
    
    preparer = DataPreparer(data_root)
    
    try:
        # Prepare all datasets
        preparer.prepare_dhm()
        preparer.prepare_trees()
        preparer.prepare_urban_atlas()
        preparer.prepare_lst_metadata()
        
        # Validate
        if preparer.validate_prepared_data():
            tqdm.write("\nAll data preparation complete!")
        else:
            tqdm.write("\nSome data preparation failed")
    finally:
        preparer.close()


if __name__ == "__main__":
    main()
