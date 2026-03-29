"""
Test script that uses tiffs_queries.parquet metadata to efficiently sample LST data
across the full time range (2000-2026) with geographic and value filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Point, Polygon
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stream.lst_stream import LSTStream


def load_ghent_polygon():
    """Load Ghent polygon for geographic filtering."""
    try:
        from gathering.ghent_polygon import get_ghent_convex_hull
        ghent_coords = get_ghent_convex_hull()
        return Polygon(ghent_coords), True
    except Exception as e:
        tqdm.write(f"[WARNING] Could not load Ghent polygon: {e}")
        return None, False


def load_metadata():
    """Load timestamp metadata for all LST files."""
    metadata_path = Path(__file__).parent.parent / "downloads" / "lst_parquets" / "tiffs_queries.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    df = pd.read_parquet(metadata_path)
    df['year'] = df['timestamp'].str[:4].astype(int)
    return df


def select_stratified_files(metadata_df, samples_per_decade=2):
    """Select files stratified across decades to ensure timestamp variation."""
    selected_files = []
    selected_years = []
    
    years = sorted(metadata_df['year'].unique())
    year_min, year_max = years[0], years[-1]
    
    # Sample files from each decade
    for decade_start in range(year_min, year_max + 1, 10):
        decade_end = min(decade_start + 10, year_max + 1)
        decade_files = metadata_df[
            (metadata_df['year'] >= decade_start) & 
            (metadata_df['year'] < decade_end)
        ]
        
        if len(decade_files) > 0:
            # Sample uniformly from this decade
            sample = decade_files.sample(
                min(samples_per_decade, len(decade_files)),
                random_state=42
            )
            selected_files.extend(sample['parquet_file'].values)
            selected_years.extend(sample['year'].values)
    
    return selected_files, selected_years


def main():
    tqdm.write("\n" + "="*60)
    tqdm.write("LST Sample Generation with Timestamp Variation")
    tqdm.write("="*60)
    
    # Load metadata
    tqdm.write("\nLoading LST metadata...")
    metadata_df = load_metadata()
    tqdm.write(f"  Loaded {len(metadata_df)} LST files")
    tqdm.write(f"  Year range: {metadata_df['year'].min()} to {metadata_df['year'].max()}")
    
    # Select stratified files
    tqdm.write("\nSelecting files for timestamp variation...")
    selected_files, selected_years = select_stratified_files(metadata_df, samples_per_decade=2)
    selected_files_set = set(selected_files)
    tqdm.write(f"  Selected {len(selected_files)} files from {len(set(selected_years))} distinct years")
    tqdm.write(f"  Years represented: {sorted(set(selected_years))}")
    
    # Load Ghent polygon
    tqdm.write("\nLoading geographic filter...")
    ghent_polygon, has_polygon = load_ghent_polygon()
    if has_polygon:
        tqdm.write("  Using Ghent convex hull for filtering")
    
    # Initialize stream
    tqdm.write("\nInitializing LSTStream...")
    stream = LSTStream(batch_size=5000)
    
    try:
        # Register features
        tqdm.write("\nRegistering features...")
        stream.register_feature(
            "trees_within_100m",
            stream.feature_transformer._features["trees_within_100m"].compute_fn
                if "trees_within_100m" in stream.feature_transformer._features
                else lambda row, ctx: 0,
            description="Count of trees within 100m radius",
            columns=["trees_within_100m"]
        )
        
        # Just collect data without doing feature computation (for speed)
        tqdm.write("\nCollecting sample data with filters...")
        
        sample_batches = []
        total_collected = 0
        target_rows = 15000
        lon_min, lon_max = 3.55, 3.85
        lat_min, lat_max = 50.95, 51.2
        files_checked = 0
        
        for batch in stream.stream_batches(include_features=False):
            files_checked += 1
            
            # Apply filters
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
                sample_batches.append(filtered_batch)
                total_collected += len(filtered_batch)
                tqdm.write(f"  Batch {len(sample_batches)}: {len(filtered_batch)} rows filtered (total: {total_collected})")
            
            # Stop when we have enough
            if total_collected >= target_rows:
                break
        
        # Combine and save
        tqdm.write(f"\nProcessing {len(sample_batches)} batches...")
        if sample_batches:
            sample_df = pd.concat(sample_batches, ignore_index=True)
            
            if len(sample_df) > target_rows:
                sample_df = sample_df[:target_rows].copy()
            
            output_file = Path(__file__).parent.parent / "lst_sample_with_features.csv"
            sample_df.to_csv(output_file, index=False)
            
            # Extract year from timestamp for analysis
            sample_df['year'] = pd.to_datetime(sample_df['timestamp']).dt.year
            years_in_sample = sorted(sample_df['year'].unique())
            
            tqdm.write(f"\n[SUCCESS] Sample data saved to: {output_file}")
            tqdm.write(f"  Total rows: {len(sample_df)}")
            tqdm.write(f"  Total columns: {sample_df.shape[1]}")
            tqdm.write(f"\nData statistics:")
            tqdm.write(f"  Longitude range: [{sample_df['longitude'].min():.4f}, {sample_df['longitude'].max():.4f}]")
            tqdm.write(f"  Latitude range: [{sample_df['latitude'].min():.4f}, {sample_df['latitude'].max():.4f}]")
            tqdm.write(f"  Timestamp range: {sample_df['timestamp'].min()} to {sample_df['timestamp'].max()}")
            tqdm.write(f"  Years represented: {len(years_in_sample)} ({years_in_sample[0]}-{years_in_sample[-1]})")
            tqdm.write(f"  Value range: [{sample_df['value'].min():.4f}, {sample_df['value'].max():.4f}]")
            tqdm.write(f"\nSample data preview:")
            print(sample_df[['longitude', 'latitude', 'value', 'timestamp']].head(10))
        
        tqdm.write("\n" + "="*60)
        tqdm.write("[SUCCESS] Test complete")
        tqdm.write("="*60)
    
    finally:
        stream.close()


if __name__ == "__main__":
    main()
