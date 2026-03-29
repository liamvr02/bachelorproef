"""
Generate LST sample with proper timestamp variation by sampling from strategically selected files.
"""
import pandas as pd
from pathlib import Path
from shapely.geometry import Point, Polygon
from tqdm import tqdm
import numpy as np

# Load metadata
metadata_path = Path("downloads/lst_parquets/tiffs_queries.parquet")
meta_df = pd.read_parquet(metadata_path)
meta_df['year'] = meta_df['timestamp'].str[:4].astype(int)

# Select files stratified across decades
selected_files = []
selected_years = []
for decade in range(2000, 2030, 5):
    decade_files = meta_df[(meta_df['year'] >= decade) & (meta_df['year'] < decade + 5)]
    if len(decade_files) > 0:
        sample = decade_files.sample(min(2, len(decade_files)), random_state=42)
        selected_files.extend(sample['parquet_file'].values)
        selected_years.extend(sample['year'].values)

tqdm.write(f"Selected {len(set(selected_years))} years: {sorted(set(selected_years))}")

# Filter parameters
lon_min, lon_max = 3.55, 3.85
lat_min, lat_max = 50.95, 51.2
target_rows = 15000
rows_per_file = 1500  # Limit rows per file for variety

# Load Ghent polygon
try:
    from gathering.ghent_polygon import get_ghent_convex_hull
    ghent_coords = get_ghent_convex_hull()
    ghent_polygon = Polygon(ghent_coords)
    use_polygon = True
except:
    use_polygon = False

# Collect data
all_data = []
for parquet_file in tqdm(selected_files, desc="Processing files"):
    if not Path(parquet_file).exists():
        continue
    
    df = pd.read_parquet(parquet_file)
    
    # Apply filters
    filtered = df[
        (df['value'] != 0) &
        (df['emissivity'] != 'NDVI') &
        (df['longitude'] >= lon_min) &
        (df['longitude'] <= lon_max) &
        (df['latitude'] >= lat_min) &
        (df['latitude'] <= lat_max)
    ]
    
    # Apply polygon filter
    if use_polygon and len(filtered) > 0:
        mask = filtered.apply(
            lambda row: ghent_polygon.contains(Point(row['longitude'], row['latitude'])),
            axis=1
        )
        filtered = filtered[mask]
    
    # Limit rows from this file
    if len(filtered) > 0:
        sample = filtered.sample(min(rows_per_file, len(filtered)), random_state=42)
        all_data.append(sample)
        tqdm.write(f"  {Path(parquet_file).name}: {len(sample)} rows")
        
        if sum(len(d) for d in all_data) >= target_rows:
            break

# Combine
if all_data:
    result_df = pd.concat(all_data, ignore_index=True)
    if len(result_df) > target_rows:
        result_df = result_df[:target_rows].copy()
    
    # Save
    output_file = Path("lst_sample_with_features.csv")
    result_df.to_csv(output_file, index=False)
    
    result_df['year'] = pd.to_datetime(result_df['timestamp'], format="%Y%m%d_%H%M%S").dt.year
    
    tqdm.write(f"\n[SUCCESS] Saved {len(result_df)} rows to {output_file}")
    tqdm.write(f"  Longitude: {result_df['longitude'].min():.4f}-{result_df['longitude'].max():.4f}")
    tqdm.write(f"  Latitude: {result_df['latitude'].min():.4f}-{result_df['latitude'].max():.4f}")
    tqdm.write(f"  Timestamps: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")
    tqdm.write(f"  Years: {sorted(result_df['year'].unique())}")
