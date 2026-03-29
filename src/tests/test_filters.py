"""Quick test to verify geographic and value filtering works"""
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Point, Polygon
from tqdm import tqdm

# Load and check metadata first
metadata_path = Path("downloads/lst_parquets/tiffs_queries.parquet")
if metadata_path.exists():
    df = pd.read_parquet(metadata_path)
    df['year'] = df['timestamp'].str[:4].astype(int)
    print("Metadata loaded:")
    print(f"  Files: {len(df)}")
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}")

# Load one LST file to test filtering
lst_file = Path("downloads/lst_parquets/pixels/5_ASTER_LT51980242000222FUI00_20000809_101119.parquet")
if lst_file.exists():
    lst_df = pd.read_parquet(lst_file)
    print(f"\n1st LST file loaded: {lst_file.name}")
    print(f"  Original rows: {len(lst_df)}")
    print(f"  Columns: {list(lst_df.columns)}")
    print(f"  Value range: {lst_df['value'].min():.2f}-{lst_df['value'].max():.2f}")
    print(f"  Emissivity values: {lst_df['emissivity'].unique()}")
    
    # Apply value filter
    filtered = lst_df[(lst_df['value'] != 0) & (lst_df['emissivity'] != 'NDVI')]
    print(f"\n  After value/emissivity filter: {len(filtered)} rows ({100*len(filtered)/len(lst_df):.1f}%)")
    
    # Apply geographic filter
    lon_min, lon_max = 3.55, 3.85
    lat_min, lat_max = 50.95, 51.2
    geo_filtered = filtered[
        (filtered['longitude'] >= lon_min) &
        (filtered['longitude'] <= lon_max) &
        (filtered['latitude'] >= lat_min) &
        (filtered['latitude'] <= lat_max)
    ]
    print(f"  After geographic filter: {len(geo_filtered)} rows ({100*len(geo_filtered)/len(lst_df):.1f}%)")
    print(f"    Lon range: {geo_filtered['longitude'].min():.4f}-{geo_filtered['longitude'].max():.4f}")
    print(f"    Lat range: {geo_filtered['latitude'].min():.4f}-{geo_filtered['latitude'].max():.4f}")
    
    # Test Ghent polygon
    try:
        from gathering.ghent_polygon import get_ghent_convex_hull
        ghent_coords = get_ghent_convex_hull()
        ghent_polygon = Polygon(ghent_coords)
        print(f"\n  Ghent polygon loaded with {len(ghent_coords)} vertices")
        
        # Apply polygon filter
        mask = geo_filtered.apply(
            lambda row: ghent_polygon.contains(Point(row['longitude'], row['latitude'])),
            axis=1
        )
        poly_filtered = geo_filtered[mask]
        print(f"  After Ghent polygon filter: {len(poly_filtered)} rows ({100*len(poly_filtered)/len(lst_df):.1f}%)")
    except Exception as e:
        print(f"  Warning: Could not apply polygon filter: {e}")

print("\n[SUCCESS] Filter test complete")
