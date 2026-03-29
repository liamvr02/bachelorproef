"""Check current CSV timestamp variation"""
import pandas as pd
from pathlib import Path

csv_file = Path("lst_sample_with_features.csv")
if csv_file.exists():
    df = pd.read_csv(csv_file)
    df['year'] = pd.to_datetime(df['timestamp']).dt.year
    print("Current CSV statistics:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Timestamp: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Years represented: {sorted(df['year'].unique())}")
    print(f"  Year span: {df['year'].min()}-{df['year'].max()} ({df['year'].max()-df['year'].min()} year range)")
    print(f"  Lon range: {df['longitude'].min():.4f}-{df['longitude'].max():.4f}")
    print(f"  Lat range: {df['latitude'].min():.4f}-{df['latitude'].max():.4f}")
else:
    print("CSV not found")
