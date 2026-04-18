"""
stream_example.py – Stream 5M rows from multiple timestamps and save to CSV.

Demonstrates:
- Building a FeatureRegistry with Urban Atlas classifications at multiple resolutions
- Configuring the stream with selected partitions and optional distribution weighting
- Streaming LST rows with computed features
- Saving results to CSV
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stream import (
    StreamConfig,
)

from features import (
    FeatureRegistry,
    nearest,
    aggregate_in_radius,
    urban_atlas_classifications_fractions,
)

from classification_groups import UA


def build_registry() -> FeatureRegistry:
    """
    Build a FeatureRegistry with base features and UA classifications.
    
    Features registered:
    - Elevation (DHM1) at query location
    - Tree density within 50m
    - UA classifications at 100m and 30m
    
    Note: NDVI is now integrated into the LST table (alongside aster_lst, modis_lst)
          and is available as an output column, not queried as a separate dataset.
    """
    reg = FeatureRegistry()
    
    # Base features
    reg.add(nearest("dhm1", columns=["elevation"], temporal="last_previous"))
    reg.add(aggregate_in_radius(
        "trees",
        radius_m=50,
        columns=[],
        agg="count",
        temporal="none",
    ))
    
    # UA classifications at 100m and 30m
    reg.add(urban_atlas_classifications_fractions(
        classification_map=UA,
        radius_m=100,
    ))
    reg.add(urban_atlas_classifications_fractions(
        classification_map=UA,
        radius_m=30,
    ))
    
    return reg


def stream_to_dataframe_filtered(
    max_rows: int = 5_000_000,
    num_timestamps: int = 20,
    multi_dimensional: bool = False,
) -> pd.DataFrame:
    """
    Stream LST rows from multiple timestamps with computed features and save to CSV.
    
    Parameters
    ----------
    max_rows : int
        Target number of rows to stream
    num_timestamps : int
        Minimum number of distinct months to include
    multi_dimensional : bool
        If True, target distributions across temperature, timestamp, and location.
        If False, use temperature-only (backwards compatible).
        
    Returns
    -------
    pd.DataFrame
        Streamed data with all computed features
    """
    # Get available partitions
    cfg = StreamConfig()
    cfg._load_catalog()
    all_partitions = sorted(cfg._partition_stats, key=lambda p: p["partition_key"])\
        if hasattr(cfg, '_partition_stats') and cfg._partition_stats else \
        [{} for _ in cfg._partition_keys]
    
    # Select partitions spanning the time range
    if len(all_partitions) < num_timestamps:
        selected_partitions = cfg._partition_keys if hasattr(cfg, '_partition_keys') else []
    else:
        step = len(cfg._partition_keys) // num_timestamps if hasattr(cfg, '_partition_keys') else 1
        selected_partitions = [
            cfg._partition_keys[i * step] if hasattr(cfg, '_partition_keys') else ""
            for i in range(num_timestamps)
        ]
    
    print(f"Streaming from {len(selected_partitions)} partitions (months)")
    
    # Configure stream with selected partitions
    cfg = StreamConfig(partition_keys=selected_partitions)
    
    if multi_dimensional:
        # Multi-dimensional distribution targeting:
        # Temperature, timestamp, and geographic location
        print("Using multi-dimensional distribution targeting...")
        
        # Temperature bins (2°C from -10 to 60°C)
        temp_edges = [float(v) for v in np.linspace(-10.0, 60.0, 36)]
        
        # Timestamp bins (seasonal: Q1-Q4 for each year from 2000 to 2025)
        ts_edges = [
            f"{year:04d}-Q{quarter}"
            for year in range(2000, 2026)
            for quarter in range(1, 5)
        ]
        
        # Coordinate bins (0.1 degree resolution)
        lon_edges = [float(v) for v in np.linspace(-180.0, 180.0, 3601)]
        lat_edges = [float(v) for v in np.linspace(-90.0, 90.0, 1801)]
        
        # Set multi-dimensional targets
        cfg.set_distribution({
            "temperature": (
                {10: 0.05, 15: 0.15, 20: 0.25, 25: 0.30, 30: 0.20, 35: 0.05},
                temp_edges
            ),
            "timestamp": (
                {
                    "2010-Q2": 0.25,  # Spring 2010
                    "2015-Q3": 0.25,  # Summer 2015
                    "2020-Q2": 0.25,  # Spring 2020
                    "2020-Q3": 0.25,  # Summer 2020
                },
                ts_edges
            ),
            "longitude": (
                {3.0: 0.5, 3.5: 0.5},  # Ghent is around 3.7°E
                lon_edges
            ),
        })
    else:
        # Simple temperature-only distribution (backwards compatible)
        print("Using temperature-only distribution targeting...")
        cfg.set_distribution({
            10: 0.05,
            15: 0.15,
            20: 0.25,
            25: 0.30,
            30: 0.20,
            35: 0.05,
        })
    
    # Build feature registry
    reg = build_registry()
    
    print(f"Streaming {max_rows:,} rows with {len(reg._descriptors)} features...")
    
    # Stream to dataframe
    df = cfg.to_dataframe(registry=reg, max_rows=max_rows)
    
    print(f"Collected {len(df):,} rows with {df.shape[1]} columns")
    
    return df


def main() -> None:
    """
    Main entry point: stream 5M rows from 20+ timestamps and save to CSV.
    
    Set multi_dimensional=True to use temperature + timestamp + location targeting,
    or False for temperature-only (default).
    """
    print("=" * 70)
    print("Streaming 5M rows from 20+ timestamps...")
    print("=" * 70)
    
    # Change multi_dimensional=True to use multi-dimensional distribution targeting
    df = stream_to_dataframe_filtered(
        max_rows=5_000_000,
        num_timestamps=20,
        multi_dimensional=False,  # Set to True for multi-dimensional targeting
    )
    
    # Save to CSV
    output_path = Path("sample_stream_output.csv")
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df):,} rows to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
