"""
utils.py
========
Shared helpers for the ds/ analysis scripts.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_DS_DIR   = Path(__file__).parent   # src/ds/
_SRC      = _DS_DIR.parent          # src/
_DATA_DIR = _SRC / "test_data"


def load_parquet_sample(
    path: Path,
    max_rows: Optional[int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Read a parquet file and return a representative random sample.

    When max_rows is None or >= total rows, the full file is returned.
    Otherwise, rows are sampled proportionally from each row group using
    pandas.DataFrame.sample() so the result spans the file's full range
    rather than taking the first N rows.
    """
    import pyarrow.parquet as pq

    pf    = pq.ParquetFile(path)
    total = pf.metadata.num_rows

    if max_rows is None or max_rows >= total:
        return pf.read().to_pandas()

    frac   = max_rows / total
    chunks = []
    for rg in range(pf.metadata.num_row_groups):
        chunk    = pf.read_row_group(rg).to_pandas()
        n_sample = max(1, round(len(chunk) * frac))
        chunks.append(chunk.sample(n=min(n_sample, len(chunk)), random_state=seed))

    df = pd.concat(chunks, ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


def list_datasets() -> List[Tuple[str, Path]]:
    """Return [(name, parquet_path)] for every dataset in test_data/."""
    if not _DATA_DIR.exists():
        return []
    return sorted((p.stem, p) for p in _DATA_DIR.glob("*.parquet"))


def run_timestamp() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def version_meta() -> Dict[str, str]:
    """Return a dict of key runtime versions for traceability."""
    return {
        "python": sys.version.split()[0],
        "numpy":  np.__version__,
        "pandas": pd.__version__,
    }


def dataset_stats(df: pd.DataFrame) -> Dict:
    """
    Compute descriptive statistics for a loaded dataset DataFrame.

    Returns a dict with row/tile counts, temporal coverage, temperature
    summary, LST source breakdown, and NDVI coverage — enough to
    identify which slice of the data drove a particular analysis result.
    """
    stats: Dict = {"n_rows": int(len(df))}

    if "tile_id" in df.columns:
        stats["n_tiles"] = int(df["tile_id"].nunique())

    if "year" in df.columns:
        stats["year_range"] = [int(df["year"].min()), int(df["year"].max())]

    if "month_of_year" in df.columns:
        stats["month_range"] = [
            int(df["month_of_year"].min()), int(df["month_of_year"].max())
        ]

    if "hour_of_day" in df.columns:
        stats["hour_range"] = [
            round(float(df["hour_of_day"].min()), 2),
            round(float(df["hour_of_day"].max()), 2),
        ]

    # Temperature summary (the analysis target)
    temp_col = next(
        (c for c in ("temperature", "aster_lst", "modis_lst") if c in df.columns),
        None,
    )
    if temp_col:
        t = df[temp_col].dropna()
        stats["temperature"] = {
            "col":  temp_col,
            "mean": round(float(t.mean()), 3),
            "std":  round(float(t.std()),  3),
            "min":  round(float(t.min()),  3),
            "max":  round(float(t.max()),  3),
        }

    # LST source breakdown
    n_aster = int(df["aster_lst"].notna().sum()) if "aster_lst" in df.columns else 0
    n_modis = int(df["modis_lst"].notna().sum()) if "modis_lst" in df.columns else 0
    if n_aster or n_modis:
        stats["lst_sources"] = {"aster": n_aster, "modis": n_modis}

    # NDVI coverage (fraction non-null)
    if "ndvi" in df.columns:
        stats["ndvi_coverage_frac"] = round(
            float(df["ndvi"].notna().mean()), 4
        )

    return stats


def compute_structural_ndvi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``ndvi_struct_mean`` and ``ndvi_struct_std`` columns to *df*.

    These are the per-tile temporal mean and standard deviation of the
    instantaneous NDVI across all observations in this dataset slice.
    Unlike per-observation NDVI (which is co-measured with LST from the
    same satellite acquisition), the tile-level mean integrates across
    multiple dates and is a proxy for persistent canopy cover — an
    independent structural greening indicator.

    If ``ndvi`` or ``tile_id`` are absent, *df* is returned unchanged.
    Missing NDVI values are excluded from the per-tile aggregation.
    """
    if "ndvi" not in df.columns or "tile_id" not in df.columns:
        return df

    agg = (
        df[["tile_id", "ndvi"]]
        .dropna(subset=["ndvi"])
        .groupby("tile_id", sort=False)["ndvi"]
        .agg(ndvi_struct_mean="mean", ndvi_struct_std="std")
        .reset_index()
    )
    return df.merge(agg, on="tile_id", how="left")
