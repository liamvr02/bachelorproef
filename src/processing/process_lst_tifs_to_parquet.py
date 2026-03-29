from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm


# --------------------------------------------------------------------------------------
# TIFF -> Parquet
# --------------------------------------------------------------------------------------
def tiff_to_parquet(
    tif_file: Path,
    landsat_id: str,
    emissivity: str,
    image_id: str,
    timestamp: Optional[str],
    output_dir: Path,
    keep_zero_values: bool = False,
) -> Optional[Path]:
    """
    Converts a TIFF into a parquet file of pixels.

    Parameters
    ----------
    tif_file : Path
    landsat_id : str
    emissivity : str
    image_id : str
    timestamp : Optional[str]
    output_dir : Path
    keep_zero_values : bool
        If True, keeps pixels where value == 0.

    Returns
    -------
    Path or None
    """
    with rasterio.open(tif_file) as src:
        data = src.read(1)

        if keep_zero_values:
            mask = ~np.isnan(data)
        else:
            mask = (~np.isnan(data)) & (data != 0)

        rows, cols = np.where(mask)

        if len(rows) == 0:
            tqdm.write(f"No valid pixels in {tif_file.name}")
            return None

        xs, ys = rasterio.transform.xy(src.transform, rows, cols)

        df = pd.DataFrame(
            {
                "longitude": xs,
                "latitude": ys,
                "value": data[rows, cols],
                "landsat_id": landsat_id,
                "emissivity": emissivity,
                "image_id": image_id,
                "timestamp": timestamp,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp if isinstance(timestamp, str) else "notimestamp"
    landsat_id_clean = str(landsat_id).replace("/", "-")
    emissivity_clean = str(emissivity).replace("/", "-")

    filename = f"{landsat_id_clean}_{emissivity_clean}_{image_id}_{ts}.parquet"
    out_path = output_dir / filename

    df.to_parquet(out_path, index=False)

    tqdm.write(f"Saved parquet: {filename}")
    return out_path


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _parse_folder_name(folder_name: str) -> Tuple[str, str, str, str, str]:
    parts = folder_name.split("_")

    if len(parts) < 6:
        raise ValueError(f"Unexpected folder name: {folder_name}")

    landsat_part = parts[0]
    landsat_id = landsat_part[1:] if landsat_part.startswith("L") else landsat_part

    emissivity = parts[1]
    start_date = parts[2]
    end_date = parts[3]
    image_id = parts[4]

    return landsat_id, emissivity, start_date, end_date, image_id


def _extract_timestamp_from_tif(tif_path: Path):
    name = tif_path.stem

    for suffix in [".LST", ".ST_B10"]:
        if suffix in name:
            name = name.split(suffix)[0]

    if "_" in name:
        return name

    return np.nan


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------
def all_lst_tifs_to_parquet(keep_zero_values: bool = False) -> None:
    base_dir = Path(__file__).parent.parent.absolute()
    tif_root = base_dir / "downloads" / "lst_tifs"
    output_root = base_dir / "downloads" / "lst_parquets"
    pixel_output_dir = output_root / "pixels"

    output_root.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"Scanning TIFF folders in: {tif_root}")

    folders = sorted(p for p in tif_root.iterdir() if p.is_dir())
    tqdm.write(f"Found {len(folders)} folders")

    meta_rows = []

    for folder in tqdm(folders, desc="Processing TIFF folders"):
        folder_name = folder.name

        try:
            landsat_id, emissivity, start_date, end_date, image_id = _parse_folder_name(
                folder_name
            )
        except ValueError:
            tqdm.write(f"Skipping invalid folder name: {folder_name}")
            continue

        tif_files = list(folder.glob("*.tif"))
        if not tif_files:
            tqdm.write(f"No TIFF found in {folder_name}")
            continue

        tif_file = tif_files[0]
        timestamp = _extract_timestamp_from_tif(tif_file)

        parquet_path = tiff_to_parquet(
            tif_file=tif_file,
            landsat_id=landsat_id,
            emissivity=emissivity,
            image_id=image_id,
            timestamp=timestamp,
            output_dir=pixel_output_dir,
            keep_zero_values=keep_zero_values,
        )

        meta_rows.append(
            {
                "foldername": folder_name,
                "image_id": image_id,
                "timestamp": timestamp,
                "query_start_date": start_date,
                "query_end_date": end_date,
                "parquet_file": str(parquet_path) if parquet_path else None,
            }
        )

    meta_df = pd.DataFrame(meta_rows)

    meta_path = output_root / "tiffs_queries.parquet"
    meta_df.to_parquet(meta_path, index=False)

    tqdm.write(f"Metadata saved: {meta_path}")
    tqdm.write("Processing of tifs to parquets complete.")


# --------------------------------------------------------------------------------------
# Script entry
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    all_lst_tifs_to_parquet(keep_zero_values=True)
