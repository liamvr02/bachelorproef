import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from pyproj import Transformer
from tqdm import tqdm

DOWNLOAD_FOLDER = Path(__file__).parent.parent.absolute() / "downloads"
OUTPUT_FOLDER = DOWNLOAD_FOLDER / "DHM_parquets"
OUTPUT_FOLDER.mkdir(exist_ok=True)

CRS_SRC = "EPSG:31370"
CRS_DST = "EPSG:4326"

transformer = Transformer.from_crs(CRS_SRC, CRS_DST, always_xy=True)


def find_tifs(folder):
    return list(folder.rglob("*.tif"))


def process_raster(tif_path, source):
    output_path = OUTPUT_FOLDER / (tif_path.stem + ".parquet")

    tqdm.write(f"Processing {tif_path.name}")

    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        dfs = []

        block_windows = list(src.block_windows(1))

        for _, window in tqdm(
            block_windows,
            desc=f"Blocks {tif_path.name}",
            leave=False
        ):
            data = src.read(1, window=window)
            transform = src.window_transform(window)

            h, w = data.shape

            rows, cols = np.meshgrid(
                np.arange(h),
                np.arange(w),
                indexing="ij"
            )

            xs, ys = rasterio.transform.xy(
                transform,
                rows,
                cols,
                offset="center"
            )

            xs = np.asarray(xs).ravel()
            ys = np.asarray(ys).ravel()
            values = data.ravel()

            mask = np.ones(values.shape, dtype=bool)
            if nodata is not None:
                mask &= values != nodata

            xs = xs[mask]
            ys = ys[mask]
            values = values[mask]

            if len(values) == 0:
                continue

            lon, lat = transformer.transform(xs, ys)

            df = pd.DataFrame({
                "longitude": lon,
                "latitude": lat,
                "value": values.astype(np.float32),
                "source": source
            })

            dfs.append(df)

        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            result.to_parquet(output_path, index=False)

    tqdm.write(f"Saved {output_path.name}")


def all_dhm_tifs_to_parquet():
    dhm1_folder = DOWNLOAD_FOLDER / "DHM1_extracted"
    dhm2_folder = DOWNLOAD_FOLDER / "DHM2_extracted"

    dhm1_tifs = find_tifs(dhm1_folder)
    dhm2_tifs = find_tifs(dhm2_folder)

    all_tasks = [(t, "DHM1") for t in dhm1_tifs] + [(t, "DHM2") for t in dhm2_tifs]

    tqdm.write(f"DHM1 rasters: {len(dhm1_tifs)}")
    tqdm.write(f"DHM2 rasters: {len(dhm2_tifs)}")
    tqdm.write(f"Total rasters: {len(all_tasks)}")

    for tif, source in tqdm(all_tasks, desc="Processing rasters"):
        process_raster(tif, source)


if __name__ == "__main__":
    all_dhm_tifs_to_parquet()