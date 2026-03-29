"""
Convert Urban Atlas land use maps from various formats (Shapefile, GPKG, FGB) to Parquet.

Converts all Urban Atlas data for Ghent (2006, 2012, 2018, 2021) from their native formats
into Parquet files optimized for longitude/latitude based spatial queries. Each output file
includes geometry as WKT and bounding box columns for efficient filtering.
"""

from pathlib import Path
from typing import Optional
import warnings

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from tqdm import tqdm

# Suppress warnings about CRS transformations
warnings.filterwarnings("ignore", category=UserWarning)

DOWNLOAD_FOLDER = Path(__file__).parent.parent.absolute() / "downloads"
URBAN_ATLAS_FOLDER = DOWNLOAD_FOLDER / "urban_atlas_extracted"
OUTPUT_FOLDER = DOWNLOAD_FOLDER / "urban_atlas_parquets"

OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

# Target CRS: WGS84 with longitude/latitude
TARGET_CRS = "EPSG:4326"


def find_input_files():
    """
    Discover all Urban Atlas input files across different formats and years.

    Returns
    -------
    dict
        Mapping of year to (file_path, format) tuples.
    """
    files = {}

    # 2006 - Shapefile
    shp_2006 = list(
        URBAN_ATLAS_FOLDER.glob("BE003L2_GENT/*/Shapefiles/*.shp")
    )
    if shp_2006:
        shp_path = shp_2006[0]
        # Skip the boundary shapefile, only use the land use one
        if "UA2006" in str(shp_path):
            files[2006] = (shp_path, "shapefile")

    # 2012 - GPKG
    gpkg_2012 = list(
        URBAN_ATLAS_FOLDER.glob(
            "BE003L2_GENT_UA2012_revised_v021/*/Data/*.gpkg"
        )
    )
    if gpkg_2012:
        files[2012] = (gpkg_2012[0], "gpkg")

    # 2018 - FGB (FlatGeobuf)
    fgb_2018 = list(
        URBAN_ATLAS_FOLDER.glob(
            "*/CLMS_UA_LCU_S2018*/*.fgb"
        )
    )
    if fgb_2018:
        files[2018] = (fgb_2018[0], "fgb")

    # 2021 - FGB (FlatGeobuf)
    fgb_2021 = list(
        URBAN_ATLAS_FOLDER.glob(
            "*/CLMS_UA_LCU_S2021*/*.fgb"
        )
    )
    if fgb_2021:
        files[2021] = (fgb_2021[0], "fgb")

    return files


def load_geodataframe(file_path: Path, file_format: str) -> Optional[gpd.GeoDataFrame]:
    """
    Load a geospatial file into a GeoDataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the file
    file_format : str
        Format identifier: 'shapefile', 'gpkg', or 'fgb'

    Returns
    -------
    GeoDataFrame or None
        Loaded geodataframe, or None if loading failed
    """
    try:
        if file_format == "shapefile":
            gdf = gpd.read_file(file_path)
        elif file_format == "gpkg":
            gdf = gpd.read_file(file_path)
        elif file_format == "fgb":
            gdf = gpd.read_file(file_path)
        else:
            tqdm.write(f"Unknown format: {file_format}")
            return None

        tqdm.write(f"Loaded {len(gdf)} features from {file_path.name}")
        return gdf

    except Exception as e:
        tqdm.write(f"Error loading {file_path.name}: {e}")
        return None


def standardize_geodataframe(
    gdf: gpd.GeoDataFrame,
    year: int
) -> gpd.GeoDataFrame:
    """
    Standardize a GeoDataFrame to a common schema for land use classification.

    Ensures consistent column names, CRS, and adds spatial index columns.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    year : int
        Year of the data

    Returns
    -------
    GeoDataFrame
        Standardized GeoDataFrame
    """
    # Transform to WGS84 if needed
    if gdf.crs is None:
        tqdm.write(f"Warning: CRS is None, assuming EPSG:31370 (Belgian Lambert)")
        gdf = gdf.set_crs("EPSG:31370")

    if gdf.crs != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    # Standardize geometry column
    if gdf.geometry.name != "geometry":
        gdf = gdf.rename_geometry("geometry")

    # Create spatial index columns from bounds (lon/lat)
    bounds = gdf.geometry.bounds
    gdf["min_longitude"] = bounds["minx"]
    gdf["max_longitude"] = bounds["maxx"]
    gdf["min_latitude"] = bounds["miny"]
    gdf["max_latitude"] = bounds["maxy"]

    # Add year column
    gdf["year"] = year

    # Convert geometry to WKT for parquet storage
    gdf["geometry_wkt"] = gdf.geometry.to_wkt()

    # Find land use classification column (varies by year/source)
    luc_columns = [col for col in gdf.columns if col.upper() in [
        "DN", "RASTERVALUE", "VALUE", "UA_2006", "UA_2012", "LC_CODE",
        "CODE_2018", "CODE_2021", "GRIDCODE"
    ]]

    if luc_columns:
        gdf = gdf.rename(columns={luc_columns[0]: "luc_code"})
    elif "luc_code" not in gdf.columns and len(gdf.columns) > 1:
        # If no obvious land use column, use the first non-geometry column
        non_geom_cols = [c for c in gdf.columns if c != "geometry"]
        if non_geom_cols:
            gdf = gdf.rename(columns={non_geom_cols[0]: "luc_code"})

    # Select final columns for parquet
    parquet_cols = [
        col for col in gdf.columns
        if col in [
            "luc_code", "min_longitude", "max_longitude",
            "min_latitude", "max_latitude", "year", "geometry_wkt"
        ] or col.lower().startswith(("code", "class", "land", "use"))
    ]

    # Always include geometry_wkt and year
    if "geometry_wkt" not in parquet_cols:
        parquet_cols.append("geometry_wkt")
    if "year" not in parquet_cols:
        parquet_cols.append("year")

    # Keep luc_code if it exists
    if "luc_code" in gdf.columns and "luc_code" not in parquet_cols:
        parquet_cols.insert(0, "luc_code")

    # Keep spatial index columns
    spatial_cols = [
        "min_longitude", "max_longitude", "min_latitude", "max_latitude"
    ]
    spatial_cols = [c for c in spatial_cols if c in gdf.columns]
    if spatial_cols:
        parquet_cols.extend([c for c in spatial_cols if c not in parquet_cols])

    # Filter to existing columns
    parquet_cols = [c for c in parquet_cols if c in gdf.columns]

    # Ensure we have at least geometry_wkt and year
    if "geometry_wkt" not in parquet_cols:
        parquet_cols.append("geometry_wkt")
    if "year" not in parquet_cols:
        parquet_cols.append("year")

    gdf = gdf[parquet_cols]

    return gdf


def process_urban_atlas_file(
    file_path: Path,
    file_format: str,
    year: int
) -> Optional[Path]:
    """
    Process a single Urban Atlas file and save as Parquet.

    Parameters
    ----------
    file_path : Path
        Path to the input file
    file_format : str
        File format identifier
    year : int
        Year of the data

    Returns
    -------
    Path or None
        Path to output parquet file, or None if processing failed
    """
    tqdm.write(f"\n--- Processing Urban Atlas {year} ({file_format}) ---")

    # Load the file
    gdf = load_geodataframe(file_path, file_format)
    if gdf is None:
        return None

    # Standardize to common schema
    gdf = standardize_geodataframe(gdf, year)

    # Convert GeoDataFrame to regular DataFrame for parquet storage
    df = pd.DataFrame(gdf)
    df = df.drop(columns=["geometry"], errors="ignore")

    # Convert Arrow dtypes to standard Python types to avoid serialization issues
    for col in df.columns:
        if hasattr(df[col].dtype, "pyarrow_dtype"):
            # Convert Arrow-backed columns to numpy-backed
            df[col] = df[col].astype(str) if df[col].dtype == "string" else df[col].array.to_numpy(dtype=object, na_value=None)

    # Save to parquet
    output_path = OUTPUT_FOLDER / f"urban_atlas_{year}.parquet"

    df.to_parquet(output_path, index=False, engine="pyarrow")
    tqdm.write(
        f"Saved {len(df)} features to {output_path.name}"
    )

    return output_path


def main():
    """Discover and process all Urban Atlas files."""
    files = find_input_files()

    if not files:
        print("No Urban Atlas files found in:", URBAN_ATLAS_FOLDER)
        return

    print(f"Found {len(files)} Urban Atlas datasets")
    print(f"Output folder: {OUTPUT_FOLDER}")

    successful = []
    for year in sorted(files.keys()):
        file_path, file_format = files[year]
        output_path = process_urban_atlas_file(file_path, file_format, year)
        if output_path:
            successful.append((year, output_path))

    print(f"\n✓ Successfully processed {len(successful)} files:")
    for year, path in successful:
        print(f"  {year}: {path.name}")


if __name__ == "__main__":
    main()
