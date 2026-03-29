import requests
from pathlib import Path
import pandas as pd
from datetime import date
import os
from tqdm import tqdm

# Direct CSV download URL
CSV_URL = "https://data.stad.gent/api/explore/v2.1/catalog/datasets/gent-in-3d/exports/csv?lang=en&timezone=Europe%2FBrussels&use_labels=true&delimiter=%3B"

# Output CSV path
OUTPUT_FILE = Path(__file__).parent.parent.absolute() / "downloads" / "g3d" / f"ghent_g3d_links_{date.today()}.csv"

def download_g3d_links_csv(url: str = CSV_URL, output_file: Path = OUTPUT_FILE):
    print(f"Downloading CSV from {url} ...")
    response = requests.get(url)
    response.raise_for_status()

    output_file.write_bytes(response.content)
    print(f"Saved CSV to {output_file}")

def get_g3d_links_csv():
    if (os.path.exists(OUTPUT_FILE)):
        print(f"{OUTPUT_FILE} already exists, reading from disk...")
    else:
        download_g3d_links_csv(CSV_URL, OUTPUT_FILE)
    
    df = pd.read_csv(OUTPUT_FILE, delimiter=";")
    return df

def download_g3d_zip_files(df: pd.DataFrame, url_column: str = "Link naar open data"):
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = row[url_column]
        filename = url.split("/")[-1]
        output_path = Path(__file__).parent.parent.absolute() / "downloads" / "g3d_zips" / filename
        
        if output_path.exists():
            tqdm.write(f"{filename} already exists, skipping download.")
            continue
        
        tqdm.write(f"Downloading {filename} from {url} ...")
        response = requests.get(url)
        response.raise_for_status()
        
        output_path.write_bytes(response.content)
        tqdm.write(f"Saved {filename} to {output_path}")

def gather_g3d():
    df = get_g3d_links_csv()
    print(f"Loaded {len(df)} rows into DataFrame")
    print(df.head())
    download_g3d_zip_files(df)

if __name__ == "__main__":
    gather_g3d()