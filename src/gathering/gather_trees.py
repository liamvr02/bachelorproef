import requests
from pathlib import Path
import pandas as pd
from datetime import date
import os

# Direct CSV download URL
CSV_URL = (
    "https://data.stad.gent/api/explore/v2.1/catalog/datasets/"
    "locaties-bomen-gent/exports/csv?lang=en&timezone=Europe%2FBrussels&use_labels=true&delimiter=%3B"
)

# Output CSV path
OUTPUT_FILE = Path(__file__).parent.parent.absolute() / "downloads" / "trees" / f"ghent_trees_{date.today()}.csv"

def download_trees_csv(url: str = CSV_URL, output_file: Path = OUTPUT_FILE):
    print(f"Downloading CSV from {url} ...")
    response = requests.get(url)
    response.raise_for_status()

    output_file.write_bytes(response.content)
    print(f"Saved CSV to {output_file}")

def get_trees_csv():
    if (os.path.exists(OUTPUT_FILE)):
        print(f"{OUTPUT_FILE} already exists")
    else:
        download_trees_csv(CSV_URL, OUTPUT_FILE)
    
    df = pd.read_csv(OUTPUT_FILE, delimiter=";")
    return df

def main():
    df = get_trees_csv()
    print(f"Loaded {len(df)} rows into DataFrame")
    print(df.head())

if __name__ == "__main__":
    main()