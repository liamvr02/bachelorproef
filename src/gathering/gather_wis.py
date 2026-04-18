import requests
from pathlib import Path

DOWNLOAD_URL = "https://data.stad.gent/api/explore/v2.1/catalog/datasets/wis-gent/exports/geojson?lang=en&timezone=Europe/Brussels"
DST_DIR = Path(__file__).parent.parent.absolute() / "downloads" / "wis"

def gather_wis():
    response = requests.get(DOWNLOAD_URL)
    if response.status_code == 200:
        DST_DIR.mkdir(parents=True, exist_ok=True)
        with open(f"{DST_DIR}/wis.geojson", "w") as f:
            f.write(response.text)
    else:
        print(f"Failed to download WIS data. Status code: {response.status_code}")

if __name__ == "__main__":
    gather_wis()