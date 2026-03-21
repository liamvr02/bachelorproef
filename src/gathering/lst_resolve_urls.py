import requests
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import date, timedelta
from time import sleep

from tqdm import tqdm

from ghent_polygon import get_ghent_convex_hull
from lst_url import get_lst_urls


def resolve_lst_urls(url_dicts: List[dict], output_file: Path, tsleep: int = 30):
    """
    Resolve Landsat URLs and append responses to a JSONL file.

    Each line format:
    {"url": url, "response": response_json_or_error}
    """

    tqdm.write("Preparing output directory...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"Appending results to: {output_file}")

    with open(output_file, "a", encoding="utf-8") as f:
        for i, url_dict in enumerate(
            tqdm(url_dicts, desc="Resolving LST URLs", unit="req"),
            start=1
        ):
            entry: Dict[str, Any] = {"query": url_dict}

            try:
                response = requests.get(url_dict["url"], timeout=120)

                try:
                    entry["response"] = response.json()
                except ValueError:
                    entry["response"] = {
                        "error": "invalid_json",
                        "status_code": response.status_code,
                        "text": response.text
                    }

            except requests.RequestException as e:
                entry["response"] = {
                    "error": "request_exception",
                    "message": str(e)
                }

            # Append immediately
            f.write(json.dumps(entry) + "\n")
            f.flush()

            tqdm.write(f"[{i}/{len(url_dicts)}] Saved result")

            sleep(tsleep)


def download_lst_urls():
    tqdm.write("Generating Ghent polygon...")
    polygon_coords = get_ghent_convex_hull()

    tqdm.write("Preparing URL queries...")
    tdelta = timedelta(days=365)
    url_dicts = get_lst_urls(polygon_coords, "yearly", start_date="2000/01/01")

    tqdm.write(f"Total queries generated: {len(url_dicts)}")

    fp = Path(__file__).parent.parent.absolute() / "downloads" / f"lst_responses.jsonl"

    resolve_lst_urls(url_dicts, fp, tsleep=1)

if __name__ == "__main__":
    download_lst_urls()