import json
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode
import copy

BASE_URL = "https://landsatlst.appspot.com"
LST_RESPONSES = Path(__file__).parent.absolute() / "lst_responses.jsonl"

def lst_req_resp_to_urls(req_resp: Dict[str, dict], use_all_images: bool = True) -> List[dict]:
    polygon_coords = req_resp["query"]["polygon_coords"]
    polygon_str = ",".join(f"[{lon},{lat}]" for lon, lat in polygon_coords)

    emissivity = req_resp["query"]["emissivity"]
    landsat = req_resp["query"]["landsat_id"]

    base_query = req_resp["query"]

    if use_all_images:
        params = {
            "polygon": polygon_str,
            "startDate": base_query["start"],
            "endDate": base_query["end"],
            "imageID": "AllImages"
        }

        query_string = urlencode(params)

        return [{
            "query": {
                **base_query,
                "imageID": "AllImages",
                "url": f"{BASE_URL}/{emissivity}{landsat}Download?{query_string}"
            },
            "response": None
        }]

    urls = []

    for v in req_resp["response"].values():
        image_id = v["imageID"]

        params = {
            "polygon": polygon_str,
            "startDate": base_query["start"],
            "endDate": base_query["end"],
            "imageID": image_id
        }

        query_string = urlencode(params)

        urls.append({
            "query": {
                **base_query,
                "imageID": image_id,
                "url": f"{BASE_URL}/{emissivity}{landsat}Download?{query_string}"
            },
            "response": None
        })

    return urls

def lst_resolve_download_urls(use_all_images: bool = True):
    urls = []
    with open(LST_RESPONSES) as f:
        for line in f.readlines():
            req_resp = json.loads(line)
            resolved = lst_req_resp_to_urls(req_resp, use_all_images)
            urls.extend(resolved)
    return urls

if __name__ == "__main__":
    urls = lst_resolve_download_urls()
    print(f"{len(urls)} urls")
    print(urls[0])