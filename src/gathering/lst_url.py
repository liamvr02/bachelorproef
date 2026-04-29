from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
import json
from typing import List, Optional, Tuple
from urllib.parse import urlencode
from tqdm import tqdm


JSON_FILE = Path(__file__).parent.absolute() / "landsatlst.json"
BASE_URL = "https://landsatlst.appspot.com"


def get_lst_url(
    polygon_coords: List[Tuple[float, float]],
    emissivity: str,
    landsat: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Build the download URL for a satellite image.

    Args:
        polygon_coords (List[Tuple[float, float]]): Polygon coordinates [(lon, lat), ...]
        emissivity (str): Emissivity type (NDVI, MODIS, ASTER).
        landsat (str): Landsat version ("5","7","8","9").
        start_date (str): Start date YYYY/MM/DD.
        end_date (str): End date YYYY/MM/DD.

    Returns:
        str: Direct download URL.
    """

    if not polygon_coords:
        raise ValueError("Polygon coordinates are required.")

    polygon_str = ",".join(f"[{lon},{lat}]" for lon, lat in polygon_coords)

    params = {
        "polygon": polygon_str,
        "startDate": start_date,
        "endDate": end_date,
    }

    query_string = urlencode(params)

    return f"{BASE_URL}/Landsat{landsat}{emissivity}?{query_string}"


def get_lst_urls(
    polygon_coords,
    period: str = "yearly",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[dict]:
    """
    Generate Landsat LST query URLs with calendar-aligned intervals.

    Args:
        polygon_coords: Polygon coordinates passed to get_lst_url.
        period: "yearly", "quarterly", or "monthly".
        start_date: Optional filter start (YYYY/MM/DD).
        end_date: Optional filter end (YYYY/MM/DD).

    Returns:
        List of URL query dictionaries.
    """

    period_map = {
        "yearly": relativedelta(years=1),
        "quarterly": relativedelta(months=3),
        "monthly": relativedelta(months=1),
    }

    if period not in period_map:
        raise ValueError("period must be 'yearly', 'quarterly', or 'monthly'")

    delta = period_map[period]

    url_dicts: List[dict] = []

    start_filter = datetime.strptime(start_date, "%Y/%m/%d") if start_date else None
    end_filter = datetime.strptime(end_date, "%Y/%m/%d") if end_date else None

    with open(JSON_FILE, "r") as f:
        landsat_json = json.load(f)

    landsats = landsat_json.get("landsats", [])

    for landsat in tqdm(landsats, desc="Processing Landsats", unit="landsat"):

        landsat_from = datetime.strptime(landsat["from"], "%Y/%m/%d")
        landsat_to = datetime.strptime(landsat["to"], "%Y/%m/%d")

        range_start = max(landsat_from, start_filter) if start_filter else landsat_from
        range_end = min(landsat_to, end_filter) if end_filter else landsat_to

        current_start = range_start

        tqdm.write(
            f"Landsat {landsat['id']} | Range: {range_start.date()} -> {range_end.date()} | "
            f"Emissivities: {landsat.get('emissivities', [])}"
        )

        while current_start <= range_end:

            next_start = current_start + delta
            current_end = min(next_start, range_end)

            start_str = current_start.strftime("%Y/%m/%d")
            end_str = current_end.strftime("%Y/%m/%d")

            for emissivity in landsat.get("emissivities", []):

                url = get_lst_url(
                    polygon_coords,
                    emissivity,
                    landsat["id"],
                    start_str,
                    end_str
                )

                url_dicts.append({
                    "url": url,
                    "polygon_coords": polygon_coords,
                    "emissivity": emissivity,
                    "landsat_id": landsat["id"],
                    "start": start_str,
                    "end": end_str
                })

            current_start = next_start

    tqdm.write(f"Generated {len(url_dicts)} total LST query URLs")

    return url_dicts