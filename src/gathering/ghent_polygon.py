"""
This script defines OpenStreetMap API calls for the city Ghent including some functions that return simplified versions, 
to get coordinates for use with other API's, such as LandsatLST, use get_ghent_convex_hull.
"""

from typing import List, Tuple

from OSMPythonTools.overpass import Overpass
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, polygonize
from math import cos, radians

overpass = Overpass()

def get_ghent_outers() -> List[Tuple[float, ...]]:
    query = """
    rel(897671);
    way(r:"outer");
    out geom;
    """

    result = overpass.query(query)

    assert result is not None, f"No results for {query}"

    data = result.toJSON()

    outer_lines = []

    for el in data["elements"]:
        if el["type"] != "way":
            continue

        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]

        if len(coords) > 1:
            outer_lines.append(LineString(coords))

    merged = unary_union(outer_lines)
    polygons = list(polygonize(merged))

    if not polygons:
        raise RuntimeError("Could not reconstruct polygon")

    poly = max(polygons, key=lambda p: p.area)

    return list(poly.exterior.coords)


def simplify_containing_polygon(coords, tolerance, buffer_amount):
    poly = Polygon(coords)

    simplified = poly.simplify(tolerance, preserve_topology=True)

    if not simplified.contains(poly):
        simplified = simplified.buffer(buffer_amount)

    return simplified


def get_ghent_outers_simplified(n: int, tolerance: float, buffer_amount: float) -> List[Tuple[float, ...]]:
    coords = get_ghent_outers()

    poly = Polygon(coords)

    tol = tolerance

    while True:
        simplified = simplify_containing_polygon(
            list(poly.exterior.coords), tol, buffer_amount
        )

        if simplified.geom_type != "Polygon":
            simplified = max(simplified.geoms, key=lambda g: g.area) # type: ignore

        result_coords = list(simplified.exterior.coords) # type: ignore

        if len(result_coords) <= n:
            return result_coords

        tol *= 1.5


def get_ghent_convex_hull() -> List[Tuple[float, ...]]:
    coords = get_ghent_outers()
    poly = Polygon(coords)

    hull = poly.convex_hull

    return list(hull.exterior.coords) # type: ignore


def get_ghent_min_bounding_rectangle() -> List[Tuple[float, ...]]:
    coords = get_ghent_outers()
    poly = Polygon(coords)

    rect = poly.minimum_rotated_rectangle

    return list(rect.exterior.coords) # type: ignore


def lon_lat_aspect(coords: List[Tuple[float, ...]]) -> float:
    """
    Correct aspect ratio for lon/lat plots.
    """
    mean_lat = sum(lat for _, lat in coords) / len(coords)
    return 1 / cos(radians(mean_lat))