"""
This script tests ghent_polygon by saving comparison plots to ghent_polygon.test_results.png
From this and the assumed workings of LandsatLST, convex_hull seems to be the preferred method for data gathering.
For processing in machine learning relevant to the city Ghent, this output should be further pruned to only include
areas with overlapping data in other datasets. 
"""

from pathlib import Path

import matplotlib.pyplot as plt

from ghent_polygon import (
    get_ghent_outers,
    get_ghent_outers_simplified,
    get_ghent_convex_hull,
    get_ghent_min_bounding_rectangle,
    lon_lat_aspect,
)


def split_coords(coords):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return xs, ys


def plot_compare(ax, original, simplified, title):
    ox, oy = split_coords(original)
    sx, sy = split_coords(simplified)

    ax.plot(ox, oy, label=f"Original ({len(original)})")
    ax.plot(sx, sy, label=f"Simplified ({len(simplified)})")

    ax.set_title(title)
    ax.legend()


def main():
    original = get_ghent_outers()

    simplified = get_ghent_outers_simplified(
        n=100,
        tolerance=0.0001,
        buffer_amount=0.0001,
    )

    convex = get_ghent_convex_hull()

    rect = get_ghent_min_bounding_rectangle()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_compare(
        axes[0],
        original,
        simplified,
        "Tolerance Simplification",
    )

    plot_compare(
        axes[1],
        original,
        convex,
        "Convex Hull",
    )

    plot_compare(
        axes[2],
        original,
        rect,
        "Minimum Bounding Rectangle",
    )

    aspect = lon_lat_aspect(original)

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect(aspect)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent.absolute() / "ghent_polygon.test_results.png")


if __name__ == "__main__":
    main()