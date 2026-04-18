"""
survey_sources.py
=================
Reports the actual columns available in every source dataset used by ingest.py.
Run this before ingestion to verify column mappings.

Usage:
    python processing/survey_sources.py
    python processing/survey_sources.py --downloads /path/to/downloads
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

_HERE      = Path(__file__).resolve().parent
_SRC       = _HERE.parent
_DOWNLOADS = _SRC / "downloads"

# ── helpers ──────────────────────────────────────────────────────────────────

def _sep(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def _col_report(columns: list[str], non_null: dict[str, int], total: int) -> None:
    """Print column name, non-null count, and null count."""
    print(f"  {'Column':<35} {'Non-null':>10}  {'Null':>10}  {'%full':>7}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*7}")
    for col in columns:
        nn   = non_null.get(col, "?")
        null = (total - nn) if isinstance(nn, int) else "?"
        pct  = f"{nn/total*100:.1f}%" if isinstance(nn, int) and total else "?"
        print(f"  {col:<35} {str(nn):>10}  {str(null):>10}  {pct:>7}")
    print(f"\n  Total rows: {total:,}")


# ── Trees CSV ─────────────────────────────────────────────────────────────────

def survey_trees(downloads: Path) -> None:
    _sep("Trees CSV")

    candidates = sorted(downloads.rglob("ghent_trees_*.csv"))
    if not candidates:
        print("  [NOT FOUND] No ghent_trees_*.csv under", downloads)
        return

    path = max(candidates, key=lambda p: p.name)
    print(f"  File: {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")

    # Detect delimiter from first 4 KB
    with open(path, encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    delimiter = ";" if sample.count(";") > sample.count(",") else ","
    print(f"  Detected delimiter: {repr(delimiter)}")

    # Read just the header + a sample to check column names
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        columns = reader.fieldnames or []
        print(f"\n  Columns ({len(columns)}):")
        for c in columns:
            print(f"    {c}")

        # Count non-nulls across up to 5000 rows for a quick profile
        non_null: dict[str, int] = {c: 0 for c in columns}
        total = 0
        for row in reader:
            total += 1
            for col in columns:
                v = row.get(col, "")
                if v and v.strip().lower() not in ("", "nan", "none", "null"):
                    non_null[col] += 1
            if total >= 5000:
                break

    print(f"\n  Non-null profile (first {total:,} rows):")
    _col_report(columns, non_null, total)

    # Print the geo_point_2d sample if present
    if "geo_point_2d" in columns:
        with open(path, encoding="utf-8", errors="replace") as f:
            reader2 = csv.DictReader(f, delimiter=delimiter)
            for i, row in enumerate(reader2):
                v = row.get("geo_point_2d", "")
                if v:
                    print(f"\n  geo_point_2d sample: {v!r}")
                    break


# ── Urban Atlas ───────────────────────────────────────────────────────────────

def survey_urban_atlas(downloads: Path) -> None:
    _sep("Urban Atlas")
    try:
        import geopandas as gpd
    except ImportError:
        print("  [SKIP] geopandas not available")
        return

    ua_root = downloads / "urban_atlas_extracted"
    if not ua_root.exists():
        print(f"  [NOT FOUND] {ua_root}")
        return

    UA_SOURCES = {
        2006: {"globs": ["BE003L2_GENT/*/Shapefiles/*.shp"],             "layer": None},
        2012: {"globs": ["BE003L2_GENT_UA2012_revised_v021/*/Data/*.gpkg"], "layer": "BE003L2_GENT_UA2012_revised"},
        2018: {"globs": ["*/CLMS_UA_LCU_S2018*/*.fgb"],                  "layer": None},
        2021: {"globs": ["*/CLMS_UA_LCU_S2021*/*.fgb"],                  "layer": None},
    }

    for year, spec in UA_SOURCES.items():
        source = None
        for glob in spec["globs"]:
            hits = list(ua_root.glob(glob))
            if hits:
                source = hits[0]
                break
        if source is None:
            print(f"\n  UA {year}: [NOT FOUND] (globs: {spec['globs']})")
            continue

        print(f"\n  UA {year}: {source.name}  ({source.stat().st_size / 1e6:.1f} MB)")
        try:
            read_kwargs = {"layer": spec["layer"]} if spec["layer"] else {}
            gdf = gpd.read_file(str(source), rows=200, **read_kwargs)
            non_null = {c: int(gdf[c].notna().sum()) for c in gdf.columns if c != "geometry"}
            _col_report(
                [c for c in gdf.columns if c != "geometry"],
                non_null,
                len(gdf),
            )
        except Exception as e:
            print(f"  [ERROR] {e}")


# ── WIS GeoJSON ───────────────────────────────────────────────────────────────

def survey_wis(downloads: Path) -> None:
    _sep("WIS GeoJSON")
    path = downloads / "wis" / "wis.geojson"
    if not path.exists():
        print(f"  [NOT FOUND] {path}")
        return

    print(f"  File: {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")

    # Stream through features without loading the whole file
    # Count properties and non-null values across first 1000 features
    prop_counts: dict[str, int] = {}
    geom_types: dict[str, int] = {}
    total = 0
    null_geom = 0

    with open(path, encoding="utf-8") as f:
        # Fast line-by-line scan for "properties" objects
        # Use json.JSONDecoder to parse incrementally if needed,
        # but for a 10MB+ file just load it (it's read-once at ingest anyway)
        try:
            data = json.load(f)
        except Exception as e:
            print(f"  [ERROR reading JSON] {e}")
            return

    features = data.get("features", [])
    sample_limit = min(1000, len(features))
    for feat in features[:sample_limit]:
        total += 1
        geom = feat.get("geometry")
        if geom is None:
            null_geom += 1
            gt = "NULL"
        else:
            gt = geom.get("type", "unknown")
        geom_types[gt] = geom_types.get(gt, 0) + 1

        props = feat.get("properties") or {}
        for k, v in props.items():
            if k not in prop_counts:
                prop_counts[k] = 0
            if v is not None and str(v).strip().lower() not in ("", "nan", "none", "null"):
                prop_counts[k] += 1

    print(f"\n  Total features in file: {len(features):,}")
    print(f"  Sample size: {total:,}")
    print(f"\n  Geometry types in sample:")
    for gt, cnt in sorted(geom_types.items()):
        print(f"    {gt}: {cnt:,}")

    print(f"\n  Properties non-null profile (sample of {total:,}):")
    all_props = sorted(prop_counts.keys())
    _col_report(all_props, prop_counts, total)

    # Show a few unique values per property
    print(f"\n  Sample values per property (up to 5 unique):")
    unique: dict[str, set] = {k: set() for k in all_props}
    for feat in features[:sample_limit]:
        props = feat.get("properties") or {}
        for k in all_props:
            v = props.get(k)
            if v is not None and str(v).strip().lower() not in ("", "nan", "none", "null"):
                unique[k].add(str(v)[:60])
                if len(unique[k]) >= 5:
                    pass  # keep collecting up to 5
    for k in all_props:
        vals = sorted(unique[k])[:5]
        print(f"    {k}: {vals}")


# ── LST folder names (emissivity field check) ─────────────────────────────────

def survey_lst(downloads: Path) -> None:
    _sep("LST folders (emissivity field check)")
    lst_root = downloads / "lst_tifs"
    if not lst_root.exists():
        print(f"  [NOT FOUND] {lst_root}")
        return

    folders = [p.name for p in lst_root.iterdir() if p.is_dir()]
    print(f"  Total folders: {len(folders):,}")

    emissivity_vals: dict[str, int] = {}
    product_vals: dict[str, int] = {}
    parse_failures = 0

    import re
    _FOLDER_RE = re.compile(
        r"^(?P<sat>L\w+)_(?P<product>[A-Z]+)_\d{8}_\d{8}_"
        r"(?P<prod_id>\w+)_(?P<date>\d{8})_(?P<time>\d{6})$"
    )

    for name in folders:
        parts = name.split("_")
        if len(parts) < 7:
            parse_failures += 1
            continue
        emissivity = parts[1]
        emissivity_vals[emissivity] = emissivity_vals.get(emissivity, 0) + 1

        m = _FOLDER_RE.match(name)
        if m:
            product = m.group("product").upper()
            product_vals[product] = product_vals.get(product, 0) + 1
        else:
            parse_failures += 1

    print(f"  Parse failures: {parse_failures}")
    print(f"\n  Emissivity values (split[1]) — {len(emissivity_vals)} unique:")
    for v, cnt in sorted(emissivity_vals.items(), key=lambda x: -x[1]):
        print(f"    {v:<20} {cnt:>6,} folders")
    print(f"\n  Product values (regex group) — {len(product_vals)} unique:")
    for v, cnt in sorted(product_vals.items(), key=lambda x: -x[1]):
        print(f"    {v:<20} {cnt:>6,} folders")
    print(f"\n  Sample folder names:")
    for name in sorted(folders)[:5]:
        parts = name.split("_")
        print(f"    {name}")
        print(f"      → split: {parts}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Survey source datasets for ingest.py")
    parser.add_argument("--downloads", type=Path, default=_DOWNLOADS,
                        help=f"Downloads directory (default: {_DOWNLOADS})")
    parser.add_argument("--only", nargs="+",
                        choices=["trees", "urban_atlas", "wis", "lst"],
                        default=["trees", "urban_atlas", "wis", "lst"],
                        help="Which datasets to survey")
    args = parser.parse_args()

    dl = args.downloads
    print(f"Downloads: {dl}")
    print(f"Surveying: {args.only}")

    if "trees"       in args.only: survey_trees(dl)
    if "urban_atlas" in args.only: survey_urban_atlas(dl)
    if "wis"         in args.only: survey_wis(dl)
    if "lst"         in args.only: survey_lst(dl)

    print()
    print("=" * 70)
    print("  Survey complete.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()