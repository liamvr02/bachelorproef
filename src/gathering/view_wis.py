import ijson
from collections import defaultdict

def inspect_geojson(filepath, max_features=None):
    geometry_types = defaultdict(int)
    property_keys = defaultdict(int)
    property_types = defaultdict(set)
    total_features = 0

    with open(filepath, 'rb') as f:
        # Stream features from GeoJSON
        features = ijson.items(f, 'features.item')

        for feature in features:
            total_features += 1

            # --- Geometry (safe) ---
            geometry = feature.get('geometry')
            if geometry and isinstance(geometry, dict):
                geom_type = geometry.get('type')
                if geom_type:
                    geometry_types[geom_type] += 1
            else:
                geometry_types['NULL'] += 1

            # --- Properties (safe) ---
            props = feature.get('properties') or {}
            if isinstance(props, dict):
                for key, value in props.items():
                    property_keys[key] += 1
                    property_types[key].add(type(value).__name__)

            # Optional: stop early if sampling
            if max_features and total_features >= max_features:
                break

            if total_features % 10000 == 0:
                print(f"Processed {total_features} features...")

    # Print results
    print("\n=== SUMMARY ===")
    print(f"Total features processed: {total_features}")

    print("\nGeometry types:")
    for k, v in geometry_types.items():
        print(f"  {k}: {v}")

    print("\nProperty keys (with counts):")
    for k, v in sorted(property_keys.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print("\nProperty types:")
    for k, v in property_types.items():
        print(f"  {k}: {', '.join(v)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a large GeoJSON file")
    parser.add_argument("filepath", help="Path to GeoJSON file")
    parser.add_argument("--max", type=int, help="Max features to process (for sampling)", default=None)

    args = parser.parse_args()

    inspect_geojson(args.filepath, args.max)