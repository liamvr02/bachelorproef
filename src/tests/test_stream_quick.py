"""Quick test of LST stream setup without full data streaming."""

from stream.lst_stream import LSTStream, make_trees_within_radius, make_height_statistics, make_urban_atlas_features
from tqdm import tqdm

tqdm.write("\n" + "="*60)
tqdm.write("LST Stream Quick Test")
tqdm.write("="*60)
tqdm.write("\nInitializing stream...")

try:
    stream = LSTStream()
    
    tqdm.write("\nRegistering features...")
    stream.register_feature(
        "trees_within_100m",
        make_trees_within_radius(100),
        depends_on=["longitude", "latitude", "timestamp"],
        description="Count of trees planted before timestamp within 100m radius"
    )
    
    stream.register_feature(
        "height_neighborhood_50m",
        make_height_statistics(radius_m=50),
        depends_on=["longitude", "latitude"],
        description="Height statistics (mean, std, max, min) within 50m radius"
    )
    
    stream.register_feature(
        "urban_atlas_2021",
        make_urban_atlas_features(2021),
        depends_on=["longitude", "latitude"],
        description="Land use code from Urban Atlas 2021"
    )
    
    tqdm.write("\n✓ Registered features:")
    info = stream.get_feature_info()
    print(info.to_string())
    
    tqdm.write("\n" + "="*60)
    tqdm.write("✓ Quick test complete")
    tqdm.write("="*60)
    
finally:
    tqdm.write("\nClosing stream...")
    stream.close()
    tqdm.write("✓ Stream closed successfully")
