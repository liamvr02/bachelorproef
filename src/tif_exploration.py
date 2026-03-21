import rasterio
from pprint import pprint

path = "./downloads/lst_tifs/L5_ASTER_20000301_20010301_LT51980242000222FUI00_20000809_101119/20000809_101119.LST.tif"

with rasterio.open(path) as src:
    print("=== Basic structure ===")
    print("Driver:", src.driver)
    print("Width, Height:", src.width, src.height)
    print("Band count:", src.count)

    print("\n=== Data types per band ===")
    print(src.dtypes)

    print("\n=== CRS ===")
    print(src.crs)

    print("\n=== Transform (pixel -> world) ===")
    print(src.transform)

    print("\n=== Bounds ===")
    print(src.bounds)

    print("\n=== Dataset-level metadata ===")
    pprint(src.meta)

    print("\n=== All metadata tags ===")
    pprint(src.tags())

    print("\n=== Namespaces of metadata ===")
    pprint(src.tag_namespaces())

    print("\n=== Metadata by namespace ===")
    for ns in src.tag_namespaces():
        print(f"\nNamespace: {ns}")
        pprint(src.tags(ns=ns))

    print("\n=== Band-level metadata ===")
    for i in range(1, src.count + 1):
        print(f"\nBand {i}")
        print("dtype:", src.dtypes[i-1])
        pprint(src.tags(i))