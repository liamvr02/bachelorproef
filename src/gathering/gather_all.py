from gather_trees import download_trees_csv
from lst_resolve_urls import download_lst_urls
from gather_lst import gather_lst

MSG = """
As not to overly exert the landsat data server, the script does not make use of parallelization. 
Meaning this script will take a long time to collect all landsat data from 2000-2026.
During this time, please keep the computer active and connected to the internet.

This process may take up to 8 hours, are you sure you want to continue? [y/n]
"""

def gather_all():
    ans = input(MSG)

    if ans not in ["y", "Y"]:
        print("Script cancelled")
        return
    
    print("Downloading tree data...")
    download_trees_csv()
    print("Preparing LST...")
    download_lst_urls()
    print("Gathering LST...")
    gather_lst()
    print("\n\nScript finished!\n")

