from gather_trees import download_trees_csv
from gathering.gather_g3d import gather_dwg
from lst_resolve_urls import download_lst_urls
from gather_lst import gather_lst
from gather_wis import gather_wis
from unzip import unzip_lst, unzip_dwg

CONFIRMATIONS = [
"""
Before running this script, please ensure the following:
- You have a stable internet connection for the duration of the script.
- You have sufficient disk space to store the downloaded data (potentially tens of GBs).
- You have downloaded the DHM zips into downloads/DHM1_zips and downloads/DHM2_zips (see readme).

If you have ensured all of the above, please enter 'y' to continue. [y/n]
""",
"""
As not to overly exert the landsat data server, the script does not make use of parallelization. 
Meaning this script will take a long time to collect all landsat data from 2000-2026.
During this time, please keep the computer active and connected to the internet.

This process may take up to 8 hours, are you sure you want to continue? [y/n]
"""
]

def gather_all():
    for msg in CONFIRMATIONS:
        ans = input(msg)

        if ans not in ["y", "Y"]:
            print("Script cancelled")
            return

    if ans not in ["y", "Y"]:
        print("Script cancelled")
        return
    
    print("Downloading tree data...")
    download_trees_csv()
    print("Preparing LST...")
    download_lst_urls()
    print("Gathering LST...")
    gather_lst()
    print("Unzipping LST...")
    unzip_lst()
    print("Gathering DWG...")
    gather_dwg()
    print("Unzipping DWG...")
    unzip_dwg()
    print("Gathering WIS data...")
    gather_wis()
    print("\n\nScript finished!\n")


if __name__ == "__main__":
    gather_all()
