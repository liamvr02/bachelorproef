from pathlib import Path
import zipfile
from tqdm import tqdm


def unzip_all_zips(src_folder: str | Path, dst_folder: str | Path) -> None:
    """
    Unzip all .zip files from src_folder into dst_folder.

    - Creates dst_folder if it does not exist.
    - Extracts each zip into a subfolder named after the zip file.
    - Uses tqdm for progress reporting.
    """
    src = Path(src_folder)
    dst = Path(dst_folder)

    if not src.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    zip_files = list(src.glob("*.zip"))

    if not zip_files:
        tqdm.write("No zip files found.")
        return

    for zip_path in tqdm(zip_files, desc="Extracting zip files", unit="zip"):
        extract_dir = dst / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        tqdm.write(f"Extracting {zip_path} -> {extract_dir}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    tqdm.write("Done extracting all zip files.")


if __name__ == "__main__":
    downloads_folder = Path(__file__).parent.parent.absolute() / "downloads"
    src_folder = downloads_folder / "lst_zips"
    dst_folder = downloads_folder / "lst_tifs"

    unzip_all_zips(src_folder, dst_folder)