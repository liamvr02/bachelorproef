from lst_resolve_download_urls import lst_resolve_download_urls
import requests
from typing import Optional
from pathlib import Path
from time import sleep
from tqdm import tqdm
import re


def lst_resolve_again(url: str, timeout: int = 120) -> Optional[str]:
    """
    Resolve an LST query URL to the actual download URL.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        data = response.json()

        if "download" in data:
            return data["download"]
        else:
            tqdm.write(f"[Warning] 'download' key missing for URL: {url}")
            return None

    except requests.RequestException as e:
        tqdm.write(f"[Error] Request failed for {url}: {e}")
    except ValueError:
        tqdm.write(f"[Error] Invalid JSON for {url}: {response.text}")  # type: ignore

    return None


def _get_filename(response: requests.Response, fallback: str) -> str:
    """
    Extract filename from Content-Disposition header.
    """
    cd = response.headers.get("Content-Disposition")
    if cd:
        match = re.search(r'filename="?(.+?)"?$', cd)
        if match:
            return match.group(1)
    return fallback


def download_file(
    download_url: str,
    output_dir: Path,
    fallback_name: str,
    chunk_size: int = 8192
) -> Optional[Path]:
    """
    Download a file using filename from response headers.

    Returns:
        Path if successful, None otherwise
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        with requests.get(download_url, stream=True, timeout=300) as r:
            r.raise_for_status()

            filename = _get_filename(r, fallback_name)
            output_path = output_dir / filename

            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        return output_path

    except requests.RequestException as e:
        tqdm.write(f"[Error] Download failed: {download_url} | {e}")
        return None


if __name__ == "__main__":
    use_all_images = False
    skip_existing = True  # True = skip, False = versions
    wait = 3

    download_dir = Path(__file__).parent.parent.absolute() / "downloads" / "lst_zips"
    download_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write("Fetching initial LST URLs...")
    url_dicts = lst_resolve_download_urls(use_all_images)
    tqdm.write(f"Retrieved {len(url_dicts)} URLs")

    for idx, entry in enumerate(tqdm(url_dicts, desc="Processing URLs", unit="url"), 1):

        query = entry.get("query", {})
        url = query.get("url")

        if not url:
            tqdm.write(f"[{idx}] Missing URL in entry, skipping")
            continue

        emissivity = query.get("emissivity", "UNK")
        landsat_id = query.get("landsat_id", "X")
        start = query.get("start", "unknown")
        end = query.get("end", "unknown")
        image_id = query.get("imageID", "unknown")

        prefix = f"L{landsat_id}_{emissivity}_{start.replace('/', '')}_{end.replace('/', '')}_{image_id}_"

        # Check existing files with prefix
        existing = list(download_dir.glob(f"{prefix}*"))

        if existing and skip_existing:
            tqdm.write(f"[{idx}] Skipping (already exists): {existing[0].name}")
            continue

        tqdm.write(f"[{idx}] Resolving URL...")
        resolved = lst_resolve_again(url)

        if resolved:
            tqdm.write(f"[{idx}] Downloading file...")

            output_path = download_file(
                resolved,
                download_dir,
                fallback_name=f"{prefix}download.zip"
            )

            if output_path:

                base_name = output_path.name
                base_path = download_dir / (prefix + base_name)

                # 🔁 Browser-style naming
                if not skip_existing:
                    candidate = base_path
                    counter = 1

                    while candidate.exists():
                        stem = base_path.stem
                        suffix = base_path.suffix
                        candidate = download_dir / f"{stem} ({counter}){suffix}"
                        counter += 1

                    new_path = candidate
                else:
                    new_path = base_path

                output_path.rename(new_path)

                tqdm.write(f"[{idx}] Downloaded: {new_path}")
        else:
            tqdm.write(f"[{idx}] Skipped due to resolution failure")

        sleep(wait)