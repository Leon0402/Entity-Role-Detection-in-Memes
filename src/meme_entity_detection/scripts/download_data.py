from pathlib import Path
import argparse
import zipfile
import logging

import requests
from tqdm import tqdm


def download_and_extract_zip(download_url: str, output_path: Path) -> None:
    """
    Downloads a ZIP file from the specified URL and extracts it to the given output directory.

    Args:
    - download_url: The URL to download the ZIP file from.
    - output_path: The path to the directory where the contents should be extracted.

    Raises:
    - Exception: If the download or extraction fails.
    """
    with open(output_path / "data.zip", 'wb') as f:
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()

            tqdm_params = {
                'desc': "Download Data",
                'total': int(r.headers.get('content-length', 0)),
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)

    with zipfile.ZipFile(output_path / "data.zip") as zip_file:
        zip_file.extractall(output_path)

    logging.info(f"Files successfully extracted to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract a ZIP file from a URL to a specified directory.")
    parser.add_argument('--download-url', type=str, help="The URL to download the ZIP file from.")
    parser.add_argument(
        '--output-path', type=Path, help="The path to the directory where the contents should be extracted."
    )
    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)
    download_and_extract_zip(args.download_url, args.output_path)


if __name__ == "__main__":
    main()
