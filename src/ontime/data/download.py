"""Download the Olist dataset via the Kaggle API.

Why: reproducibility. Anyone with Kaggle credentials can regenerate `data/raw` from scratch.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ontime.config import load_config, project_path

log = logging.getLogger(__name__)

CREDENTIALS_HELP = (
    "Kaggle credentials not found. Create an API token at kaggle.com -> Settings -> API, then "
    "save it as ~/.kaggle/kaggle.json (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json), or set "
    "KAGGLE_USERNAME and KAGGLE_KEY. Alternatively, download the zip manually from "
    "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and unzip the 9 CSVs "
    "into data/raw/."
)


def missing_files(raw_dir: Path, expected: list[str]) -> list[str]:
    """Return expected CSV names not present in raw_dir."""
    return [name for name in expected if not (raw_dir / name).exists()]


def download(force: bool = False) -> Path:
    """Download and unzip the dataset into raw_dir unless all files are already there."""
    cfg = load_config()
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected = list(cfg["raw_files"].values())

    if not force and not missing_files(raw_dir, expected):
        log.info(
            "All %d raw files already present in %s; skipping download.", len(expected), raw_dir
        )
        return raw_dir

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception as exc:  # kaggle raises OSError/IOError when creds are missing
        raise RuntimeError(CREDENTIALS_HELP) from exc

    log.info("Downloading %s to %s", cfg["kaggle"]["dataset"], raw_dir)
    api.dataset_download_files(cfg["kaggle"]["dataset"], path=str(raw_dir), unzip=True)

    still_missing = missing_files(raw_dir, expected)
    if still_missing:
        raise FileNotFoundError(f"Download finished but files are missing: {still_missing}")
    return raw_dir
