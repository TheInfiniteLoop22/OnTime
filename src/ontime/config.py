"""Config loader and logging setup.

Why: one place reads `configs/config.yaml` so no path or parameter is hardcoded elsewhere.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _resolve_config_path(path: str | None) -> Path:
    """Explicit argument > ONTIME_CONFIG environment variable > configs/config.yaml.

    The environment override lets the integration tests (and CI) run the real pipeline on a tiny
    fixture dataset with different paths and dates, without touching real data.
    """
    if path:
        return Path(path)
    env = os.environ.get("ONTIME_CONFIG")
    return Path(env) if env else CONFIG_PATH


@lru_cache(maxsize=8)
def _load(path_str: str) -> dict[str, Any]:
    with Path(path_str).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load (and cache per file) the YAML config."""
    return _load(str(_resolve_config_path(path)))


def project_path(relative: str) -> Path:
    """Resolve a config path against the repo root (absolute paths are returned unchanged)."""
    return PROJECT_ROOT / relative


def setup_logging(level: int = logging.INFO) -> None:
    """Configure consistent logging for all pipeline stages."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("matplotlib", "PIL", "interpret", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
