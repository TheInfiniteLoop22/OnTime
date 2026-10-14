"""Config loader and logging setup.

Why: one place reads `configs/config.yaml` so no path or parameter is hardcoded elsewhere.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load the YAML config once and cache it."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """Resolve a project-relative path from config against the repo root."""
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
