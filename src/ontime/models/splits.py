"""Time-based train/validation/test assignment from config (never random).

Boundaries are whole days on `order_approved_at`: train < val_start <= validation < test_start
<= test. Phase 5 builds evaluation helpers on top of this.
"""

from __future__ import annotations

import pandas as pd

from ontime.config import load_config


def split_boundaries(cfg: dict | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (val_start, test_start) after asserting the config has no gap or overlap."""
    s = (cfg or load_config())["splits"]
    one_day = pd.Timedelta(days=1)
    train_end, val_start = pd.Timestamp(s["train_end"]), pd.Timestamp(s["val_start"])
    val_end, test_start = pd.Timestamp(s["val_end"]), pd.Timestamp(s["test_start"])
    if train_end + one_day != val_start:
        raise ValueError("train_end and val_start must be consecutive days (no gap/overlap)")
    if val_end + one_day != test_start:
        raise ValueError("val_end and test_start must be consecutive days (no gap/overlap)")
    return val_start, test_start


def assign_split(approved_at: pd.Series, cfg: dict | None = None) -> pd.Series:
    """Label each order 'train' / 'validation' / 'test' by approval time."""
    val_start, test_start = split_boundaries(cfg)
    out = pd.Series("train", index=approved_at.index, dtype="object")
    out[approved_at >= val_start] = "validation"
    out[approved_at >= test_start] = "test"
    return out
