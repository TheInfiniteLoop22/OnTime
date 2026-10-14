"""Model-ready data frames: time-based sub-splits and a stable category contract.

Sub-splits of the validation set (all by time, never random):

    train | validation = [ val_tune | val_calib = ( calib_fit | calib_eval ) ] | test (locked)

* val_tune   : hyperparameter search and early stopping.
* calib_fit  : fit calibrators.
* calib_eval : honest check of calibrators (scored by a calibrator fit on calib_fit).
Hyperparameters never see val_calib, and calibrators never see val_tune.
"""

from __future__ import annotations

import pandas as pd

from ontime.config import load_config, project_path
from ontime.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from ontime.models.splits import assert_no_time_overlap, select_split


def load_feature_table() -> pd.DataFrame:
    """Read data/processed/features.parquet and verify split ordering."""
    t = pd.read_parquet(project_path(load_config()["paths"]["features_parquet"]))
    assert_no_time_overlap(t)
    return t


def category_levels(table: pd.DataFrame) -> dict[str, list[str]]:
    """Category levels per categorical feature, taken from the full table (stable contract)."""
    return {c: list(table[c].cat.categories) for c in CATEGORICAL_FEATURES}


def prepare_frame(df: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    """Feature columns in canonical order with categorical dtypes fixed to saved levels.

    Why: LightGBM requires identical category sets at fit and predict time; a single new order
    scored in the app must go through the same function as the training table.
    """
    X = df[FEATURE_COLUMNS].copy()
    for c, cats in levels.items():
        X[c] = pd.Categorical(X[c].astype(object), categories=cats)
    return X


def modelling_frames(table: pd.DataFrame, cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """train / validation / val_tune / val_calib / calib_fit / calib_eval (test not included)."""
    cfg = cfg or load_config()
    t = cfg["tuning"]
    train = select_split(table, "train")
    val = select_split(table, "validation")
    day = pd.Timedelta(days=1)
    tune_end = pd.Timestamp(t["val_tune_end"]) + day
    fit_end = pd.Timestamp(t["calib_fit_end"]) + day
    frames = {
        "train": train,
        "validation": val,
        "val_tune": val[val["approved_at"] < tune_end],
        "val_calib": val[val["approved_at"] >= tune_end],
    }
    frames["calib_fit"] = frames["val_calib"][frames["val_calib"]["approved_at"] < fit_end]
    frames["calib_eval"] = frames["val_calib"][frames["val_calib"]["approved_at"] >= fit_end]
    for name, f in frames.items():
        if len(f) == 0:
            raise ValueError(f"empty sub-split {name}; check tuning dates in config")
    return frames
