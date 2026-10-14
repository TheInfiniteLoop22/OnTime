"""Scoring + explanation service used by the Streamlit app (and a future API).

Why in `src/`: CLAUDE.md rule 6. The app only calls these functions, so live scoring, reason
codes and the what-if recomputation are unit-testable without Streamlit.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ontime.config import load_config, project_path
from ontime.explain.contributions import EXTRACTORS
from ontime.explain.reason_codes import (
    add_display_fields,
    build_reason_codes,
    feature_to_group,
    group_contributions,
    load_templates,
)
from ontime.features.build import FEATURE_COLUMNS
from ontime.models.data import prepare_frame

MODELS = ["logreg", "lightgbm", "ebm"]
MODEL_LABELS = {"logreg": "Logistic regression", "lightgbm": "LightGBM", "ebm": "EBM (glass-box)"}


def recompute_dependent_features(
    row: pd.DataFrame, extra_promise_days: float = 0.0, total_freight: float | None = None
) -> pd.DataFrame:
    """Apply a what-if edit and recompute every feature that is derived from the edited ones.

    Promise edit : promised_days -> promise_per_1000km (= promised_days / (distance_km/1000 + 1))
    Freight edit : total_freight -> freight_ratio (= total_freight / total_price)
    Nothing else changes: other features (distance, seller history, calendar) are facts of the
    order and do not depend on these levers.
    """
    r = row.copy()
    if extra_promise_days:
        r["promised_days"] = r["promised_days"] + extra_promise_days
        r["promise_per_1000km"] = r["promised_days"] / (r["distance_km"] / 1000 + 1)
    if total_freight is not None:
        r["total_freight"] = float(total_freight)
        r["freight_ratio"] = np.where(
            r["total_price"] > 0, r["total_freight"] / r["total_price"], np.nan
        )
    return r


class RiskService:
    """Loads saved models once; scores and explains orders from the feature table."""

    def __init__(self) -> None:
        cfg = load_config()
        mdir = project_path(cfg["paths"]["models_dir"])
        self.arts = {m: joblib.load(mdir / f"{m}.joblib") for m in MODELS}
        self.levels = self.arts["logreg"]["levels"]
        self.templates = load_templates()
        self.f2g = feature_to_group(self.templates)
        self.group_names = list(self.templates["groups"])

    # ------------------------------------------------------------------ scoring
    def score(self, df: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray]:
        """(raw model probability, deployed calibrated probability) per row."""
        art = self.arts[model]
        X = prepare_frame(df, self.levels)
        raw = np.asarray(art["model"].predict_proba(X)[:, 1], dtype=float)
        return raw, np.asarray(art["calibrator"].transform(raw), dtype=float)

    # ------------------------------------------------------------------ explanation
    def grouped_contributions(self, df: pd.DataFrame, model: str) -> pd.DataFrame:
        """Reason-group contributions (log-odds), one row per input row."""
        X = prepare_frame(df, self.levels)
        contrib, _ = EXTRACTORS[model](self.arts[model]["model"], X)
        return group_contributions(contrib, self.f2g)

    def reasons(self, df: pd.DataFrame, model: str) -> pd.DataFrame:
        """Plain-English reason codes (lists of sentences) for each row."""
        X = prepare_frame(df, self.levels)
        grouped = self.grouped_contributions(df, model)
        return build_reason_codes(grouped, add_display_fields(X), self.templates)

    def explain_one(self, row: pd.DataFrame, model: str) -> dict:
        """Score + reasons + group contributions for a single-row frame."""
        raw, prob = self.score(row, model)
        grouped = self.grouped_contributions(row, model)
        codes = self.reasons(row, model).iloc[0]
        return {
            "probability": float(prob[0]),
            "raw": float(raw[0]),
            "contributions": grouped.iloc[0].sort_values(),
            "risk_factors": list(codes["risk_factors"]),
            "protective_factors": list(codes["protective_factors"]),
        }

    def promise_sweep(
        self, row: pd.DataFrame, model: str, max_extra_days: int = 15
    ) -> pd.DataFrame:
        """Model risk as the promise is extended by 0..max_extra_days (what-if curve)."""
        days = np.arange(0, max_extra_days + 1)
        rows = pd.concat(
            [recompute_dependent_features(row, float(d)) for d in days], ignore_index=True
        )
        _, prob = self.score(rows, model)
        return pd.DataFrame({"extra_promise_days": days, "probability": prob})


def required_columns() -> list[str]:
    return list(FEATURE_COLUMNS)
