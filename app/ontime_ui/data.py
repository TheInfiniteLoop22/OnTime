"""Cached loaders for every artifact the app shows. Nothing numeric is hard-coded in the pages:
they read these generated files (results.json, impact.json, features, models...)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ontime.config import load_config, project_path
from ontime.features.build import FEATURE_COLUMNS
from ontime.serving.service import MODELS, RiskService

CFG = load_config()
REPORTS = project_path(CFG["paths"]["reports_dir"])
FIGURES = project_path(CFG["paths"]["figures_dir"])

REQUIRED = {
    "features": project_path(CFG["paths"]["features_parquet"]),
    "results": REPORTS / "results.json",
    "impact": REPORTS / "impact.json",
    "validation": REPORTS / "validation_results.json",
    "reason codes": REPORTS / "test_reason_codes.parquet",
    "logreg model": project_path(CFG["paths"]["models_dir"]) / "logreg.joblib",
    "lightgbm model": project_path(CFG["paths"]["models_dir"]) / "lightgbm.joblib",
    "ebm model": project_path(CFG["paths"]["models_dir"]) / "ebm.joblib",
}


def require_artifacts() -> None:
    """Friendly stop if the pipeline has not been run yet."""
    missing = [k for k, p in REQUIRED.items() if not Path(p).exists()]
    if missing:
        st.error(
            "OnTime needs the pipeline outputs before the app can start. Missing: "
            + ", ".join(missing)
            + ". Run `make all` (or the individual stages) and reload."
        )
        st.stop()


@st.cache_resource(show_spinner="Loading models...")
def get_service() -> RiskService:
    return RiskService()


@st.cache_data(show_spinner=False)
def read_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def features_all() -> pd.DataFrame:
    return pd.read_parquet(REQUIRED["features"])


@st.cache_data(show_spinner=False)
def reason_codes() -> pd.DataFrame:
    return pd.read_parquet(REQUIRED["reason codes"])


@st.cache_data(show_spinner=False)
def slices() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "slices.csv")


@st.cache_data(show_spinner="Scoring the test period...")
def test_scored() -> pd.DataFrame:
    """Test-period orders with calibrated scores and within-day ranks for each model."""
    svc = get_service()
    t = features_all()
    test = t[t["split"] == "test"].reset_index(drop=True)
    keep = ["order_id", "approved_at", "is_late", "delivery_days", "primary_seller_id",
            "seller_state", "customer_state", "distance_km", "promised_days", "category_group",
            "seller_n_past", "seller_late_rate_shrunk", "route_late_rate_shrunk"]  # fmt: skip
    out = test[keep].copy()
    out["day"] = out["approved_at"].dt.normalize()
    out["n_day"] = out.groupby("day")["order_id"].transform("size")
    for m in MODELS:
        _, prob = svc.score(test, m)
        out[f"p_{m}"] = prob
        out[f"rank_{m}"] = out.groupby("day")[f"p_{m}"].rank(ascending=False, method="first")
        out[f"pct_{m}"] = out[f"rank_{m}"] / out["n_day"]
    return out


def band_for(rank_pct: float) -> str:
    """Risk band from the order's percentile rank within its day (rank, not probability).

    High = inside the smallest ops-capacity tier (config), Elevated = inside the largest tier.
    """
    caps = CFG["policy"]["capacity_grid"]
    if rank_pct <= caps[0]:
        return "High"
    if rank_pct <= caps[-1]:
        return "Elevated"
    return "Normal"


def feature_row(order_id: str) -> pd.DataFrame:
    t = features_all()
    return t.loc[t["order_id"] == order_id, FEATURE_COLUMNS].head(1).reset_index(drop=True)


def short_id(order_id: str) -> str:
    return f"{order_id[:8]}…"


def primary_model() -> str:
    return read_json("results.json")["primary_model"]


def policy_table() -> pd.DataFrame:
    return pd.DataFrame(read_json("results.json")["policy"])


def np_isnan(v) -> bool:  # noqa: ANN001
    return isinstance(v, float) and bool(np.isnan(v))
