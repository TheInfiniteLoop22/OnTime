"""Batch scoring of the test period: one parquet so the app starts fast."""

from __future__ import annotations

import logging

import pandas as pd

from ontime.config import load_config, project_path
from ontime.serving.service import MODELS, RiskService

log = logging.getLogger(__name__)
KEEP = [
    "order_id", "approved_at", "is_late", "delivery_days", "primary_seller_id", "seller_state",
    "customer_state", "distance_km", "promised_days", "category_group", "seller_n_past",
    "seller_late_rate_shrunk", "route_late_rate_shrunk",
]  # fmt: skip


def score_test_period(service: RiskService | None = None) -> pd.DataFrame:
    """Test-period orders with deployed-calibrated scores and within-day ranks for each model."""
    cfg = load_config()
    svc = service or RiskService()
    t = pd.read_parquet(project_path(cfg["paths"]["features_parquet"]))
    test = t[t["split"] == "test"].reset_index(drop=True)
    out = test[KEEP].copy()
    out["day"] = out["approved_at"].dt.normalize()
    out["n_day"] = out.groupby("day")["order_id"].transform("size")
    for m in MODELS:
        _, prob = svc.score(test, m)
        out[f"p_{m}"] = prob
        out[f"rank_{m}"] = out.groupby("day")[f"p_{m}"].rank(ascending=False, method="first")
        out[f"pct_{m}"] = out[f"rank_{m}"] / out["n_day"]
    return out


def write_test_scores() -> str:
    cfg = load_config()
    path = project_path(cfg["paths"]["reports_dir"]) / "test_scores.parquet"
    df = score_test_period()
    df.to_parquet(path, index=False)
    log.info("scores | wrote %s (%d orders)", path, len(df))
    return str(path)
