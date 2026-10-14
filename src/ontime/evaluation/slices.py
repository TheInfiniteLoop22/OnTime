"""Slice analysis: where is the model weak?

A model with a good overall PR-AUC can still fail for specific groups (a region, new sellers, a
month). Each slice reports size, late rate, ranking quality and calibration-in-the-large, and
is flagged when the slice is large enough to trust AND the model is clearly weaker there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MIN_LATE_FOR_METRICS = 30  # below this, per-slice AUCs are noise and are not reported


def add_slice_columns(df: pd.DataFrame, new_seller_threshold: int = 20) -> pd.DataFrame:
    """Derived grouping columns used only for slicing."""
    d = df.copy()
    d["seller_tenure"] = np.where(
        d["seller_n_past"] < new_seller_threshold, "new (<20 past orders)", "established"
    )
    d["distance_bucket"] = pd.cut(
        d["distance_km"],
        [0, 100, 300, 600, 1000, 1500, np.inf],
        labels=["<100", "100-300", "300-600", "600-1000", "1000-1500", ">1500"],
        right=False,
    ).astype(str)
    d["approval_month"] = d["approved_at"].dt.strftime("%Y-%m")
    d["route_type"] = np.where(d["same_state"] == 1, "same state", "different states")
    return d


def slice_table(
    d: pd.DataFrame, p: np.ndarray, columns: list[str], weak_margin: float = 0.05
) -> pd.DataFrame:
    """One row per (slice column, value). `d` must hold is_late and the slice columns."""
    y = d["is_late"].to_numpy()
    p = np.asarray(p, dtype=float)
    overall_auc = roc_auc_score(y, p)
    rows = []
    for col in columns:
        for val, idx in d.groupby(col, observed=True).indices.items():
            ys, ps = y[idx], p[idx]
            late = int(ys.sum())
            row = {
                "slice": col,
                "value": str(val),
                "orders": int(len(idx)),
                "late_orders": late,
                "late_rate": float(ys.mean()),
                "mean_predicted": float(ps.mean()),
                "pr_auc": np.nan,
                "roc_auc": np.nan,
            }
            if late >= MIN_LATE_FOR_METRICS and late < len(idx):
                row["pr_auc"] = float(average_precision_score(ys, ps))
                row["roc_auc"] = float(roc_auc_score(ys, ps))
            rows.append(row)
    t = pd.DataFrame(rows)
    t["lift_pr_auc"] = t["pr_auc"] / t["late_rate"]
    t["weak"] = (t["roc_auc"] < overall_auc - weak_margin).fillna(False)
    t["overall_roc_auc"] = overall_auc
    return t
