"""Evaluation metrics for an imbalanced, capacity-constrained problem.

Why these metrics (not accuracy):
* PR-AUC (average precision) - focuses on the rare late class; the primary metric.
* ROC-AUC - threshold-free ranking quality, reported for completeness (it is optimistic when
  the positive class is rare).
* Brier score and log loss - quality of the *probabilities*, which the app shows to humans.
* precision@k / recall@k / lift - what ops actually experiences: "if we can act on k% of each
  day's orders, how many of those are truly late and how many late orders do we catch?"

Confidence intervals come from percentile bootstrap. Order-level metrics resample orders;
daily top-k metrics resample whole DAYS, because orders on the same day are exposed to the same
carrier conditions and are not independent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


# ------------------------------------------------------------------ order-level metrics
def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """PR-AUC, ROC-AUC, Brier, log loss and the base rate for probabilities `p`."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "base_rate": float(y.mean()),
    }


def bootstrap_metric_ci(
    y: np.ndarray, p: np.ndarray, n_boot: int, seed: int, level: float = 0.95
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for pr_auc, roc_auc, brier, log_loss (orders resampled)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws: dict[str, list[float]] = {k: [] for k in ("pr_auc", "roc_auc", "brier", "log_loss")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():  # degenerate resample (no positives): skip
            continue
        m = classification_metrics(yb, pb)
        for k in draws:
            draws[k].append(m[k])
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in draws.items()}


# ------------------------------------------------------------------ capacity (top-k per day)
def daily_topk_table(
    dates: pd.Series, y: np.ndarray, score: np.ndarray, k_frac: float, seed: int
) -> pd.DataFrame:
    """Per-day counts when ops acts on the top `k_frac` of that day's orders by score.

    n_selected = ceil(k_frac * n_orders_that_day) (at least 1). Ties in score are broken by a
    seeded random jitter, so constant-score baselines are evaluated fairly (random selection)
    and results are reproducible.
    """
    d = pd.DataFrame(
        {
            "day": pd.to_datetime(dates).dt.normalize().to_numpy(),
            "y": np.asarray(y),
            "score": np.asarray(score, dtype=float),
        }
    )
    d["jitter"] = np.random.default_rng(seed).random(len(d))
    d = d.sort_values(["day", "score", "jitter"], ascending=[True, False, False])
    d["rank"] = d.groupby("day").cumcount()
    d["n_day"] = d.groupby("day")["y"].transform("size")
    quota = np.maximum(1, np.ceil(k_frac * d["n_day"]).astype(int))
    d["picked"] = (d["rank"] < quota).astype(int)
    d["captured"] = d["picked"] * d["y"]  # late orders that were picked
    out = d.groupby("day").agg(
        orders=("y", "size"),
        late=("y", "sum"),
        selected=("picked", "sum"),
        captured=("captured", "sum"),
    )
    return out.reset_index()


def _topk_summary(t: pd.DataFrame) -> dict[str, float]:
    sel, cap, late, n = t["selected"].sum(), t["captured"].sum(), t["late"].sum(), t["orders"].sum()
    precision = cap / sel if sel else math.nan
    recall = cap / late if late else math.nan
    base = late / n
    return {
        "precision": float(precision),
        "recall": float(recall),
        "lift": float(precision / base) if base else math.nan,
        "selected": int(sel),
        "captured_late": int(cap),
    }


def topk_metrics(
    dates: pd.Series, y: np.ndarray, score: np.ndarray, k_frac: float, seed: int
) -> dict[str, float]:
    """Pooled precision@k, recall@k and lift@k when acting on top-k% of each day's orders."""
    return {"k": k_frac, **_topk_summary(daily_topk_table(dates, y, score, k_frac, seed))}


def bootstrap_topk_ci(
    dates: pd.Series,
    y: np.ndarray,
    score: np.ndarray,
    k_frac: float,
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs for precision/recall/lift@k, resampling whole days (cluster bootstrap)."""
    t = daily_topk_table(dates, y, score, k_frac, seed)
    rng = np.random.default_rng(seed)
    cols = t[["orders", "late", "selected", "captured"]].to_numpy()
    draws = {"precision": [], "recall": [], "lift": []}
    for _ in range(n_boot):
        s = cols[rng.integers(0, len(cols), len(cols))].sum(axis=0)
        n, late, sel, cap = s
        if sel == 0 or late == 0:
            continue
        draws["precision"].append(cap / sel)
        draws["recall"].append(cap / late)
        draws["lift"].append((cap / sel) / (late / n))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in draws.items()}


def evaluate_scores(
    dates: pd.Series,
    y: np.ndarray,
    p: np.ndarray,
    k_grid: list[float],
    n_boot: int,
    seed: int,
) -> dict:
    """One-stop evaluation: order-level metrics + top-k metrics, each with bootstrap CIs."""
    out: dict = {"n": int(len(y)), "late": int(np.sum(y))}
    point = classification_metrics(y, p)
    ci = bootstrap_metric_ci(y, p, n_boot, seed)
    out["metrics"] = {
        k: {"value": v, **({"ci95": list(ci[k])} if k in ci else {})} for k, v in point.items()
    }
    out["top_k"] = []
    for k in k_grid:
        row = topk_metrics(dates, y, p, k, seed)
        row["ci95"] = {
            m: list(v) for m, v in bootstrap_topk_ci(dates, y, p, k, n_boot, seed).items()
        }
        out["top_k"].append(row)
    return out
