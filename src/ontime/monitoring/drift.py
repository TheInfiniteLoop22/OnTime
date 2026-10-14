"""Drift monitoring: has the world moved away from what the model was trained on?

Three signals, computed per calendar month against the TRAINING period as the reference:

* Input drift   - Population Stability Index (PSI) for every model feature.
* Label drift   - the late rate itself.
* Performance   - ranking quality (ROC-AUC, PR-AUC, lift@10%) of the deployed model.

PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%). Common reading:
< 0.10 stable, 0.10-0.25 moderate shift (watch), > 0.25 major shift (investigate).
Performance monitoring needs outcomes, which arrive with a delay; PSI does not, so PSI is the
early-warning signal and performance is the confirmation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ontime.evaluation.metrics import topk_metrics
from ontime.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS

log = logging.getLogger(__name__)
EPS = 1e-4


def _psi_from_counts(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected / max(expected.sum(), 1), EPS, None)
    a = np.clip(actual / max(actual.sum(), 1), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """PSI with quantile bins from the reference; missing values form their own bin."""
    r, c = ref.astype(float), cur.astype(float)
    edges = np.unique(np.nanquantile(r, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant reference: compare value shares instead
        return psi_categorical(r.round(6).astype(str), c.round(6).astype(str))
    edges[0], edges[-1] = -np.inf, np.inf

    def counts(x: pd.Series) -> np.ndarray:
        h = np.histogram(x.dropna(), bins=edges)[0].astype(float)
        return np.append(h, float(x.isna().sum()))

    return _psi_from_counts(counts(r), counts(c))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    levels = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    e = ref.astype(str).value_counts().reindex(levels, fill_value=0).to_numpy(float)
    a = cur.astype(str).value_counts().reindex(levels, fill_value=0).to_numpy(float)
    return _psi_from_counts(e, a)


def monthly_psi(
    table: pd.DataFrame,
    reference_mask: pd.Series | None = None,
    trailing_months: int | None = None,
    min_orders: int = 500,
    bins: int = 10,
    first_month: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Long table: month x feature -> PSI.

    Two reference choices (give exactly one):
    * `reference_mask`   - a fixed reference (the training period): long-run drift.
    * `trailing_months`  - the N months immediately before each month: sudden change.
    Only months from `first_month` (default: after the fixed reference) are reported.
    """
    if (reference_mask is None) == (trailing_months is None):
        raise ValueError("give exactly one of reference_mask / trailing_months")
    month = table["approved_at"].dt.to_period("M").dt.start_time
    fixed = table[reference_mask] if reference_mask is not None else None
    if first_month is None and fixed is not None:
        first_month = (fixed["approved_at"].max() + pd.Timedelta(days=1)).to_period("M").start_time
    rows = []
    for m, idx in table.groupby(month).groups.items():
        cur = table.loc[idx]
        if len(cur) < min_orders or (first_month is not None and m < first_month):
            continue
        if fixed is not None:
            if cur["approved_at"].min() <= fixed["approved_at"].max():
                continue
            ref = fixed
        else:
            start = m - pd.DateOffset(months=trailing_months)
            ref = table[(table["approved_at"] >= start) & (table["approved_at"] < m)]
            if len(ref) < min_orders:
                continue
        for f in FEATURE_COLUMNS:
            v = (
                psi_categorical(ref[f], cur[f])
                if f in CATEGORICAL_FEATURES
                else psi_numeric(ref[f], cur[f], bins)
            )
            rows.append({"month": m, "feature": f, "psi": v, "orders": len(cur)})
    return pd.DataFrame(rows)


def status_from_psi(psi: float, watch: float, alert: float) -> str:
    return "alert" if psi >= alert else "watch" if psi >= watch else "stable"


def performance_by_month(
    scores: pd.DataFrame, k: float = 0.10, seed: int = 42, min_late: int = 30
) -> pd.DataFrame:
    """Ranking quality of one score column per month.

    scores needs: approved_at, is_late, score. Months with fewer than `min_late` late orders get
    NaN AUCs (too noisy to read).
    """
    s = scores.assign(month=scores["approved_at"].dt.to_period("M").dt.start_time)
    rows = []
    for m, g in s.groupby("month"):
        y, p = g["is_late"].to_numpy(), g["score"].to_numpy()
        ok = y.sum() >= min_late and y.sum() < len(y)
        tk = topk_metrics(g["approved_at"], y, p, k, seed)
        rows.append(
            {
                "month": m,
                "orders": len(g),
                "late_rate": float(y.mean()),
                "roc_auc": float(roc_auc_score(y, p)) if ok else np.nan,
                "pr_auc": float(average_precision_score(y, p)) if ok else np.nan,
                "lift_at_10": tk["lift"],
            }
        )
    return pd.DataFrame(rows)


def month_status(
    psi_long: pd.DataFrame,
    perf: pd.DataFrame,
    watch: float,
    alert: float,
    roc_drop: float = 0.05,
    exclude: tuple[str, ...] = (),
) -> pd.DataFrame:
    """One status line per month.

    `psi_long` must have a `baseline` column ("training" or "trailing"). Features in `exclude`
    (time-indexed by construction: they drift every month by design) do not count toward status.

    Rules (deliberately simple and explainable):
      alert  : >= 3 features with trailing-PSI >= alert, OR ROC-AUC fell >= `roc_drop` below the
               first measured month (performance confirmation, needs outcomes);
      watch  : >= 1 feature with trailing-PSI >= alert, OR >= 3 with training-PSI >= alert;
      stable : otherwise.
    """
    live = psi_long[~psi_long["feature"].isin(exclude)]

    def agg(baseline: str, prefix: str) -> pd.DataFrame:
        g = live[live["baseline"] == baseline].groupby("month")["psi"]
        return pd.DataFrame(
            {
                f"{prefix}_alert": g.apply(lambda s: int((s >= alert).sum())),
                f"{prefix}_watch": g.apply(lambda s: int(((s >= watch) & (s < alert)).sum())),
                f"{prefix}_max": g.max(),
            }
        )

    out = agg("training", "train").join(agg("trailing", "trail"), how="outer")
    p = perf.set_index("month")
    out = out.join(p, how="left")
    ref_auc = p["roc_auc"].dropna().iloc[0] if p["roc_auc"].notna().any() else np.nan
    out["roc_vs_first"] = out["roc_auc"] - ref_auc
    trail = out["trail_alert"].fillna(0)
    out["status"] = np.where(
        (trail >= 3) | (out["roc_vs_first"] <= -roc_drop),
        "alert",
        np.where((trail >= 1) | (out["train_alert"].fillna(0) >= 3), "watch", "stable"),
    )
    return out.reset_index().rename(columns={"index": "month"})
