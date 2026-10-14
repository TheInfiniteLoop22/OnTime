"""Calibration diagnostics and the production-style rolling recalibration.

Why rolling recalibration: Phase 6 showed a calibrator fitted on one period over-predicts when the
late rate later falls. In production you would refit the (cheap, 2-parameter) calibrator
regularly on recently *observed* outcomes. `rolling_recalibrate` simulates exactly that, using
only outcomes that were already known at the start of each evaluation month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ontime.models.calibrate import SigmoidCalibrator


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned reliability: mean predicted vs observed late rate per bin."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    g = d.groupby("bin").agg(
        orders=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean")
    )
    return g.reset_index(drop=True)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean |predicted - observed| across quantile bins (lower is better)."""
    t = reliability_table(y, p, n_bins)
    w = t["orders"] / t["orders"].sum()
    return float((w * (t["mean_predicted"] - t["observed"]).abs()).sum())


def rolling_recalibrate(
    scores: pd.DataFrame,
    eval_start: str,
    eval_end: str,
    window_days: int = 60,
    freq: str = "MS",
) -> pd.Series:
    """Recalibrated probabilities for orders approved in [eval_start, eval_end].

    scores needs columns: approved_at, delivered_at, y, raw. For each calendar month M starting
    at s, a sigmoid calibrator is fitted on orders DELIVERED in [s - window_days, s) - i.e. only
    outcomes observable before the month began - and applied to that month's orders.
    Returns a Series indexed like `scores` (NaN outside the evaluation window).
    """
    out = pd.Series(np.nan, index=scores.index, dtype=float)
    in_window = (scores["approved_at"] >= pd.Timestamp(eval_start)) & (
        scores["approved_at"] < pd.Timestamp(eval_end) + pd.Timedelta(days=1)
    )
    months = pd.date_range(pd.Timestamp(eval_start).replace(day=1), eval_end, freq=freq)
    for s in months:
        e = s + pd.offsets.MonthBegin(1)
        pool = scores[
            (scores["delivered_at"] >= s - pd.Timedelta(days=window_days))
            & (scores["delivered_at"] < s)
        ]
        target = in_window & (scores["approved_at"] >= s) & (scores["approved_at"] < e)
        if len(pool) < 200 or pool["y"].nunique() < 2 or not target.any():
            continue
        cal = SigmoidCalibrator().fit(pool["raw"].to_numpy(), pool["y"].to_numpy())
        out[target] = cal.transform(scores.loc[target, "raw"].to_numpy())
    return out
