"""Report figures for the final evaluation. Every function saves a PNG (dpi 150) and returns its
file name so `results.md` can reference it. Titles and axes are written for a non-specialist."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from ontime.evaluation.calibration import reliability_table  # noqa: E402

COLORS = {
    "logreg": "#4C78A8",
    "lightgbm": "#F58518",
    "ebm": "#54A24B",
    "random_forest": "#B279A2",
    "rule": "#9D9D9D",
    "prior_rate": "#CFCFCF",
    "random": "#7F7F7F",
    "oracle": "#222222",
}
LABELS = {
    "logreg": "Logistic regression",
    "lightgbm": "LightGBM",
    "ebm": "EBM (glass-box)",
    "random_forest": "Random forest",
    "rule": "Hand-written rule",
    "prior_rate": "Guess the training rate",
    "random": "Random selection",
    "oracle": "Perfect ranking",
}


def _style() -> None:
    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3}
    )


def _save(fig: plt.Figure, fig_dir: Path, name: str) -> str:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=150)
    plt.close(fig)
    return name


def pr_curves(y: np.ndarray, probs: dict[str, np.ndarray], fig_dir: Path, tag: str) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=LABELS.get(name, name), color=COLORS.get(name), lw=1.8)
    ax.axhline(y.mean(), color="k", ls="--", lw=1, label=f"Base rate ({y.mean():.1%})")
    ax.set_xlabel("Recall: share of all late orders caught")
    ax.set_ylabel("Precision: share of flagged orders that are truly late")
    ax.set_title(f"Precision-recall curves ({tag})")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    return _save(fig, fig_dir, f"{tag}_pr_curves.png")


def calibration_plot(
    y: np.ndarray,
    probs: dict[str, np.ndarray],
    rolling: tuple[str, np.ndarray, np.ndarray] | None,
    fig_dir: Path,
    tag: str,
) -> str:
    """Reliability diagram: points on the diagonal = honest probabilities."""
    _style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    hi = 0.0
    for name, p in probs.items():
        t = reliability_table(y, p)
        hi = max(hi, t["mean_predicted"].max(), t["observed"].max())
        ax[0].plot(t["mean_predicted"], t["observed"], marker="o", label=LABELS.get(name, name),
                   color=COLORS.get(name))  # fmt: skip
    hi = max(hi, 0.1) * 1.05
    ax[0].plot([0, hi], [0, hi], "k--", lw=1, label="Perfect calibration")
    ax[0].set_xlabel("Predicted probability of being late")
    ax[0].set_ylabel("Observed share late")
    ax[0].set_title("Deployed calibrators (fit on validation)")
    ax[0].legend(fontsize=8)
    if rolling is not None:
        name, p_dep, p_roll = rolling
        for lab, p, c in [
            ("Deployed", p_dep, "#C0392B"),
            ("Rolling recalibration", p_roll, "#2E86C1"),
        ]:
            m = ~np.isnan(p)
            t = reliability_table(y[m], p[m])
            ax[1].plot(t["mean_predicted"], t["observed"], marker="o", label=lab, color=c)
        ax[1].plot([0, hi], [0, hi], "k--", lw=1)
        ax[1].set_title(f"{LABELS.get(name, name)}: fixed vs monthly-recalibrated")
        ax[1].set_xlabel("Predicted probability of being late")
        ax[1].set_ylabel("Observed share late")
        ax[1].legend(fontsize=8)
    return _save(fig, fig_dir, f"{tag}_calibration.png")


def gains_and_lift(y: np.ndarray, probs: dict[str, np.ndarray], fig_dir: Path, tag: str) -> str:
    """Cumulative gains and lift when ordering ALL orders by risk (pooled ranking)."""
    _style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    n, total = len(y), y.sum()
    x = np.arange(1, n + 1) / n
    for name, p in probs.items():
        order = np.argsort(-p, kind="stable")
        gains = np.cumsum(y[order]) / total
        ax[0].plot(x, gains, label=LABELS.get(name, name), color=COLORS.get(name))
        ax[1].plot(
            x[: n // 2], (gains / x)[: n // 2], color=COLORS.get(name), label=LABELS.get(name, name)
        )
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax[0].set_xlabel("Share of orders acted on (highest risk first)")
    ax[0].set_ylabel("Share of all late orders caught")
    ax[0].set_title("Cumulative gains")
    ax[0].legend(fontsize=8)
    ax[1].axhline(1, color="k", ls="--", lw=1)
    ax[1].set_xlabel("Share of orders acted on (highest risk first)")
    ax[1].set_ylabel("Lift over random (x)")
    ax[1].set_title("Lift curve")
    return _save(fig, fig_dir, f"{tag}_gains_lift.png")


def monthly_pr_auc(monthly: pd.DataFrame, fig_dir: Path, tag: str) -> str:
    """PR-AUC per month with the month's base rate: skill must be judged against the floor."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, g in monthly.groupby("model"):
        ax.plot(
            g["month"],
            g["pr_auc"],
            marker="o",
            label=LABELS.get(name, name),
            color=COLORS.get(name),
        )
    base = monthly.drop_duplicates("month")
    ax.plot(base["month"], base["late_rate"], "k--", marker="s", label="Base rate (random)")
    ax.set_xlabel("Month of order approval")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Monthly ranking quality vs the month's base rate")
    ax.legend(fontsize=8)
    return _save(fig, fig_dir, f"{tag}_monthly_pr_auc.png")


def policy_curves(policy: pd.DataFrame, p_values: list[float], fig_dir: Path, tag: str) -> str:
    """Late orders prevented (share of all late) vs ops capacity, per assumed success rate."""
    _style()
    n = len(p_values)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, p_values, strict=True):
        sub = policy[np.isclose(policy["p"], p)]
        for name, g in sub.groupby("strategy"):
            g = g.sort_values("k")
            ls = "--" if name in ("random", "oracle") else "-"
            ax.plot(g["k"] * 100, g["prevented_share_of_late"] * 100, ls=ls, marker="o",
                    label=LABELS.get(name, name), color=COLORS.get(name))  # fmt: skip
        ax.set_title(f"Intervention success = {p:.0%}")
        ax.set_xlabel("Ops capacity: % of each day's orders")
    axes[0].set_ylabel("Late orders prevented (% of all late)")
    axes[-1].legend(fontsize=7)
    return _save(fig, fig_dir, f"{tag}_policy_curves.png")
