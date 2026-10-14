"""SHAP plots for the LightGBM model (global summary, importance bars, dependence, waterfall).

SHAP assigns each feature its Shapley value: its average marginal contribution to this order's
prediction over all orderings of the features. For trees it is computed exactly (TreeSHAP).
Limitation to remember: with correlated features (distance vs promise_per_1000km, several late-rate
features) SHAP splits credit between them somewhat arbitrarily.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402


def _save_current(path: Path) -> None:
    plt.gcf().tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def prepared_for_plot(X: pd.DataFrame) -> pd.DataFrame:
    """SHAP's plotting wants numbers: encode categoricals as integer codes (labels kept)."""
    d = X.copy()
    for c in d.columns:
        if str(d[c].dtype) == "category":
            d[c] = d[c].cat.codes.astype(float)
    return d


def beeswarm(values: np.ndarray, X: pd.DataFrame, path: Path, max_display: int = 15) -> None:
    plt.figure(figsize=(8, 6))
    shap.summary_plot(values, prepared_for_plot(X), max_display=max_display, show=False)
    plt.title("SHAP summary (LightGBM): each dot is an order")
    _save_current(path)


def bar(values: np.ndarray, X: pd.DataFrame, path: Path, max_display: int = 15) -> None:
    plt.figure(figsize=(8, 5.5))
    shap.summary_plot(
        values, prepared_for_plot(X), plot_type="bar", max_display=max_display, show=False
    )
    plt.title("SHAP global importance (mean |SHAP|, LightGBM)")
    _save_current(path)


def dependence(values: np.ndarray, X: pd.DataFrame, features: list[str], path: Path) -> None:
    n = len(features)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows))
    Xp = prepared_for_plot(X)
    for ax, f in zip(np.atleast_1d(axes).ravel(), features, strict=False):
        shap.dependence_plot(f, values, Xp, ax=ax, show=False, interaction_index=None)
        ax.set_title(f"SHAP dependence: {f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def waterfall(
    row_values: np.ndarray, base: float, row_data: pd.Series, names: list[str], path: Path,
    title: str,
) -> None:  # fmt: skip
    expl = shap.Explanation(
        values=row_values,
        base_values=base,
        data=row_data.to_numpy(dtype=object),
        feature_names=names,
    )
    plt.figure(figsize=(8, 5.5))
    shap.plots.waterfall(expl, max_display=10, show=False)
    plt.title(title, fontsize=10)
    _save_current(path)
