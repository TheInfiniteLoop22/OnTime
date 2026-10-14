"""EBM global explanations: term importances, shape functions, top interaction.

An EBM is a sum of one learned function per feature (plus a few pairwise terms). Plotting those
functions IS the explanation: you can read exactly how each feature moves the log-odds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def term_importance_table(ebm) -> pd.DataFrame:  # noqa: ANN001
    """Mean |score| per term (main effects and interactions), sorted high to low."""
    imp = np.asarray(ebm.term_importances())
    t = pd.DataFrame(
        {
            "term": ebm.term_names_,
            "kind": ["interaction" if " & " in n else "main effect" for n in ebm.term_names_],
            "importance": imp,
        }
    )
    t["share_of_total"] = t["importance"] / t["importance"].sum()
    return t.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_importance(table: pd.DataFrame, path: Path, top: int = 15) -> None:
    t = table.head(top).iloc[::-1]
    colors = np.where(t["kind"] == "interaction", "#F58518", "#4C78A8")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(t["term"], t["importance"], color=colors)
    ax.set_xlabel("Mean absolute contribution to the log-odds of being late")
    ax.set_title("EBM: which terms matter most (orange = pairwise interaction)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _step_xy(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.asarray(data["names"], dtype=float)
    return (
        edges[:-1],
        np.asarray(data["scores"]),
        np.asarray(data["lower_bounds"]),
        np.asarray(data["upper_bounds"]),
    )


def plot_shapes(ebm, features: list[str], X_ref: pd.DataFrame, path: Path) -> None:  # noqa: ANN001
    """Shape functions of continuous main effects, x clipped to the 1st-99th percentile of data."""
    g = ebm.explain_global()
    names = list(ebm.term_names_)
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.3 * rows))
    for ax, f in zip(np.atleast_1d(axes).ravel(), features, strict=False):
        d = g.data(names.index(f))
        x, y, lo, hi = _step_xy(d)
        ax.step(x, y, where="post", color="#4C78A8", lw=1.6)
        ax.fill_between(x, lo, hi, step="post", color="#4C78A8", alpha=0.2)
        q1, q99 = X_ref[f].quantile([0.01, 0.99])
        ax.set_xlim(q1, q99)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(f, fontsize=10)
        ax.set_xlabel(f)
        ax.set_ylabel("Effect on log-odds of late")
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    fig.suptitle("EBM shape functions: how each feature moves the risk (band = model uncertainty)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_interaction(ebm, term_name: str, path: Path) -> None:  # noqa: ANN001
    g = ebm.explain_global()
    d = g.data(list(ebm.term_names_).index(term_name))
    left, right = term_name.split(" & ")
    S = np.asarray(d["scores"])  # shape: [left bins, right bins]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(S.T, origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(S).max(), vmax=np.abs(S).max())  # fmt: skip
    ax.set_xlabel(f"{left} (low -> high bins)")
    ax.set_ylabel(f"{right} (low -> high bins)")
    ax.set_title(f"EBM interaction: {left} x {right}")
    fig.colorbar(im, label="Extra effect on log-odds")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
