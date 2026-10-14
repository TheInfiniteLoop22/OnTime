"""Do different explanation methods tell the same story?

Two checks across models (LR coefficients, EBM terms, LightGBM SHAP):
1. Rank agreement of global importances (Spearman correlation + top-k overlap).
2. Direction agreement per numeric feature: does higher feature value push risk up or down?

Agreement raises trust ("three different methods say distance matters and in which direction").
Disagreement is not a failure; it is a prompt to ask why (correlated features, different
functional forms, or regime changes between model fits).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FLAT_THRESHOLD = 0.2  # |rho| below this = no clear monotone direction (contributions)
MARGINAL_THRESHOLD = (
    0.03  # single features are individually weak; |rho| >= 0.03 is nonzero at n=20k
)


def global_importance(contrib: pd.DataFrame) -> pd.Series:
    """Mean |contribution| per feature (log-odds)."""
    return contrib.abs().mean().sort_values(ascending=False)


def rank_agreement(imps: dict[str, pd.Series], top_k: int = 10) -> dict:
    """Pairwise Spearman correlation of importances and overlap of the top-k feature sets."""
    names = list(imps)
    frame = pd.DataFrame(imps)
    out: dict = {"spearman": {}, "top_k_overlap": {}, "top_k": top_k}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rho = spearmanr(frame[a], frame[b]).statistic
            out["spearman"][f"{a} vs {b}"] = float(rho)
            ta, tb = set(imps[a].head(top_k).index), set(imps[b].head(top_k).index)
            out["top_k_overlap"][f"{a} vs {b}"] = len(ta & tb) / top_k
    return out


def _direction(rho: float, threshold: float = FLAT_THRESHOLD) -> str:
    if np.isnan(rho) or abs(rho) < threshold:
        return "flat/non-monotone"
    return "higher value -> more risk" if rho > 0 else "higher value -> less risk"


def direction_table(
    contribs: dict[str, pd.DataFrame],
    X: pd.DataFrame,
    numeric_features: list[str],
    y: np.ndarray | None = None,
) -> pd.DataFrame:
    """Spearman(feature value, feature contribution) per model, with a direction label.

    If `y` is given, also reports the MARGINAL direction: Spearman(feature, outcome) one feature
    at a time. When a model's conditional direction differs from the marginal one, the feature
    is usually being used as a correction for correlated features (not a stand-alone effect).
    """
    rows = []
    for f in numeric_features:
        row: dict = {"feature": f}
        if y is not None:
            m = spearmanr(X[f].astype(float), y, nan_policy="omit").statistic
            row["marginal_rho"] = float(m)
            row["marginal"] = _direction(m, MARGINAL_THRESHOLD)
        labels = []
        for m, c in contribs.items():
            if c[f].abs().sum() == 0:
                row[m] = "not used"
                continue
            rho = spearmanr(X[f].astype(float), c[f], nan_policy="omit").statistic
            row[m] = _direction(rho)
            row[f"{m}_rho"] = float(rho)
            labels.append(row[m])
        clear = [x for x in labels if x != "flat/non-monotone"]
        row["models_with_clear_direction"] = len(clear)
        row["agree"] = len(clear) >= 2 and len(set(clear)) == 1
        row["conflict"] = len(set(clear)) > 1
        rows.append(row)
    return pd.DataFrame(rows)
