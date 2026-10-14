"""Per-order, per-feature contributions in LOG-ODDS for each model family.

Every extractor returns (contributions DataFrame [n x features], intercept) such that

    sum(contributions, axis=1) + intercept == the model's raw log-odds for that order.

That additive identity is unit-tested, so explanations are exact decompositions, not
approximations (except SHAP's own definition for the tree model).

Contributions are relative to the model's average behaviour: positive = pushes this order's
risk up, negative = pulls it down. Calibrators (sigmoid) are monotone and keep the direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ontime.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from ontime.models.train import INT_AS_CATEGORY, _as_strings

_CAT_PREFIXES = sorted(CATEGORICAL_FEATURES + INT_AS_CATEGORY, key=len, reverse=True)


def source_feature(transformed_name: str, extra: list[str] | None = None) -> str:
    """Map a preprocessed column name back to the original feature it came from.

    `extra` lists additional numeric features (the stage-2 model adds some)."""
    if transformed_name.startswith("missingindicator_"):
        return transformed_name[len("missingindicator_") :]
    if transformed_name in FEATURE_COLUMNS or transformed_name in (extra or []):
        return transformed_name
    for c in _CAT_PREFIXES:  # one-hot names look like "<feature>_<level>"
        if transformed_name.startswith(c + "_"):
            return c
    raise KeyError(f"cannot map transformed column {transformed_name!r} to a source feature")


def _to_frame(mat: np.ndarray, names: list[str], index, columns=None) -> pd.DataFrame:  # noqa: ANN001
    df = pd.DataFrame(mat, columns=names, index=index)
    out = df.T.groupby(level=0).sum().T  # sum columns that share a source feature
    return out.reindex(columns=columns or FEATURE_COLUMNS, fill_value=0.0)


def logreg_contributions(
    model,
    X: pd.DataFrame,
    columns: list[str] | None = None,  # noqa: ANN001
) -> tuple[pd.DataFrame, float]:
    """coefficient x preprocessed value, summed back onto source features.

    `columns` overrides the output feature list (stage 2 passes FEATURE_COLUMNS + its own)."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Z = prep.transform(_as_strings(X))
    extra = [c for c in (columns or []) if c not in FEATURE_COLUMNS]
    names = [source_feature(n, extra) for n in prep.get_feature_names_out()]
    contrib = _to_frame(Z * clf.coef_[0], names, X.index, columns)
    return contrib, float(clf.intercept_[0])


def ebm_contributions(ebm, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:  # noqa: ANN001
    """EBM term scores; an interaction term is split equally between its two features."""
    terms = np.asarray(ebm.eval_terms(X.reset_index(drop=True)))
    feat_names = list(ebm.feature_names_in_)
    out = np.zeros((len(X), len(feat_names)))
    for j, idx in enumerate(ebm.term_features_):
        for f in idx:
            out[:, f] += terms[:, j] / len(idx)
    df = pd.DataFrame(out, columns=feat_names, index=X.index)
    return df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0), float(np.ravel(ebm.intercept_)[0])


def lgbm_contributions(
    model,
    X: pd.DataFrame,
    columns: list[str] | None = None,  # noqa: ANN001
) -> tuple[pd.DataFrame, float]:
    """TreeSHAP values (exact Shapley values for the tree ensemble), in log-odds."""
    import shap

    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    if isinstance(sv, list):  # older shap returned [neg, pos]
        sv = sv[-1]
    base = float(np.ravel(ex.expected_value)[-1])
    df = pd.DataFrame(np.asarray(sv), columns=list(X.columns), index=X.index)
    return df.reindex(columns=columns or FEATURE_COLUMNS, fill_value=0.0), base


EXTRACTORS = {
    "logreg": logreg_contributions,
    "ebm": ebm_contributions,
    "lightgbm": lgbm_contributions,
}
