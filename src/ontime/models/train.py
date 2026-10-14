"""Model builders and fitters: logistic regression, random forest, LightGBM, EBM.

All four expose `predict_proba(X) -> (n, 2)` so evaluation code treats them identically.
Every random process takes its seed from config (CLAUDE.md rule 7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ontime.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS

# Right-skewed, non-negative quantities: log1p before scaling (FEATURE_SPEC section 6).
LOG_FEATURES = [
    "distance_km",
    "total_price",
    "total_freight",
    "total_weight_kg",
    "total_volume_l",
    "max_item_weight_kg",
    "seller_n_past",
    "seller_load_7d",
    "seller_days_since_first_order",
]
# Calendar integers are cyclical, not ordered quantities: one-hot them in linear models.
INT_AS_CATEGORY = ["approve_dow", "approve_month"]
DROP_FOR_LINEAR = ["log_distance_km"]  # duplicate of log(distance_km)


def _log1p_nonneg(a: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(a, 0, None))


def build_preprocessor() -> ColumnTransformer:
    """Median-impute (+ missing indicators), log + scale skewed numerics, scale the rest,
    one-hot categoricals. Fit on TRAIN only (it lives inside the sklearn Pipeline)."""
    cat = CATEGORICAL_FEATURES + INT_AS_CATEGORY
    num_all = [f for f in FEATURE_COLUMNS if f not in CATEGORICAL_FEATURES + INT_AS_CATEGORY]
    num_all = [f for f in num_all if f not in DROP_FOR_LINEAR]
    log_cols = [f for f in LOG_FEATURES if f in num_all]
    lin_cols = [f for f in num_all if f not in log_cols]
    log_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    lin_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipe = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False
    )
    return ColumnTransformer(
        [("log", log_pipe, log_cols), ("lin", lin_pipe, lin_cols), ("cat", cat_pipe, cat)],
        verbose_feature_names_out=False,
    )


def _as_strings(X: pd.DataFrame) -> pd.DataFrame:
    """Linear/forest pipelines one-hot calendar ints and categories as plain strings."""
    X = X.copy()
    for c in CATEGORICAL_FEATURES + INT_AS_CATEGORY:
        X[c] = X[c].astype(object).astype(str)
    return X


class _StringifyPipeline(Pipeline):
    """Pipeline that casts categorical/calendar columns to strings before preprocessing."""

    def fit(self, X, y=None, **kw):  # noqa: ANN001
        return super().fit(_as_strings(X), y, **kw)

    def predict_proba(self, X):  # noqa: ANN001
        return super().predict_proba(_as_strings(X))


def pos_weight(option: str, y: np.ndarray) -> float:
    """scale_pos_weight for an option: none=1, sqrt=sqrt(neg/pos), balanced=neg/pos."""
    ratio = (len(y) - y.sum()) / max(y.sum(), 1)
    return {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "balanced": float(ratio)}[option]


def fit_logreg(X: pd.DataFrame, y: np.ndarray, params: dict, seed: int) -> Pipeline:
    clf = LogisticRegression(
        C=params["C"],
        class_weight=None if params["class_weight"] == "none" else "balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    return _StringifyPipeline([("prep", build_preprocessor()), ("clf", clf)]).fit(X, y)


def fit_random_forest(X: pd.DataFrame, y: np.ndarray, seed: int) -> Pipeline:
    """Fixed, sensible settings: a sanity comparison, not a tuned contender."""
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=50,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    return _StringifyPipeline([("prep", build_preprocessor()), ("clf", clf)]).fit(X, y)


def fit_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    seed: int,
    X_es: pd.DataFrame,
    y_es: np.ndarray,
    early_stopping_rounds: int,
):
    """LightGBM, early-stopped on PR-AUC (average precision) of the time-ordered es set."""
    import lightgbm as lgb

    p = {k: v for k, v in params.items() if k != "class_weight"}
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=4000,
        metric="average_precision",
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        subsample_freq=1,
        scale_pos_weight=pos_weight(params.get("class_weight", "none"), y),
        verbose=-1,
        **p,
    )
    model.fit(
        X,
        y,
        eval_set=[
            (X_es, y_es)
        ],  # `eval_set` still works across lightgbm 4.x (newer name: eval_X/eval_y)
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False)
        ],
    )
    return model


def _ebm_bags(approved_at: pd.Series, outer_bags: int, es_fraction: float, seed: int) -> np.ndarray:
    """Bag definitions: latest `es_fraction` of rows (by TIME) = validation (-1) in every bag;
    the rest are bootstrap-resampled per bag (+k = included k times).

    Why: EBM's default early-stopping split is random; time-ordered validation respects
    CLAUDE.md rule 3 while keeping outer bagging's variance reduction.
    """
    n = len(approved_at)
    cut = approved_at.quantile(1 - es_fraction)
    is_val = (approved_at > cut).to_numpy()
    rng = np.random.default_rng(seed)
    bags = np.zeros((n, outer_bags), dtype=np.int8)
    tr_idx = np.flatnonzero(~is_val)
    for b in range(outer_bags):
        bags[tr_idx, b] = rng.multinomial(len(tr_idx), np.full(len(tr_idx), 1 / len(tr_idx)))
        bags[is_val, b] = -1
    return bags


def fit_ebm(
    X: pd.DataFrame,
    y: np.ndarray,
    approved_at: pd.Series,
    params: dict,
    seed: int,
    es_fraction: float,
    outer_bags: int = 4,
):
    from interpret.glassbox import ExplainableBoostingClassifier

    ebm = ExplainableBoostingClassifier(
        max_bins=params["max_bins"],
        interactions=params["interactions"],
        learning_rate=params["learning_rate"],
        max_leaves=params["max_leaves"],
        min_samples_leaf=params["min_samples_leaf"],
        outer_bags=outer_bags,
        inner_bags=0,
        max_rounds=5000,
        early_stopping_rounds=50,
        random_state=seed,
    )
    bags = _ebm_bags(approved_at.reset_index(drop=True), outer_bags, es_fraction, seed)
    return ebm.fit(X.reset_index(drop=True), y, bags=bags)
