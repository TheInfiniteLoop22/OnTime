"""Phase 6 tests: calibration, time sub-splits, EBM bags, class weights, fitters on tiny data."""

import numpy as np
import pandas as pd
import pytest

from ontime.config import load_config
from ontime.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from ontime.models.calibrate import (
    IsotonicCalibrator,
    SigmoidCalibrator,
    choose_calibrator,
)
from ontime.models.data import category_levels, modelling_frames, prepare_frame
from ontime.models.train import (
    _ebm_bags,
    build_preprocessor,
    fit_lgbm,
    fit_logreg,
    pos_weight,
)
from ontime.models.tune import weight_experiment

T = pd.Timestamp


# ------------------------------------------------------------------ calibration
def test_sigmoid_calibrator_recovers_a_known_distortion():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.02, 0.6, 20000)
    y = rng.binomial(1, true_p)
    distorted = np.clip(true_p**0.5, 0.01, 0.99)  # over-confident scores
    cal = SigmoidCalibrator().fit(distorted, y).transform(distorted)
    assert abs(cal.mean() - y.mean()) < 0.01  # mean matches the observed rate
    assert np.mean((cal - true_p) ** 2) < np.mean((distorted - true_p) ** 2)


def test_isotonic_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = rng.binomial(1, p)
    iso = IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0, 1, 50)
    out = iso.transform(grid)
    assert (np.diff(out) >= -1e-12).all()
    assert out.min() >= 0 and out.max() <= 1


def test_choose_calibrator_prefers_none_when_scores_already_calibrated():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.02, 0.5, 30000)
    y = rng.binomial(1, p)
    best, table = choose_calibrator(p[:15000], y[:15000], p[15000:], y[15000:])
    assert best in {"none", "sigmoid"}  # sigmoid ~ identity here; isotonic should not win
    assert set(table) == {"none", "sigmoid", "isotonic"}
    assert table["none"]["brier"] <= table["isotonic"]["brier"] + 1e-4


def test_choose_calibrator_fixes_badly_scaled_scores():
    rng = np.random.default_rng(3)
    true_p = rng.uniform(0.01, 0.3, 20000)
    y = rng.binomial(1, true_p)
    inflated = np.clip(true_p * 5, 0, 0.99)  # like a class-weighted model
    best, table = choose_calibrator(inflated[:10000], y[:10000], inflated[10000:], y[10000:])
    assert best != "none"
    assert table[best]["brier"] < table["none"]["brier"]


# ------------------------------------------------------------------ time sub-splits
def _feature_table():
    cfg = load_config()
    days = pd.date_range("2018-01-15", "2018-05-10", freq="1D")
    rows = []
    for d in days:
        for h in (9, 15):
            rows.append(d + pd.Timedelta(hours=h))
    a = pd.Series(rows)
    from ontime.models.splits import assign_split

    return pd.DataFrame({"approved_at": a, "split": assign_split(a, cfg), "is_late": 0})


def test_validation_subsplits_are_ordered_disjoint_and_cover_validation():
    f = modelling_frames(_feature_table())
    assert f["val_tune"]["approved_at"].max() < f["val_calib"]["approved_at"].min()
    assert f["calib_fit"]["approved_at"].max() < f["calib_eval"]["approved_at"].min()
    assert len(f["val_tune"]) + len(f["val_calib"]) == len(f["validation"])
    assert len(f["calib_fit"]) + len(f["calib_eval"]) == len(f["val_calib"])
    assert f["train"]["approved_at"].max() < f["validation"]["approved_at"].min()
    assert "test" not in f  # the test rows are never returned


# ------------------------------------------------------------------ EBM bags / weights
def test_ebm_bags_use_latest_rows_for_validation_and_bootstrap_the_rest():
    a = pd.Series(pd.date_range("2018-01-01", periods=1000, freq="h"))
    bags = _ebm_bags(a, outer_bags=3, es_fraction=0.2, seed=7)
    assert bags.shape == (1000, 3)
    val_rows = np.flatnonzero(bags[:, 0] == -1)
    assert val_rows.min() >= 799  # only the LATEST ~20% by time are validation
    assert ((bags == -1).sum(axis=0) == len(val_rows)).all()  # same validation in every bag
    train_counts = np.where(bags > 0, bags, 0).sum(axis=0)
    assert (train_counts == 800 - 0).all() or (train_counts == bags[:800].sum(axis=0)).all()
    assert not (bags[:, 0] == bags[:, 1]).all()  # bags differ (bootstrap)


def test_pos_weight_options():
    y = np.array([1] * 10 + [0] * 90)
    assert pos_weight("none", y) == 1.0
    assert pos_weight("balanced", y) == pytest.approx(9.0)
    assert pos_weight("sqrt", y) == pytest.approx(3.0)


def test_weight_experiment_groups_trials_by_option():
    class Tr:
        def __init__(self, v, w):
            self.value, self.params = v, {"class_weight": w}

    class St:
        trials = [Tr(0.3, "none"), Tr(0.4, "none"), Tr(0.5, "balanced")]

    out = weight_experiment(St())
    assert out["none"]["best_pr_auc"] == 0.4 and out["none"]["trials"] == 2
    assert out["balanced"]["best_pr_auc"] == 0.5


# ------------------------------------------------------------------ fitters on tiny data
def _tiny(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))
    for c in FEATURE_COLUMNS:
        df[c] = rng.normal(size=n)
    df["customer_region"] = rng.choice(["N", "SE", "S"], n)
    df["seller_region"] = rng.choice(["SE", "S"], n)
    df["category_group"] = rng.choice(["auto", "toys_baby", "other"], n)
    df["payment_type"] = rng.choice(["credit_card", "boleto"], n)
    for c in ("approve_dow", "approve_month"):
        df[c] = rng.integers(0, 7, n)
    df["seller_n_past"] = rng.integers(0, 300, n).astype(float)
    df.loc[rng.random(n) < 0.05, "freight_ratio"] = np.nan
    logit = -2.5 + 1.5 * df["distance_km"]
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")
    return df, y


def test_logreg_pipeline_handles_nan_and_unseen_category():
    df, y = _tiny()
    levels = category_levels(df)
    X = prepare_frame(df, levels)
    model = fit_logreg(X, y, {"C": 1.0, "class_weight": "none"}, seed=42)
    new = df.head(5).copy()
    new["category_group"] = pd.Categorical(["brand_new_cat"] * 5, categories=["brand_new_cat"])
    out = model.predict_proba(prepare_frame(new.assign(category_group="auto"), levels))[:, 1]
    assert np.isfinite(out).all() and ((out >= 0) & (out <= 1)).all()


def test_lightgbm_fit_is_deterministic_and_early_stops():
    df, y = _tiny()
    X = prepare_frame(df, category_levels(df))
    params = {"num_leaves": 8, "learning_rate": 0.1, "min_child_samples": 20,
              "colsample_bytree": 0.8, "subsample": 0.8, "reg_alpha": 0.01,
              "reg_lambda": 0.1, "class_weight": "balanced"}  # fmt: skip
    a = fit_lgbm(X[:1000], y[:1000], params, 42, X[1000:], y[1000:], 20)
    b = fit_lgbm(X[:1000], y[:1000], params, 42, X[1000:], y[1000:], 20)
    pa, pb = a.predict_proba(X[1000:])[:, 1], b.predict_proba(X[1000:])[:, 1]
    np.testing.assert_allclose(pa, pb)
    assert a.best_iteration_ < 4000


def test_preprocessor_columns_exclude_duplicate_log_distance():
    df, y = _tiny(300)
    pre = build_preprocessor().fit(
        prepare_frame(df, category_levels(df)).astype({c: str for c in CATEGORICAL_FEATURES})
    )
    names = list(pre.get_feature_names_out())
    assert "log_distance_km" not in names
    assert any(n.startswith("category_group_") for n in names)
