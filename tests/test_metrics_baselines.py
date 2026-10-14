"""Phase 5 tests: metrics vs sklearn, top-k by hand, bootstrap, baselines, test-split lock."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from ontime.evaluation.metrics import (
    bootstrap_metric_ci,
    bootstrap_topk_ci,
    classification_metrics,
    daily_topk_table,
    topk_metrics,
)
from ontime.models.baselines import PriorRateBaseline, RuleBaseline
from ontime.models.splits import assert_no_time_overlap, select_split

T = pd.Timestamp


# ------------------------------------------------------------------ metrics vs sklearn
def test_classification_metrics_match_sklearn():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7, 0.05, 0.6, 0.55, 0.3])
    m = classification_metrics(y, p)
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    assert m["log_loss"] == pytest.approx(log_loss(y, p))
    assert m["base_rate"] == pytest.approx(0.4)


def test_log_loss_is_finite_for_hard_zero_one_probabilities():
    m = classification_metrics(np.array([0, 1, 1, 0]), np.array([0.0, 1.0, 0.0, 1.0]))
    assert np.isfinite(m["log_loss"])


def test_constant_scores_have_prauc_equal_base_rate_and_auc_half():
    y = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    m = classification_metrics(y, np.full(10, 0.3))
    assert m["pr_auc"] == pytest.approx(y.mean())
    assert m["roc_auc"] == pytest.approx(0.5)


# ------------------------------------------------------------------ top-k, hand-computed
def _two_days():
    # day 1: 5 orders, 2 late; day 2: 5 orders, 1 late. Scores rank the late ones first/second.
    dates = pd.Series([T("2018-03-01 08:00")] * 5 + [T("2018-03-02 09:00")] * 5)
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.6, 0.1, 0.3, 0.1, 0.8, 0.2, 0.3, 0.4])
    return dates, y, s


def test_topk_precision_recall_lift_by_hand():
    dates, y, s = _two_days()
    # k=40% -> ceil(0.4*5)=2 picks per day. Day 1 picks scores .9,.6 (both late: 2/2).
    # Day 2 picks .8 (late) and .4 (not late): 1/2. Pooled: 3 captured / 4 picked.
    r = topk_metrics(dates, y, s, 0.4, seed=0)
    assert r["selected"] == 4 and r["captured_late"] == 3
    assert r["precision"] == pytest.approx(3 / 4)
    assert r["recall"] == pytest.approx(3 / 3)  # all 3 late orders captured
    assert r["lift"] == pytest.approx((3 / 4) / (3 / 10))


def test_topk_quota_is_per_day_not_global():
    dates, y, s = _two_days()
    t = daily_topk_table(dates, y, s, 0.2, seed=0)  # ceil(0.2*5)=1 per day
    assert t["selected"].tolist() == [1, 1]


def test_topk_ties_are_random_but_reproducible():
    dates = pd.Series([T("2018-03-01")] * 100)
    y = np.array([1] * 50 + [0] * 50)
    const = np.zeros(100)
    a = topk_metrics(dates, y, const, 0.1, seed=1)
    b = topk_metrics(dates, y, const, 0.1, seed=1)
    c = topk_metrics(dates, y, const, 0.1, seed=2)
    assert a == b  # same seed -> identical
    assert a["selected"] == 10 and c["selected"] == 10


def test_perfect_scores_give_perfect_topk_precision():
    dates = pd.Series([T("2018-03-01")] * 20)
    y = np.array([1] * 4 + [0] * 16)
    r = topk_metrics(dates, y, y.astype(float), 0.2, seed=0)  # picks exactly 4
    assert r["precision"] == 1.0 and r["recall"] == 1.0


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_ci_brackets_point_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 600)
    p = np.clip(0.2 + 0.3 * (y - 0.2) + rng.normal(0, 0.2, 600), 0.01, 0.99)
    point = classification_metrics(y, p)
    a = bootstrap_metric_ci(y, p, n_boot=200, seed=5)
    b = bootstrap_metric_ci(y, p, n_boot=200, seed=5)
    assert a == b
    lo, hi = a["pr_auc"]
    assert lo <= point["pr_auc"] <= hi


def test_daily_bootstrap_resamples_days_and_is_reproducible():
    rng = np.random.default_rng(1)
    dates = pd.Series(
        pd.to_datetime("2018-03-01") + pd.to_timedelta(rng.integers(0, 30, 900), unit="D")
    )
    y = rng.binomial(1, 0.15, 900)
    s = y * 0.5 + rng.random(900)
    a = bootstrap_topk_ci(dates, y, s, 0.1, n_boot=200, seed=3)
    b = bootstrap_topk_ci(dates, y, s, 0.1, n_boot=200, seed=3)
    assert a == b
    pt = topk_metrics(dates, y, s, 0.1, seed=3)
    lo, hi = a["precision"]
    assert lo <= pt["precision"] <= hi


# ------------------------------------------------------------------ baselines
def _train(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    same = rng.binomial(1, 0.4, n)
    prom = rng.uniform(5, 45, n)
    dist = rng.uniform(10, 2500, n)
    p = 0.02 + 0.20 * ((same == 0) & (prom < 20))
    return pd.DataFrame(
        {
            "same_state": same,
            "promised_days": prom,
            "distance_km": dist,
            "is_late": rng.binomial(1, p),
        }
    )


def test_prior_rate_baseline_predicts_train_rate():
    y = np.array([1, 0, 0, 0])
    m = PriorRateBaseline().fit(y)
    assert m.rate_ == 0.25
    assert m.predict_proba(pd.DataFrame({"a": [1, 2, 3]})).tolist() == [0.25] * 3


def test_rule_baseline_recovers_planted_rule_and_uses_train_only():
    train = _train()
    rule = RuleBaseline().fit(train)
    assert rule.family == "interstate_tight_promise"
    assert 15 < rule.threshold < 25  # planted at 20
    assert rule.p_flagged > rule.p_unflagged
    # fitting depends ONLY on the frame passed to fit(): predicting a different frame changes
    # nothing about the fitted parameters
    before = (rule.family, rule.threshold, rule.p_flagged, rule.p_unflagged)
    rule.predict_proba(_train(seed=99))
    assert before == (rule.family, rule.threshold, rule.p_flagged, rule.p_unflagged)


# ------------------------------------------------------------------ split lock
def _table():
    return pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "approved_at": [T("2018-01-01"), T("2018-03-01"), T("2018-06-01")],
        }
    )


def test_test_split_is_locked_by_default():
    t = _table()
    assert len(select_split(t, "train")) == 1
    assert len(select_split(t, "validation")) == 1
    with pytest.raises(PermissionError):
        select_split(t, "test")
    assert len(select_split(t, "test", final_evaluation=True)) == 1


def test_unknown_split_rejected_and_overlap_detected():
    with pytest.raises(ValueError):
        select_split(_table(), "holdout")
    assert_no_time_overlap(_table())
    bad = _table()
    bad.loc[0, "approved_at"] = T("2018-03-15")  # train after validation start
    with pytest.raises(ValueError):
        assert_no_time_overlap(bad)
