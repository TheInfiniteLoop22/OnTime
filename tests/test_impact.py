"""Phase 3 tests: review de-duplication, effect direction, bootstrap, dose bins, rate tables."""

import numpy as np
import pandas as pd
import pytest

from ontime.analysis import impact
from ontime.analysis.eda import add_eda_columns, rate_by

T = pd.Timestamp


def test_one_review_per_order_keeps_latest_answer():
    r = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "order_id": ["a", "a", "b"],
            "review_score": [1, 5, 3],
            "review_creation_date": [T("2018-01-01")] * 3,
            "review_answer_timestamp": [T("2018-01-02"), T("2018-01-05"), T("2018-01-03")],
        }
    )
    out = impact.one_review_per_order(r).set_index("order_id")
    assert len(out) == 2
    assert out.loc["a", "review_score"] == 5  # the later answer wins


def _synthetic(n=2000, penalty=-1.5, seed=0):
    rng = np.random.default_rng(seed)
    late = rng.binomial(1, 0.1, n)
    score = np.clip(np.round(4.5 + penalty * late + rng.normal(0, 0.8, n)), 1, 5).astype(int)
    return pd.DataFrame(
        {"is_late": late, "review_score": score, "low_score": (score <= 2).astype(int)}
    )


def test_mann_whitney_detects_penalty_direction():
    mw = impact.mann_whitney(_synthetic())
    assert mw["p_value"] < 1e-6
    assert mw["prob_ontime_scores_higher_than_late"] > 0.8


def test_mann_whitney_no_effect_is_near_half():
    mw = impact.mann_whitney(_synthetic(penalty=0.0, seed=1))
    assert 0.4 < mw["prob_ontime_scores_higher_than_late"] < 0.6


def test_bootstrap_ci_contains_point_estimate_and_is_reproducible():
    d = _synthetic()
    a = impact.bootstrap_differences(d, n_boot=200, seed=42)
    b = impact.bootstrap_differences(d, n_boot=200, seed=42)
    lo, hi = a["mean_score_diff_ci95"]
    assert lo <= a["mean_score_diff"] <= hi
    assert a == b  # same seed -> identical result (determinism rule)
    assert a["mean_score_diff"] < 0


def test_dose_response_bins_by_days_vs_estimate():
    d = pd.DataFrame(
        {
            "days_vs_estimate": [-20, -3, 0, 2, 5, 10, 30],
            "review_score": [5, 4, 4, 3, 2, 1, 1],
            "low_score": [0, 0, 0, 0, 1, 1, 1],
        }
    )
    t = impact.dose_response(d).set_index("timing")
    assert t.loc[">=8 days early", "orders"] == 1
    assert t.loc["on the promised day", "orders"] == 1
    assert t.loc["1-3 days late", "orders"] == 1
    assert t.loc[">14 days late", "orders"] == 1
    assert t["orders"].sum() == len(d)


def test_timing_of_review_splits_groups_and_sums_to_one():
    d = pd.DataFrame(
        {
            "is_late": [1, 1, 1, 0],
            "answered_before_delivery": [True, True, False, False],
            "review_score": [1, 2, 4, 5],
        }
    )
    t = impact.timing_of_review(d).set_index("group")
    assert t.loc["late, reviewed BEFORE arrival", "orders"] == 2
    assert t.loc["late, reviewed BEFORE arrival", "share_of_group_orders"] == pytest.approx(2 / 3)


def test_rate_by_wilson_interval_brackets_rate_and_respects_min_n():
    d = pd.DataFrame(
        {"g": ["a"] * 100 + ["b"] * 10, "is_late": [1] * 10 + [0] * 90 + [1] * 5 + [0] * 5}
    )
    t = rate_by(d, "g", min_n=50)
    assert t["g"].tolist() == ["a"]  # b dropped: too few orders
    row = t.iloc[0]
    assert row["late_rate"] == pytest.approx(0.1)
    assert row["ci_low"] < 0.1 < row["ci_high"]


def test_add_eda_columns_target_consistency():
    df = pd.DataFrame(
        {
            "order_approved_at": [T("2018-01-01 10:00")],
            "order_delivered_customer_date": [T("2018-01-12 09:00")],
            "order_estimated_delivery_date": [T("2018-01-10")],
            "delivery_days": [10.96],
            "seller_lat": [-23.5],
            "seller_lng": [-46.6],
            "customer_lat": [-22.9],
            "customer_lng": [-43.2],
            "seller_state": ["SP"],
            "customer_state": ["RJ"],
        }
    )
    out = add_eda_columns(df)
    assert out.loc[0, "days_vs_estimate"] == 2  # delivered date is 2 calendar days after estimate
    assert out.loc[0, "route"] == "SP->RJ"
    assert out.loc[0, "same_state"] == False  # noqa: E712
    assert 300 < out.loc[0, "distance_km"] < 450
