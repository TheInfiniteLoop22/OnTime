"""Phase 7 tests: policy arithmetic, slices, rolling recalibration (leak guard), reliability."""

import numpy as np
import pandas as pd
import pytest

from ontime.evaluation.calibration import (
    expected_calibration_error,
    reliability_table,
    rolling_recalibrate,
)
from ontime.evaluation.policy import simulate_policy
from ontime.evaluation.report import render_results_md
from ontime.evaluation.slices import slice_table

T = pd.Timestamp


# ------------------------------------------------------------------ policy, by hand
def _two_days():
    dates = pd.Series([T("2018-03-01 08:00")] * 5 + [T("2018-03-02 09:00")] * 5)
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.6, 0.1, 0.3, 0.1, 0.8, 0.2, 0.3, 0.4])
    return dates, y, s


def test_policy_arithmetic_matches_hand_calculation():
    dates, y, s = _two_days()
    pol = simulate_policy(dates, y, {"m": s}, k_grid=[0.4], p_grid=[0.5], seed=0, n_boot=50)
    g = pol.set_index("strategy")
    # k=40% -> 2 picks/day (4 total). Model captures 3 of 3 late orders.
    assert g.loc["m", "captured_late"] == 3
    assert g.loc["m", "prevented_expected"] == pytest.approx(0.5 * 3)
    # random expectation: day1 2*(2/5)=0.8, day2 2*(1/5)=0.4 -> 1.2 late orders captured
    assert g.loc["random", "captured_late"] == pytest.approx(1.2)
    # oracle: min(2, 2)=2 on day 1, min(2, 1)=1 on day 2 -> 3
    assert g.loc["oracle", "captured_late"] == pytest.approx(3.0)
    assert g.loc["m", "extra_vs_random"] == pytest.approx(0.5 * (3 - 1.2))
    assert g.loc["m", "recall"] == pytest.approx(1.0)


def test_policy_model_is_between_random_and_oracle_for_noisy_scores():
    rng = np.random.default_rng(0)
    n = 4000
    dates = pd.Series(
        pd.to_datetime("2018-05-01") + pd.to_timedelta(rng.integers(0, 20, n), unit="D")
    )
    y = rng.binomial(1, 0.1, n)
    s = y * 0.4 + rng.random(n)
    pol = simulate_policy(dates, y, {"m": s}, [0.1], [0.3], seed=1, n_boot=50).set_index("strategy")
    assert (
        pol.loc["random", "captured_late"]
        < pol.loc["m", "captured_late"]
        <= pol.loc["oracle", "captured_late"]
    )
    lo, hi = pol.loc["m", "extra_vs_random_ci_low"], pol.loc["m", "extra_vs_random_ci_high"]
    assert lo <= pol.loc["m", "extra_vs_random"] <= hi


# ------------------------------------------------------------------ slices
def test_slice_table_counts_and_small_slice_metrics_are_withheld():
    rng = np.random.default_rng(0)
    n = 3000
    d = pd.DataFrame(
        {"is_late": rng.binomial(1, 0.15, n), "grp": rng.choice(["a", "b"], n, p=[0.98, 0.02])}
    )
    p = d["is_late"] * 0.3 + rng.random(n)
    t = slice_table(d, p.to_numpy(), ["grp"]).set_index("value")
    assert t["orders"].sum() == n
    assert np.isfinite(t.loc["a", "roc_auc"])
    assert np.isnan(t.loc["b", "roc_auc"])  # <30 late orders -> not reported


def test_slice_flags_group_where_model_is_random():
    rng = np.random.default_rng(1)
    n = 6000
    grp = rng.choice(["good", "bad"], n)
    y = rng.binomial(1, 0.2, n)
    p = np.where(grp == "good", y * 0.6 + rng.random(n), rng.random(n))  # no signal in "bad"
    t = slice_table(pd.DataFrame({"is_late": y, "grp": grp}), p, ["grp"]).set_index("value")
    assert bool(t.loc["bad", "weak"]) and not bool(t.loc["good", "weak"])


# ------------------------------------------------------------------ calibration helpers
def test_reliability_and_ece_zero_for_perfect_and_large_for_inflated():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.5, 40000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p * 2, 0, 1)) > 0.1
    t = reliability_table(y, p, 10)
    assert len(t) == 10 and t["orders"].sum() == len(y)


def test_rolling_recalibration_uses_only_outcomes_delivered_before_month_start():
    rng = np.random.default_rng(3)
    n = 1500
    approved = pd.Series(T("2018-03-01") + pd.to_timedelta(rng.integers(0, 90 * 24, n), unit="h"))
    delivered = approved + pd.to_timedelta(rng.integers(2, 12, n), unit="D")
    y = rng.binomial(1, 0.1, n)
    raw = np.clip(0.1 + 0.2 * (y - 0.1) + rng.normal(0, 0.05, n), 0.01, 0.99)
    sc = pd.DataFrame({"approved_at": approved, "delivered_at": delivered, "y": y, "raw": raw})
    base = rolling_recalibrate(sc, "2018-05-01", "2018-05-31")

    # Change outcomes of orders delivered AFTER May 1: May recalibration must not change.
    after = sc["delivered_at"] >= T("2018-05-01")
    tampered = sc.copy()
    tampered.loc[after, "y"] = 1 - tampered.loc[after, "y"]
    again = rolling_recalibrate(tampered, "2018-05-01", "2018-05-31")
    pd.testing.assert_series_equal(base, again)
    # and only May orders get values
    may = (sc["approved_at"] >= T("2018-05-01")) & (sc["approved_at"] < T("2018-06-01"))
    assert base[~may].isna().all() and base[may].notna().all()


# ------------------------------------------------------------------ report rendering
def test_report_renders_from_minimal_results_without_typed_numbers():
    m = {"value": 0.2, "ci95": [0.1, 0.3]}
    e = {
        "metrics": {"pr_auc": m, "roc_auc": {"value": 0.6}, "brier": {"value": 0.1},
                    "log_loss": {"value": 0.3}},
        "top_k": [{"k": 0.1, "precision": 0.3, "recall": 0.2, "lift": 2.0,
                   "ci95": {"lift": [1.5, 2.5], "precision": [0.2, 0.4]}}],
        "calibrator": "none", "ece_deployed": 0.01,
        "rolling_recalibration": {"brier": 0.09, "ece": 0.02},
        "description": "rule",
    }  # fmt: skip
    r = {
        "dry_run": True, "primary_model": "logreg",
        "primary_selection_rule": "rule text",
        "population": {"orders": 10, "late_orders": 1, "late_rate": 0.1, "from": "a", "to": "b",
                       "train_late_rate": 0.05},
        "models": {k: e for k in ["logreg", "random_forest", "lightgbm", "ebm"]},
        "baselines": {"prior_rate": e, "rule": e},
        "constant_oracle_brier": 0.09,
        "paired_pr_auc_diff": {k: {"diff": 0.01, "ci95": [-0.01, 0.02]} for k in
                               ["ebm_minus_lightgbm", "logreg_minus_lightgbm", "ebm_minus_logreg",
                                "logreg_minus_rule"]},
        "monthly": [], "slices_primary_weak": [], "slices_file": "slices.csv",
        "policy": [], "error_analysis_files": ["e.csv"],
    }  # fmt: skip
    md = render_results_md(r)
    assert "includes zero" in md  # sentence derived from CI, not typed
    assert "dryrun_pr_curves.png" in md and "dryrun_policy_curves.png" in md
