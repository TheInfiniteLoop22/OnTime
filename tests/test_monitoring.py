"""Drift monitoring: PSI behaves like PSI; month status rules."""

import numpy as np
import pandas as pd
import pytest

from ontime.monitoring.drift import (
    month_status,
    monthly_psi,
    performance_by_month,
    psi_categorical,
    psi_numeric,
    status_from_psi,
)


def test_psi_is_near_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 20000))
    same = pd.Series(rng.normal(0, 1, 20000))
    shifted = pd.Series(rng.normal(1.5, 1, 20000))
    assert psi_numeric(ref, same) < 0.02
    assert psi_numeric(ref, shifted) > 0.5


def test_psi_is_non_negative_and_zero_for_identical():
    s = pd.Series(np.arange(1000, dtype=float))
    assert psi_numeric(s, s) == pytest.approx(0.0, abs=1e-9)
    assert psi_numeric(s, s.iloc[::-1].reset_index(drop=True)) >= 0.0


def test_psi_counts_missing_values_as_a_bin():
    ref = pd.Series(np.arange(1000, dtype=float))
    cur = ref.copy()
    cur.iloc[:300] = np.nan  # 30% newly missing
    assert psi_numeric(ref, cur) > 0.1


def test_psi_categorical_new_level_registers():
    ref = pd.Series(["a"] * 90 + ["b"] * 10)
    cur = pd.Series(["a"] * 50 + ["b"] * 10 + ["c"] * 40)
    assert psi_categorical(ref, cur) > 0.25
    assert psi_categorical(ref, ref) == pytest.approx(0.0, abs=1e-9)


def test_constant_reference_does_not_crash():
    ref = pd.Series([1.0] * 500)
    cur = pd.Series([1.0] * 250 + [2.0] * 250)
    assert psi_numeric(ref, cur) > 0.25


def test_status_thresholds():
    assert status_from_psi(0.05, 0.1, 0.25) == "stable"
    assert status_from_psi(0.12, 0.1, 0.25) == "watch"
    assert status_from_psi(0.30, 0.1, 0.25) == "alert"


def _frame(n=3000, seed=1):
    from ontime.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURE_COLUMNS})
    for c in CATEGORICAL_FEATURES:
        df[c] = rng.choice(["a", "b"], n)
    t = pd.date_range("2018-01-01", "2018-04-30", periods=n)
    df["approved_at"] = t
    df["split"] = np.where(t < pd.Timestamp("2018-03-01"), "train", "test")
    df["is_late"] = rng.binomial(1, 0.1, n)
    return df


def test_monthly_psi_only_reports_months_after_reference_and_flags_a_shifted_feature():
    df = _frame()
    df.loc[df["approved_at"] >= pd.Timestamp("2018-04-01"), "distance_km"] += 5.0
    out = monthly_psi(df, df["split"] == "train", min_orders=100)
    assert out["month"].min() >= pd.Timestamp("2018-03-01")
    apr = out[out["month"] == pd.Timestamp("2018-04-01")].set_index("feature")["psi"]
    assert apr["distance_km"] > 0.25 and apr["distance_km"] == apr.max()


def test_performance_by_month_and_status_rules():
    rng = np.random.default_rng(0)
    n = 4000
    t = pd.date_range("2018-02-01", "2018-05-31", periods=n)
    y = rng.binomial(1, 0.15, n)
    good = np.where(t < pd.Timestamp("2018-04-01"), y + rng.normal(0, 0.5, n), rng.normal(0, 1, n))
    perf = performance_by_month(pd.DataFrame({"approved_at": t, "is_late": y, "score": good}))
    assert perf.set_index("month").loc[pd.Timestamp("2018-02-01"), "roc_auc"] > 0.8
    assert perf.set_index("month").loc[pd.Timestamp("2018-05-01"), "roc_auc"] < 0.6
    psi = pd.DataFrame(
        {"month": [m for m in perf["month"] for _ in range(4)], "feature": list("abcd") * len(perf),
         "psi": 0.01, "orders": 1, "baseline": "trailing"}
    )  # fmt: skip
    st = month_status(psi, perf, 0.1, 0.25).set_index("month")
    assert st.loc[pd.Timestamp("2018-02-01"), "status"] == "stable"
    assert st.loc[pd.Timestamp("2018-05-01"), "status"] == "alert"  # performance fell > 0.05


def test_expected_drift_features_do_not_drive_alerts():
    months = pd.date_range("2018-02-01", periods=3, freq="MS")
    perf = pd.DataFrame({"month": months, "orders": 1000, "late_rate": 0.1,
                         "roc_auc": [0.7, 0.7, 0.7], "pr_auc": 0.1, "lift_at_10": 2.0})  # fmt: skip
    rows = [{"month": m, "feature": f, "psi": 5.0 if f.startswith("state") else 0.01,
             "orders": 1, "baseline": "trailing"}
            for m in months for f in ["state_a", "state_b", "state_c", "real_a"]]  # fmt: skip
    psi = pd.DataFrame(rows)
    noisy = month_status(psi, perf, 0.1, 0.25)
    quiet = month_status(psi, perf, 0.1, 0.25, exclude=("state_a", "state_b", "state_c"))
    assert (noisy["status"] == "alert").all()
    assert (quiet["status"] == "stable").all()


def test_trailing_reference_flags_sudden_change_only_in_the_changed_month():
    rng = np.random.default_rng(3)
    n = 6000
    t = pd.date_range("2018-01-01", "2018-06-30", periods=n)
    from ontime.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS

    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURE_COLUMNS})
    for c in CATEGORICAL_FEATURES:
        df[c] = rng.choice(["a", "b"], n)
    df["approved_at"] = t
    df.loc[t >= pd.Timestamp("2018-06-01"), "promised_days"] -= 3.0
    out = monthly_psi(df, trailing_months=3, min_orders=100, first_month=pd.Timestamp("2018-04-01"))
    pm = out[out["feature"] == "promised_days"].set_index("month")["psi"]
    assert pm[pd.Timestamp("2018-06-01")] > 0.25
    assert pm[pd.Timestamp("2018-04-01")] < 0.1 and pm[pd.Timestamp("2018-05-01")] < 0.1


def test_monthly_psi_requires_exactly_one_reference():
    df = pd.DataFrame({"approved_at": pd.date_range("2018-01-01", periods=10)})
    with pytest.raises(ValueError):
        monthly_psi(df)
