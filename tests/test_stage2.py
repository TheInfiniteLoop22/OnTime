"""Stage-2 (handover-time) features: leakage tests + vectorized-vs-naive equality."""

import numpy as np
import pandas as pd
import pytest

from ontime.stage2.features import STAGE2_FEATURES, STAGE2_FORBIDDEN, backlog_share, stage2_features

T = pd.Timestamp
ALPHA = 20.0


def _world(seed=0, n=500):
    rng = np.random.default_rng(seed)
    base = T("2018-01-01")
    appr = base + pd.to_timedelta(rng.integers(0, 120 * 24 * 3600, n), unit="s")
    handover = appr + pd.to_timedelta(rng.integers(0, 5 * 24 * 3600, n), unit="s")
    deliv = handover + pd.to_timedelta(rng.integers(2 * 24 * 3600, 30 * 24 * 3600, n), unit="s")
    never = rng.random(n) < 0.06
    limit = appr + pd.to_timedelta(rng.integers(1, 4, n), unit="D")
    ev = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(n)],
            "primary_seller_id": rng.choice([f"s{i}" for i in range(8)], n),
            "customer_state": rng.choice(["SP", "RJ", "BA"], n),
            "order_approved_at": appr,
            "max_shipping_limit_date": limit,
            "handover_at": handover,
            "delivered_at": pd.Series(deliv).where(~never),
            "late_handover": (handover > limit).astype(int),
        }
    )
    q = ev.assign(order_estimated_delivery_date=deliv + pd.Timedelta(days=3))
    return ev, q


def test_feature_list_has_no_forbidden_names():
    assert not set(STAGE2_FEATURES) & set(STAGE2_FORBIDDEN)
    assert len(set(STAGE2_FEATURES)) == len(STAGE2_FEATURES)


def test_own_handover_features_by_hand():
    ev, q = _world(n=50)
    f = stage2_features(q, ev, ALPHA)
    r = q.iloc[0]
    t1 = max(r["order_approved_at"], r["handover_at"])
    assert f.loc[0, "days_to_handover"] == pytest.approx(
        (t1 - r["order_approved_at"]).total_seconds() / 86400
    )
    assert f.loc[0, "handover_vs_limit_days"] == pytest.approx(
        (r["handover_at"] - r["max_shipping_limit_date"]).total_seconds() / 86400
    )


def test_carrier_before_approval_uses_approval_as_decision_time():
    ev, q = _world(n=30)
    q.loc[0, "handover_at"] = q.loc[0, "order_approved_at"] - pd.Timedelta(days=2)
    f = stage2_features(q, ev, ALPHA)
    assert f.loc[0, "t1"] == q.loc[0, "order_approved_at"]
    assert f.loc[0, "days_to_handover"] == 0.0
    assert f.loc[0, "handover_vs_limit_days"] < 0  # raw (unclipped) comparison with the limit


def test_backlog_ignores_deliveries_not_yet_known_at_t1():
    """Moving deliveries that happen AT/AFTER the query day must not change the backlog."""
    ev, q = _world(seed=1)
    qs = q.iloc[:80]
    t1 = qs[["order_approved_at", "handover_at"]].max(axis=1)
    base = backlog_share(ev, qs["customer_state"], t1, ALPHA)
    ev2 = ev.copy()
    # for every event, push deliveries that are >= the latest query day 10 days later: no query
    # (all have day <= that) can know about them, so nothing may change
    latest_day = t1.max().normalize()
    future = ev2["delivered_at"] >= latest_day
    ev2.loc[future, "delivered_at"] += pd.Timedelta(days=10)
    again = backlog_share(ev2, qs["customer_state"], t1, ALPHA)
    np.testing.assert_allclose(base, again, equal_nan=True)


def test_backlog_reacts_when_a_delivery_becomes_known():
    ev, q = _world(seed=2)
    qs = q.iloc[:40]
    t1 = qs[["order_approved_at", "handover_at"]].max(axis=1)
    base = backlog_share(ev, qs["customer_state"], t1, ALPHA)
    ev2 = ev.copy()
    ev2["delivered_at"] = pd.NaT  # nothing is ever delivered: backlog can only go up
    worse = backlog_share(ev2, qs["customer_state"], t1, ALPHA)
    ok = ~np.isnan(base)
    assert (worse[ok] >= base[ok] - 1e-12).all() and (worse[ok] > base[ok]).any()


def _naive_backlog(ev, state, t1, alpha, far=28, near=8):
    day = t1.normalize()
    lo, hi = day - pd.Timedelta(days=far), day - pd.Timedelta(days=near)  # days [lo, hi] inclusive
    h = ev["handover_at"].dt.normalize()
    coh = ev[(h >= lo) & (h <= hi)]
    und_all = (coh["delivered_at"].isna() | (coh["delivered_at"].dt.normalize() >= day)).sum()
    prior = und_all / len(coh) if len(coh) else np.nan
    c = coh[coh["customer_state"] == state]
    und = (c["delivered_at"].isna() | (c["delivered_at"].dt.normalize() >= day)).sum()
    return (und + alpha * prior) / (len(c) + alpha)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backlog_vectorized_matches_naive_loop(seed):
    ev, q = _world(seed=seed, n=400)
    qs = q.sample(60, random_state=seed)
    t1 = qs[["order_approved_at", "handover_at"]].max(axis=1)
    fast = backlog_share(ev, qs["customer_state"], t1, ALPHA)
    slow = np.array(
        [_naive_backlog(ev, s, t, ALPHA) for s, t in zip(qs["customer_state"], t1, strict=True)]
    )
    np.testing.assert_allclose(fast, slow, rtol=1e-9, equal_nan=True)


def test_seller_handover_rate_excludes_self_and_future():
    ev, q = _world(seed=4, n=300)
    f = stage2_features(q, ev, ALPHA)
    i = 10
    r = q.iloc[i]
    t1 = f.loc[i, "t1"]
    win = ev[
        (ev["primary_seller_id"] == r["primary_seller_id"])
        & (ev["handover_at"] < t1)
        & (ev["handover_at"] >= t1 - pd.Timedelta(days=30))
        & (ev["order_id"] != r["order_id"])
    ]
    allw = ev[(ev["handover_at"] < t1) & (ev["handover_at"] >= t1 - pd.Timedelta(days=30))
              & (ev["order_id"] != r["order_id"])]  # fmt: skip
    prior = allw["late_handover"].mean() if len(allw) else np.nan
    expected = (win["late_handover"].sum() + ALPHA * prior) / (len(win) + ALPHA)
    # the global prior in the implementation may include the order itself when its handover was
    # earlier than t1; allow a tiny tolerance for that single order
    assert f.loc[i, "seller_late_handover_rate_30d"] == pytest.approx(expected, abs=2e-3)


# ------------------------------------------------------------------ stage-2 reason groups
def test_stage2_templates_cover_every_stage2_feature_exactly_once():
    import pytest

    from ontime.explain.reason_codes import load_templates
    from ontime.stage2.explain import ALL_COLUMNS, stage2_templates, validate_stage2_templates

    t = stage2_templates(load_templates())
    assert validate_stage2_templates(t) == []
    from ontime.explain.reason_codes import feature_to_group

    assert set(feature_to_group(t)) == set(ALL_COLUMNS)
    bad = load_templates()
    bad["stage2_groups"] = {"distance": {"features": ["days_to_handover"], "text": "x"}}
    with pytest.raises(ValueError):
        stage2_templates(bad)
