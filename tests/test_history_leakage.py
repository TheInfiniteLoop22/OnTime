"""Leakage tests for history features (FEATURE_SPEC section 5, the five required tests + extras).

Rule under test: a past order contributes only if delivered_customer_date(p) < t0(o).
"""

import numpy as np
import pandas as pd
import pytest

from ontime.features.history import build_history_features, past_counts, shrunk_rate

T = pd.Timestamp
ALPHA = 20.0


def _orders(rows: list[dict]) -> pd.DataFrame:
    """rows: order_id, seller, appr, deliv, late, [seller_state, customer_state]."""
    df = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "order_id": df["order_id"],
            "primary_seller_id": df["seller"],
            "seller_state": df.get("seller_state", "SP"),
            "customer_state": df.get("customer_state", "RJ"),
            "order_approved_at": pd.to_datetime(df["appr"], format="mixed"),
            "order_delivered_customer_date": pd.to_datetime(df["deliv"], format="mixed"),
            "is_late": df["late"].astype(int),
        }
    )


def _universe(orders: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    u = orders[["order_id", "order_approved_at", "primary_seller_id"]]
    return pd.concat([u, extra], ignore_index=True) if extra is not None else u


def _feat(orders: pd.DataFrame, universe: pd.DataFrame | None = None) -> pd.DataFrame:
    f = build_history_features(
        orders, _universe(orders) if universe is None else universe, alpha=ALPHA
    )
    return f.set_index(orders["order_id"].to_numpy())


# --------------------------------------------------------------- required test 1
def test_past_order_approved_before_but_delivered_after_t0_is_not_counted():
    orders = _orders(
        [
            # p: approved Jan 1, delivered Jan 20 (late). Still in transit at o's t0 (Jan 10).
            dict(order_id="p", seller="S", appr="2018-01-01", deliv="2018-01-20", late=1),
            # q: another seller, fully known by Jan 5 -> defines the global prior
            dict(order_id="q", seller="Z", appr="2018-01-01", deliv="2018-01-05", late=0),
            dict(order_id="o", seller="S", appr="2018-01-10", deliv="2018-01-15", late=0),
            dict(order_id="o2", seller="S", appr="2018-01-25", deliv="2018-01-30", late=0),
        ]
    )
    f = _feat(orders)
    assert f.loc["o", "seller_n_past"] == 0  # p not yet delivered at t0(o)
    assert f.loc["o2", "seller_n_past"] == 2  # by Jan 25 both p (Jan 20) and o (Jan 15) are known
    # o2's seller rate: 1 late (p) of 2 known (p, o) -> (1 + 20*m)/(2 + 20).
    # global events known before Jan 25: q (on time), p (late), o (on time) -> m = 1/3
    assert f.loc["o2", "seller_late_rate_shrunk"] == pytest.approx((1 + 20 / 3) / 22)


# --------------------------------------------------------------- required test 2
def test_order_never_counts_itself():
    orders = _orders(
        [dict(order_id="a", seller="S", appr="2018-01-01", deliv="2018-01-10", late=1)]
    )
    f = _feat(orders)
    assert f.loc["a", "seller_n_past"] == 0
    assert np.isnan(f.loc["a", "seller_late_rate_shrunk"])  # no history at all -> prior unknown
    assert f.loc["a", "seller_load_7d"] == 0  # own approval is not "before" t0


def test_order_delivered_at_same_second_as_approved_does_not_count_itself():
    orders = _orders(
        [
            dict(
                order_id="a",
                seller="S",
                appr="2018-01-01 10:00:00",
                deliv="2018-01-01 10:00:00",
                late=1,
            ),
            dict(order_id="b", seller="S", appr="2018-01-01 10:00:00", deliv="2018-01-03", late=0),
        ]
    )
    f = _feat(orders)
    assert f["seller_n_past"].tolist() == [0, 0]  # strict '<' even for identical timestamps


# --------------------------------------------------------------- required test 3
def test_shuffling_rows_does_not_change_features():
    orders = _synthetic(seed=3)
    base = _feat(orders)
    shuffled = orders.sample(frac=1.0, random_state=7)
    other = _feat(shuffled.reset_index(drop=True))
    pd.testing.assert_frame_equal(
        base.sort_index(), other.sort_index(), check_exact=False, rtol=1e-12
    )


# --------------------------------------------------------------- required test 4
def test_seller_with_zero_history_gets_exactly_the_prior():
    orders = _orders(
        [
            dict(order_id="a", seller="A", appr="2018-01-01", deliv="2018-01-03", late=1),
            dict(order_id="b", seller="B", appr="2018-01-01", deliv="2018-01-04", late=0),
            dict(order_id="c", seller="C", appr="2018-01-01", deliv="2018-01-05", late=0),
            dict(order_id="d", seller="D", appr="2018-01-02", deliv="2018-01-06", late=0),
            # new seller NEW appears after 3 outcomes are known: 1 late of 3 -> m = 1/3
            dict(order_id="n", seller="NEW", appr="2018-01-06", deliv="2018-01-09", late=0),
        ]
    )
    f = _feat(orders)
    assert f.loc["n", "seller_n_past"] == 0
    assert f.loc["n", "seller_late_rate_shrunk"] == pytest.approx(1 / 3)  # exactly m
    assert f.loc["n", "seller_late_rate_90d_shrunk"] == pytest.approx(1 / 3)


# --------------------------------------------------------------- required test 5 (property)
def _synthetic(seed: int, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2018-01-01")
    appr = base + pd.to_timedelta(rng.integers(0, 200 * 24 * 3600, n), unit="s")
    deliv = appr + pd.to_timedelta(rng.integers(0, 40 * 24 * 3600, n), unit="s")
    return pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(n)],
            "primary_seller_id": rng.choice([f"s{i}" for i in range(15)], n),
            "seller_state": rng.choice(["SP", "RJ", "MG"], n),
            "customer_state": rng.choice(["SP", "BA", "RS", "PE"], n),
            "order_approved_at": appr,
            "order_delivered_customer_date": deliv,
            "is_late": rng.binomial(1, 0.15, n),
        }
    )


def _naive(orders: pd.DataFrame, universe: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Obviously-correct O(n^2) reference implementation of every history feature."""
    rows = []
    first = universe.groupby("primary_seller_id")["order_approved_at"].min()
    for _, o in orders.iterrows():
        t0 = o["order_approved_at"]
        known = orders[orders["order_delivered_customer_date"] < t0]  # outcome observable
        assert o["order_id"] not in set(known["order_id"])  # sanity: never itself
        m = known["is_late"].mean() if len(known) else np.nan

        def shrunk(sub: pd.DataFrame, m: float = m) -> float:
            return (sub["is_late"].sum() + alpha * m) / (len(sub) + alpha)

        s = known[known["primary_seller_id"] == o["primary_seller_id"]]
        s90 = s[s["order_delivered_customer_date"] >= t0 - pd.Timedelta(days=90)]
        r = known[
            (known["seller_state"] == o["seller_state"])
            & (known["customer_state"] == o["customer_state"])
        ]
        c = known[known["customer_state"] == o["customer_state"]]
        g30 = known[known["order_delivered_customer_date"] >= t0 - pd.Timedelta(days=30)]
        uni = universe[
            (universe["primary_seller_id"] == o["primary_seller_id"])
            & (universe["order_approved_at"] < t0)
            & (universe["order_approved_at"] >= t0 - pd.Timedelta(days=7))
        ]
        rows.append(
            {
                "seller_n_past": len(s),
                "seller_late_rate_shrunk": shrunk(s),
                "seller_late_rate_90d_shrunk": shrunk(s90),
                "route_late_rate_shrunk": shrunk(r),
                "customer_state_late_rate_shrunk": shrunk(c),
                "global_late_rate_30d": g30["is_late"].mean() if len(g30) else np.nan,
                "seller_load_7d": len(uni),
                "seller_days_since_first_order": (
                    t0 - first[o["primary_seller_id"]]
                ).total_seconds()
                / 86400,
            }
        )
    return pd.DataFrame(rows, index=orders["order_id"].to_numpy())


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_vectorized_matches_naive_loop(seed):
    orders = _synthetic(seed)
    rng = np.random.default_rng(100 + seed)
    # universe has extra approved-but-never-delivered orders (they matter for load only)
    extra = pd.DataFrame(
        {
            "order_id": [f"x{i}" for i in range(120)],
            "order_approved_at": pd.Timestamp("2018-01-01")
            + pd.to_timedelta(rng.integers(0, 200 * 24 * 3600, 120), unit="s"),
            "primary_seller_id": rng.choice([f"s{i}" for i in range(15)], 120),
        }
    )
    universe = _universe(orders, extra)
    fast = _feat(orders, universe)
    slow = _naive(orders, universe, ALPHA)
    pd.testing.assert_frame_equal(
        fast[slow.columns], slow, check_exact=False, rtol=1e-9, atol=1e-12, check_dtype=False
    )


# --------------------------------------------------------------- extra guards
def test_window_is_inclusive_at_lower_edge_and_exclusive_at_t0():
    t0 = T("2018-06-01 12:00:00")
    ev_time = pd.Series([t0 - pd.Timedelta(days=90), t0 - pd.Timedelta(days=90, seconds=1), t0])
    key = pd.Series(["s"] * 3)
    n, _ = past_counts(key, ev_time, np.zeros(3), pd.Series(["s"]), pd.Series([t0]), window_days=90)
    # exactly 90d before counts; 90d+1s before does not; an event AT t0 does not (strict <)
    assert n.tolist() == [1.0]


def test_load_feature_does_not_depend_on_outcomes():
    orders = _synthetic(seed=11, n=100)
    base = _feat(orders)
    flipped = orders.assign(is_late=1 - orders["is_late"])
    flipped["order_delivered_customer_date"] += pd.Timedelta(days=3)
    other = _feat(flipped, _universe(orders))  # same universe (approvals unchanged)
    pd.testing.assert_series_equal(base["seller_load_7d"], other["seller_load_7d"])


def test_shrinkage_limits():
    prior = np.array([0.1])
    assert shrunk_rate(np.array([0.0]), np.array([0.0]), prior, 20)[0] == pytest.approx(0.1)
    big = shrunk_rate(np.array([500.0]), np.array([1000.0]), prior, 20)[0]
    assert abs(big - 0.5) < 0.01  # lots of data -> close to the raw rate
