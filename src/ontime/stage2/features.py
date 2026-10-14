"""Stage 2: features available at CARRIER HANDOVER (t1), days after the approval-time stage.

Decision time t1 = max(order_delivered_carrier_date, order_approved_at). At t1 we know everything
from stage 1 PLUS when/if the seller handed the parcel over. We still know NOTHING about the
customer delivery or reviews of this order. Rules (same discipline as FEATURE_SPEC section 5):

* handover features of THIS order use its own handover timestamp (observed at t1);
* seller handover history counts a past order only if its handover happened strictly before t1;
* destination BACKLOG counts, among orders handed over 8-28 days before t1, those still
  undelivered at t1. "Not delivered by t1" is observable at t1; the delivery date itself is used
  only when it falls on a day before t1's day. Events come from ALL orders (not only ones that
  were eventually delivered), so inclusion never depends on the future.

The backlog reacts to a regional shock days before delivered outcomes (the stage-1 history) can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ontime.features.history import DAY_S, past_counts, shrunk_rate

STAGE2_FEATURES = [
    "days_to_handover",
    "handover_vs_limit_days",
    "promise_left_at_handover_days",
    "seller_late_handover_rate_30d",
    "dest_backlog_share",
]
# Still forbidden at stage 2: the customer delivery date (target), review data.
STAGE2_FORBIDDEN = ["order_delivered_customer_date", "review_score", "is_late", "delivery_days"]


def make_events(orders_all: pd.DataFrame, carrier: pd.Series) -> pd.DataFrame:
    """One row per order that was handed to a carrier (any final status).

    orders_all : output of build_all_orders (all orders) with primary_seller_id, customer_state,
                 max_shipping_limit_date, order_delivered_customer_date, order_approved_at.
    carrier    : order_delivered_carrier_date indexed like orders_all["order_id"].
    """
    e = orders_all[
        [
            "order_id",
            "primary_seller_id",
            "customer_state",
            "order_approved_at",
            "max_shipping_limit_date",
            "order_delivered_customer_date",
        ]
    ].copy()
    e["handover_at"] = e["order_id"].map(carrier)
    e = e[
        e["handover_at"].notna() & e["primary_seller_id"].notna() & e["order_approved_at"].notna()
    ]
    # data errors: handed over after it was delivered. Not observable truth: drop.
    bad = e["order_delivered_customer_date"].notna() & (
        e["handover_at"] > e["order_delivered_customer_date"]
    )
    e = e[~bad]
    e["late_handover"] = (e["handover_at"] > e["max_shipping_limit_date"]).astype(int)
    return e.rename(columns={"order_delivered_customer_date": "delivered_at"}).reset_index(
        drop=True
    )


def _day_index(ts: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return ((ts.dt.normalize() - origin).dt.days).to_numpy()


def backlog_share(
    events: pd.DataFrame,
    q_state: pd.Series,
    q_time: pd.Series,
    alpha: float,
    cohort_far: int = 28,
    cohort_near: int = 8,
) -> np.ndarray:
    """Shrunk share of the destination state's cohort (handed over `cohort_near`..`cohort_far`
    days before the query day) that is still undelivered at the start of the query day.

    Day-resolution 2-D prefix sums make this exact and loop-free over queries:
      M[state][h_day, d_day] = count of events; last column = never delivered within the data.
    Delivery on the query day itself or later counts as undelivered (conservative: it is not known
    to be delivered before t1).
    """
    origin = min(events["handover_at"].min(), q_time.min()).normalize()
    h = _day_index(events["handover_at"], origin)
    nd = int(max(h.max(), _day_index(q_time, origin).max())) + 2
    d_raw = events["delivered_at"]
    d = np.where(d_raw.notna(), _day_index(d_raw.fillna(origin), origin), nd)  # nd = "never"
    d = np.clip(d, 0, nd)

    def prefix(mask: np.ndarray) -> np.ndarray:
        M = np.zeros((nd + 1, nd + 1), dtype=np.float64)
        np.add.at(M, (h[mask], d[mask]), 1.0)
        return M.cumsum(0).cumsum(1)

    def rect(P: np.ndarray, u0: np.ndarray, u1: np.ndarray, v1: np.ndarray) -> np.ndarray:
        """Sum of M[u0..u1, 0..v1] with clipping; empty ranges give 0."""
        u0c, u1c = np.clip(u0, 0, nd), np.clip(u1, -1, nd)
        v = np.clip(v1, -1, nd)
        ok = (u1c >= u0c) & (v >= 0)
        hi = np.where(ok, P[np.clip(u1c, 0, nd), np.clip(v, 0, nd)], 0.0)
        lo = np.where(ok & (u0c > 0), P[np.clip(u0c - 1, 0, nd), np.clip(v, 0, nd)], 0.0)
        return hi - lo

    T = _day_index(q_time, origin)
    u0, u1 = T - cohort_far, T - cohort_near
    never = np.full(len(T), nd)  # include the 'never delivered' column when counting cohort size
    known = T - 1  # deliveries on days <= T-1 are known at t1

    all_P = prefix(np.ones(len(events), dtype=bool))
    n_all = rect(all_P, u0, u1, never)
    deliv_all = rect(all_P, u0, u1, known)
    prior = np.where(n_all > 0, (n_all - deliv_all) / np.maximum(n_all, 1), np.nan)

    out = np.full(len(T), np.nan)
    states = q_state.to_numpy()
    ev_state = events["customer_state"].to_numpy()
    for s in pd.unique(states):
        sel = states == s
        P = prefix(ev_state == s)
        n = rect(P, u0[sel], u1[sel], never[sel])
        und = n - rect(P, u0[sel], u1[sel], known[sel])
        out[sel] = shrunk_rate(und, n, prior[sel], alpha)
    return out


def stage2_features(
    q: pd.DataFrame, events: pd.DataFrame, alpha: float = 20.0, handover_window_days: int = 30
) -> pd.DataFrame:
    """Compute STAGE2_FEATURES for the query orders `q`.

    q needs: order_id, primary_seller_id, customer_state, order_approved_at,
    order_estimated_delivery_date, max_shipping_limit_date, handover_at.
    Returns a frame indexed like q with the stage-2 features and `t1`.
    """
    t1 = q[["order_approved_at", "handover_at"]].max(axis=1)
    out = pd.DataFrame(index=q.index)
    out["t1"] = t1
    out["days_to_handover"] = (t1 - q["order_approved_at"]).dt.total_seconds() / DAY_S
    out["handover_vs_limit_days"] = (
        q["handover_at"] - q["max_shipping_limit_date"]
    ).dt.total_seconds() / DAY_S
    out["promise_left_at_handover_days"] = (
        q["order_estimated_delivery_date"] - t1
    ).dt.total_seconds() / DAY_S

    late_h = events["late_handover"].to_numpy()
    zero_e = pd.Series(0, index=events.index)
    zero_q = pd.Series(0, index=q.index)
    n_g, l_g = past_counts(
        zero_e, events["handover_at"], late_h, zero_q, t1, window_days=handover_window_days
    )
    prior = np.where(n_g > 0, l_g / np.maximum(n_g, 1), np.nan)
    n, nl = past_counts(
        events["primary_seller_id"], events["handover_at"], late_h,
        q["primary_seller_id"], t1, window_days=handover_window_days,
    )  # fmt: skip
    # remove the order's own event if it fell inside the window (handover strictly before t1)
    own_in = (q["handover_at"] < t1) & (
        q["handover_at"] >= t1 - pd.Timedelta(days=handover_window_days)
    )
    own_late = (q["handover_at"] > q["max_shipping_limit_date"]).astype(int).to_numpy()
    n = n - own_in.to_numpy().astype(float)
    nl = nl - (own_in.to_numpy() * own_late).astype(float)
    out["seller_late_handover_rate_30d"] = shrunk_rate(nl, n, prior, alpha)
    out["dest_backlog_share"] = backlog_share(events, q["customer_state"], t1, alpha)
    return out
