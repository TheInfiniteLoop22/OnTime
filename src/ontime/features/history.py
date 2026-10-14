"""Leak-safe historical features (FEATURE_SPEC section 5) - the most error-prone family.

THE RULE. For an order `o` approved at t0(o), a past order `p` may contribute to a history
feature only if its OUTCOME was observable before t0(o):

    delivered_customer_date(p) < t0(o)          (strict)

"p was approved before t0" is NOT enough: p may still be in transit, and counting its eventual
lateness would use information from the future.

Consequences that fall out of the rule:
* An order never counts itself: delivered(o) >= approved(o) = t0(o), so it is never < t0(o).
* Row order is irrelevant (everything is computed from sorted keys).

IMPLEMENTATION. No per-row Python loops. Each (key, time) pair is encoded as one int64
`code * 2**34 + epoch_seconds`; events are sorted once, then `np.searchsorted` gives, for every
query at once, how many same-key events happened strictly before t0 (and inside an optional
window), and `cumsum` gives how many of them were late.

Known simplification: an order becomes "known late" once its promised date passes undelivered,
which is earlier than its delivery. We ignore that earlier signal and wait for delivery: simpler,
strictly conservative, and impossible to leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SHIFT = np.int64(1) << 34  # > any epoch-seconds value (~1.7e9), keeps key and time separable
DAY_S = 86_400


def _epoch_seconds(ts: pd.Series) -> np.ndarray:
    return ts.to_numpy().astype("datetime64[s]").astype(np.int64)


def past_counts(
    event_key: pd.Series,
    event_time: pd.Series,
    event_late: np.ndarray | pd.Series,
    query_key: pd.Series,
    query_time: pd.Series,
    window_days: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query: (n_events, n_late_events) with the same key and event_time in
    [query_time - window, query_time)  -- strictly before the query time.

    `window_days=None` means "any time before the query".
    """
    codes, _ = pd.factorize(pd.concat([event_key, query_key], ignore_index=True))
    ev_code, q_code = (
        codes[: len(event_key)].astype(np.int64),
        codes[len(event_key) :].astype(np.int64),
    )
    ev_comb = ev_code * _SHIFT + _epoch_seconds(event_time)
    order = np.argsort(ev_comb, kind="stable")
    ev_sorted = ev_comb[order]
    late_sorted = np.asarray(event_late, dtype=np.int64)[order]
    cum_late = np.concatenate([[0], np.cumsum(late_sorted)])

    q_t = _epoch_seconds(query_time)
    hi = np.searchsorted(ev_sorted, q_code * _SHIFT + q_t, side="left")  # events < t0
    lower = q_t - int(window_days * DAY_S) if window_days is not None else np.zeros_like(q_t)
    lo = np.searchsorted(ev_sorted, q_code * _SHIFT + lower, side="left")
    return (hi - lo).astype(float), (cum_late[hi] - cum_late[lo]).astype(float)


def shrunk_rate(n_late: np.ndarray, n: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    """Beta-binomial (Bayesian) shrinkage: (late + alpha*prior) / (n + alpha).

    With n = 0 this returns exactly `prior`; with large n it converges to the raw rate. The
    pseudo-count `alpha` says "trust the seller's own record as much as `alpha` orders of average
    behaviour". If the prior itself is unknown (no history at all) the result is NaN.
    """
    return (n_late + alpha * prior) / (n + alpha)


def build_history_features(
    orders: pd.DataFrame,
    universe: pd.DataFrame,
    alpha: float,
    recent_window_days: int = 90,
    load_window_days: int = 7,
    global_window_days: int = 30,
) -> pd.DataFrame:
    """Compute all history features for `orders` (same index).

    orders   : modelling population. Needs order_id, primary_seller_id, seller_state,
               customer_state, order_approved_at, order_delivered_customer_date, is_late.
    universe : ALL approved orders with a seller (order_id, order_approved_at,
               primary_seller_id), regardless of final status. Used only for features that need
               no outcome (load, seller tenure).
    """
    t0 = orders["order_approved_at"]
    deliv = orders["order_delivered_customer_date"]
    late = orders["is_late"].to_numpy()
    out = pd.DataFrame(index=orders.index)

    # Global late rate known before t0: the prior m that every shrunk rate is pulled toward.
    zero = pd.Series(0, index=orders.index)
    n_g, l_g = past_counts(zero, deliv, late, zero, t0)
    with np.errstate(invalid="ignore", divide="ignore"):
        prior = np.where(n_g > 0, l_g / np.where(n_g > 0, n_g, 1), np.nan)

    # Seller track record (all-time and recent window)
    seller = orders["primary_seller_id"]
    n, nl = past_counts(seller, deliv, late, seller, t0)
    out["seller_n_past"] = n
    out["seller_late_rate_shrunk"] = shrunk_rate(nl, n, prior, alpha)
    n90, nl90 = past_counts(seller, deliv, late, seller, t0, window_days=recent_window_days)
    out["seller_late_rate_90d_shrunk"] = shrunk_rate(nl90, n90, prior, alpha)

    # Route (seller_state -> customer_state) and destination-state track records
    route = orders["seller_state"] + "|" + orders["customer_state"]
    nr, nlr = past_counts(route, deliv, late, route, t0)
    out["route_late_rate_shrunk"] = shrunk_rate(nlr, nr, prior, alpha)
    cs = orders["customer_state"]
    nc, nlc = past_counts(cs, deliv, late, cs, t0)
    out["customer_state_late_rate_shrunk"] = shrunk_rate(nlc, nc, prior, alpha)

    # System-wide stress: raw late rate of everything delivered in the last N days
    n30, l30 = past_counts(zero, deliv, late, zero, t0, window_days=global_window_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["global_late_rate_30d"] = np.where(n30 > 0, l30 / np.where(n30 > 0, n30, 1), np.nan)

    # Load: seller's orders APPROVED in the last 7 days. Needs no outcome, so it uses the full
    # universe (delivered or not) - otherwise "was it later delivered?" would leak in.
    nload, _ = past_counts(
        universe["primary_seller_id"],
        universe["order_approved_at"],
        np.zeros(len(universe)),
        seller,
        t0,
        window_days=load_window_days,
    )
    out["seller_load_7d"] = nload

    # Tenure: time since the seller's first approved order in the data (always <= t0).
    first = universe.groupby("primary_seller_id")["order_approved_at"].min()
    out["seller_days_since_first_order"] = (t0 - seller.map(first)).dt.total_seconds() / DAY_S
    return out
