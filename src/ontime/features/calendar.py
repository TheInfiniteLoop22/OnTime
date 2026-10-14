"""Promise/handling-window and calendar features (FEATURE_SPEC sections 2-3).

Everything here uses only: approval time, purchase time, the promised (estimated) date shown at
checkout and the seller's shipping-limit date. All are fixed at or before approval.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

DAY = np.timedelta64(1, "D")


def brazil_holidays(start_year: int, end_year: int) -> np.ndarray:
    """Sorted array of Brazilian NATIONAL holiday dates (datetime64[D]) for the year span."""
    h = holidays.Brazil(years=range(start_year, end_year + 1))
    return np.array(sorted(h.keys()), dtype="datetime64[D]")


def add_calendar_features(
    df: pd.DataFrame,
    black_friday_dates: list[str],
    black_friday_window_days: int,
    holiday_dates: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return calendar/promise feature columns for `df` (same index).

    Requires columns: order_purchase_timestamp, order_approved_at, order_estimated_delivery_date,
    max_shipping_limit_date, and `distance_km` (from add_geo_features) for promise_per_1000km.
    """
    appr = df["order_approved_at"]
    out = pd.DataFrame(index=df.index)

    # --- promise & handling windows
    out["promised_days"] = (df["order_estimated_delivery_date"] - appr).dt.total_seconds() / 86400
    out["approval_lag_hours"] = (appr - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    out["handling_window_days"] = (df["max_shipping_limit_date"] - appr).dt.total_seconds() / 86400
    out["promise_per_1000km"] = out["promised_days"] / (df["distance_km"] / 1000 + 1)

    # --- calendar parts of the approval time
    out["approve_dow"] = appr.dt.dayofweek
    out["approve_hour"] = appr.dt.hour
    out["approve_month"] = appr.dt.month
    out["is_weekend_approval"] = (appr.dt.dayofweek >= 5).astype(int)

    # --- holidays (national). Searchsorted on a sorted array keeps this loop-free.
    if holiday_dates is None:
        holiday_dates = brazil_holidays(appr.dt.year.min(), appr.dt.year.max() + 1)
    a_day = appr.dt.normalize().to_numpy().astype("datetime64[D]")
    e_day = df["order_estimated_delivery_date"].dt.normalize().to_numpy().astype("datetime64[D]")
    nxt = np.searchsorted(holiday_dates, a_day, side="left")  # a holiday today counts as 0
    has_next = nxt < len(holiday_dates)
    days_to = np.full(len(df), np.nan)
    days_to[has_next] = (holiday_dates[nxt[has_next]] - a_day[has_next]) / DAY
    out["days_to_next_holiday"] = days_to
    # holidays strictly after approval day up to and including the promised day
    n_hol = np.searchsorted(holiday_dates, e_day, side="right") - np.searchsorted(
        holiday_dates, a_day, side="right"
    )
    out["holidays_in_promise_window"] = np.clip(n_hol, 0, None)

    # --- Black Friday rush window (dates and half-width come from config)
    bf = pd.to_datetime(pd.Series(black_friday_dates)).to_numpy().astype("datetime64[D]")
    gap = np.abs((a_day[:, None] - bf[None, :]) / DAY)  # days to each Black Friday
    out["is_black_friday_window"] = (gap.min(axis=1) <= black_friday_window_days).astype(int)
    return out
