"""Exploratory data analysis helpers: derived columns, rate tables and report figures.

Why this lives in `src/`: CLAUDE.md rule 6 says notebooks only import and narrate. Every function
here returns the table behind its figure so the notebook can print it and the narrative can
quote real numbers.

IMPORTANT: these derived columns are for *analysis only*. Model features are built (and
leak-tested) separately in Phase 4.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from ontime.config import load_config, project_path  # noqa: E402
from ontime.features.geo import haversine_km  # noqa: E402

log = logging.getLogger(__name__)

BLUE, ORANGE, RED, GREY = "#4C78A8", "#F58518", "#C0392B", "#7F8C8D"
DISTANCE_BINS = [0, 100, 300, 600, 1000, 1500, 2500, np.inf]
DISTANCE_LABELS = ["<100", "100-300", "300-600", "600-1000", "1000-1500", "1500-2500", ">2500"]


def set_style() -> None:
    """One consistent look for every report figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )


def load_orders() -> pd.DataFrame:
    """Read the Phase 2 order table (delivered, validated, censoring-trimmed)."""
    return pd.read_parquet(project_path(load_config()["paths"]["orders_parquet"]))


def add_eda_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add analysis columns. `slack_days` = promised - actual: positive means early."""
    d = df.copy()
    d["approve_month"] = d["order_approved_at"].dt.to_period("M").dt.start_time
    d["approve_date"] = d["order_approved_at"].dt.normalize()
    d["approve_dow"] = d["order_approved_at"].dt.dayofweek
    d["approve_hour"] = d["order_approved_at"].dt.hour
    d["promised_days"] = (
        d["order_estimated_delivery_date"] - d["order_approved_at"]
    ).dt.total_seconds() / 86400
    d["distance_km"] = haversine_km(
        d["seller_lat"], d["seller_lng"], d["customer_lat"], d["customer_lng"]
    )
    d["distance_bucket"] = pd.cut(
        d["distance_km"], DISTANCE_BINS, labels=DISTANCE_LABELS, right=False
    )
    d["route"] = d["seller_state"] + "->" + d["customer_state"]
    d["same_state"] = d["seller_state"] == d["customer_state"]
    d["slack_days"] = d["promised_days"] - d["delivery_days"]
    # whole days between delivered DATE and estimated DATE (>0 means late); matches the target
    d["days_vs_estimate"] = (
        d["order_delivered_customer_date"].dt.normalize()
        - d["order_estimated_delivery_date"].dt.normalize()
    ).dt.days
    return d


def rate_by(df: pd.DataFrame, col: str, min_n: int = 200) -> pd.DataFrame:
    """Late rate with Wilson 95% CI per group; groups smaller than `min_n` are dropped.

    Why Wilson: stays inside [0, 1] and behaves for small counts / rare events, unlike the
    naive normal interval.
    """
    g = df.groupby(col, observed=True)["is_late"].agg(["size", "sum"]).reset_index()
    g.columns = [col, "orders", "late_orders"]
    g = g[g["orders"] >= min_n].copy()
    lo, hi = proportion_confint(g["late_orders"], g["orders"], method="wilson")
    g["late_rate"], g["ci_low"], g["ci_high"] = g["late_orders"] / g["orders"], lo, hi
    return g.reset_index(drop=True)


def _save(fig: plt.Figure, name: str) -> Path:
    out = project_path(load_config()["paths"]["figures_dir"]) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _bar_with_ci(ax, tbl: pd.DataFrame, col: str, color: str = BLUE) -> None:
    y = tbl["late_rate"] * 100
    err = np.vstack([y - tbl["ci_low"] * 100, tbl["ci_high"] * 100 - y])
    ax.bar(tbl[col].astype(str), y, yerr=err, color=color, capsize=2, alpha=0.9)
    ax.set_ylabel("Late deliveries (% of orders)")


def plot_monthly_late_rate(df: pd.DataFrame, splits: dict | None = None) -> pd.DataFrame:
    """Monthly volume and late rate, with the train/val/test boundaries marked."""
    set_style()
    m = rate_by(df, "approve_month", min_n=1)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].bar(m["approve_month"], m["orders"], width=25, color=BLUE)
    ax[0].set_ylabel("Delivered orders approved")
    ax[0].set_title("Monthly volume and late-delivery rate")
    ax[1].plot(m["approve_month"], m["late_rate"] * 100, marker="o", color=RED)
    ax[1].fill_between(
        m["approve_month"], m["ci_low"] * 100, m["ci_high"] * 100, color=RED, alpha=0.15
    )
    ax[1].set_ylabel("Late deliveries (% of orders)")
    ax[1].set_xlabel("Month of order approval")
    if splits:
        for key, label in [("val_start", "validation starts"), ("test_start", "test starts")]:
            for a in ax:
                a.axvline(pd.Timestamp(splits[key]), color=GREY, ls="--")
            ax[0].text(
                pd.Timestamp(splits[key]), ax[0].get_ylim()[1] * 0.92, " " + label, color=GREY
            )
    _save(fig, "eda_monthly_late_rate.png")
    return m


def plot_state_rates(df: pd.DataFrame, col: str, fname: str, who: str) -> pd.DataFrame:
    """Late rate by state, sorted from worst to best."""
    set_style()
    t = rate_by(df, col, min_n=200).sort_values("late_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    _bar_with_ci(ax, t, col)
    ax.axhline(df["is_late"].mean() * 100, color=GREY, ls="--", label="overall rate")
    ax.set_xlabel(f"{who} state")
    ax.set_title(f"Late-delivery rate by {who.lower()} state (error bars: 95% CI)")
    ax.legend()
    _save(fig, fname)
    return t


def plot_route_heatmap(df: pd.DataFrame, top: int = 8, min_n: int = 100) -> pd.DataFrame:
    """Seller-state x customer-state late rate for the busiest states (cells with n<min_n blank)."""
    set_style()
    s_top = df["seller_state"].value_counts().head(top).index
    c_top = df["customer_state"].value_counts().head(top).index
    sub = df[df["seller_state"].isin(s_top) & df["customer_state"].isin(c_top)]
    piv = sub.pivot_table(
        index="seller_state", columns="customer_state", values="is_late", aggfunc="mean"
    )
    cnt = sub.pivot_table(
        index="seller_state", columns="customer_state", values="is_late", aggfunc="size"
    )
    piv = piv.where(cnt >= min_n) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Customer state")
    ax.set_ylabel("Seller state")
    ax.set_title(f"Late rate (%) by route - busiest states, cells with >= {min_n} orders")
    fig.colorbar(im, label="Late deliveries (%)")
    _save(fig, "eda_route_heatmap.png")
    return piv.round(1)


def plot_category_rates(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Late rate for the `top` most common product categories."""
    set_style()
    keep = df["top_item_category"].fillna("unknown").value_counts().head(top).index
    d = df.assign(category=df["top_item_category"].fillna("unknown"))
    t = rate_by(d[d["category"].isin(keep)], "category", min_n=1).sort_values("late_rate")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(t["category"], t["late_rate"] * 100, color=BLUE)
    ax.axvline(df["is_late"].mean() * 100, color=GREY, ls="--", label="overall rate")
    ax.set_xlabel("Late deliveries (% of orders)")
    ax.set_title(f"Late rate by product category (top {top} by volume)")
    ax.legend()
    _save(fig, "eda_category_rates.png")
    return t


def plot_distance_rates(df: pd.DataFrame) -> pd.DataFrame:
    set_style()
    t = rate_by(df, "distance_bucket", min_n=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    _bar_with_ci(ax, t, "distance_bucket")
    ax.set_xlabel("Seller-to-customer straight-line distance (km)")
    ax.set_title("Late-delivery rate by distance")
    _save(fig, "eda_distance_rates.png")
    return t


def plot_time_of_week(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Late rate by approval weekday and hour (does *when* you order matter?)."""
    set_style()
    dow = rate_by(df, "approve_dow", min_n=1)
    dow["approve_dow"] = dow["approve_dow"].map(
        dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))
    )
    hour = rate_by(df, "approve_hour", min_n=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    _bar_with_ci(ax[0], dow, "approve_dow")
    ax[0].set_xlabel("Weekday of approval")
    ax[0].set_title("By weekday")
    ax[1].plot(hour["approve_hour"], hour["late_rate"] * 100, marker="o", color=ORANGE)
    ax[1].set_xlabel("Hour of approval")
    ax[1].set_title("By hour")
    _save(fig, "eda_time_of_week.png")
    return dow, hour


def plot_promise_vs_actual(df: pd.DataFrame) -> pd.DataFrame:
    """How generous is Olist's estimate? Compare promised and actual days."""
    set_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(0, 70, 2)
    ax[0].hist(df["promised_days"].clip(upper=69), bins, alpha=0.6, label="promised", color=BLUE)
    ax[0].hist(df["delivery_days"].clip(upper=69), bins, alpha=0.6, label="actual", color=ORANGE)
    ax[0].set_xlabel("Days from approval (capped at 69)")
    ax[0].set_ylabel("Orders")
    ax[0].set_title("Promised vs actual delivery time")
    ax[0].legend()
    ax[1].hist(df["slack_days"].clip(-20, 40), np.arange(-20, 41, 2), color=BLUE)
    ax[1].axvline(0, color=RED, ls="--", label="late if left of line")
    ax[1].set_xlabel("Slack = promised days - actual days (clipped)")
    ax[1].set_title("Most orders arrive well before the promise")
    ax[1].legend()
    _save(fig, "eda_promise_vs_actual.png")
    s = df[["promised_days", "delivery_days", "slack_days"]].describe(percentiles=[0.1, 0.5, 0.9])
    return s.round(1)


def plot_black_friday(df: pd.DataFrame, year_month: str = "2017-11") -> pd.DataFrame:
    """Daily volume and late rate in Black Friday month with the configured window shaded."""
    set_style()
    cfg = load_config()["features"]
    bf = pd.Timestamp(next(d for d in cfg["black_friday_dates"] if d.startswith(year_month[:4])))
    w = cfg["black_friday_window_days"]
    m = df[df["approve_date"].dt.strftime("%Y-%m") == year_month]
    daily = m.groupby("approve_date").agg(orders=("is_late", "size"), late_rate=("is_late", "mean"))
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].bar(daily.index, daily["orders"], color=BLUE)
    ax[0].set_ylabel("Orders approved")
    ax[0].set_title(f"Black Friday {bf.date()}: daily volume and late rate of those orders")
    ax[1].plot(daily.index, daily["late_rate"] * 100, marker="o", color=RED)
    ax[1].set_ylabel("Late deliveries (% of orders)")
    for a in ax:
        a.axvspan(bf - pd.Timedelta(days=w), bf + pd.Timedelta(days=w), color=ORANGE, alpha=0.2)
    _save(fig, "eda_black_friday.png")
    inside = df["approve_date"].between(bf - pd.Timedelta(days=w), bf + pd.Timedelta(days=w))
    daily.attrs["window_rate"] = float(df.loc[inside, "is_late"].mean())
    daily.attrs["window_orders"] = int(inside.sum())
    daily.attrs["month_rate_outside"] = float(
        m.loc[
            ~m["approve_date"].between(bf - pd.Timedelta(days=w), bf + pd.Timedelta(days=w)),
            "is_late",
        ].mean()
    )
    return daily
