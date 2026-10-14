"""Build the order-level modelling table with DuckDB + explicit population funnel.

Why two steps: SQL builds one row for EVERY order (joins/aggregations are what SQL is good at);
population filters are then applied in Python one at a time so each exclusion is counted and
logged. That funnel is the evidence for "who is in the model and why".
"""

from __future__ import annotations

import logging
from pathlib import Path
from string import Template

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ontime.config import load_config, project_path  # noqa: E402
from ontime.data.load import load_all  # noqa: E402
from ontime.data.quality import upsert_section  # noqa: E402
from ontime.data.schemas import validate_order_table  # noqa: E402

log = logging.getLogger(__name__)

SQL_PATH = Path(__file__).resolve().parents[1] / "sql" / "build_order_table.sql"
FUNNEL_START, FUNNEL_END = "<!-- order-funnel:start -->", "<!-- order-funnel:end -->"


def build_all_orders(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run the SQL script; returns one row per order (all statuses, nothing filtered)."""
    geo = load_config()["geo"]
    script = Template(SQL_PATH.read_text(encoding="utf-8")).substitute(geo)
    con = duckdb.connect()
    for name, df in tables.items():
        con.register(name, df)
    con.execute(script)
    out = con.execute("SELECT * FROM orders_all").df()
    con.close()
    log.info("SQL built %d order rows", len(out))
    return out


def apply_population_filters(
    df: pd.DataFrame, censoring_cutoff: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the modelling-population filters in order; return (filtered, funnel table)."""
    steps = [
        ("all orders", lambda d: d),
        ("status == delivered", lambda d: d[d["order_status"] == "delivered"]),
        ("approved_at not null", lambda d: d[d["order_approved_at"].notna()]),
        (
            "delivered_customer_date not null",
            lambda d: d[d["order_delivered_customer_date"].notna()],
        ),
        (
            "estimated_delivery_date not null",
            lambda d: d[d["order_estimated_delivery_date"].notna()],
        ),
        (
            "delivered_customer_date >= approved_at",
            lambda d: d[d["order_delivered_customer_date"] >= d["order_approved_at"]],
        ),
        (
            "has at least one item (primary seller known)",
            lambda d: d[d["primary_seller_id"].notna()],
        ),
        (
            f"approved_at < censoring cutoff ({censoring_cutoff})",
            lambda d: d[d["order_approved_at"] < pd.Timestamp(censoring_cutoff)],
        ),
    ]
    rows, cur = [], df
    for label, fn in steps:
        before = len(cur)
        cur = fn(cur)
        rows.append({"step": label, "rows_remaining": len(cur), "rows_dropped": before - len(cur)})
        log.info("funnel | %-48s remaining=%6d dropped=%5d", label, len(cur), before - len(cur))
    return cur.reset_index(drop=True), pd.DataFrame(rows)


def censoring_weekly(df_all: pd.DataFrame) -> pd.DataFrame:
    """Per approval week: volume, delivered share, mean delivery days (among delivered).

    Right-censoring shows up as the mean delivery time collapsing in the last weeks: only the
    fast orders have been delivered by the data extraction date.
    """
    d = df_all[df_all["order_approved_at"].notna()].copy()
    d["week"] = d["order_approved_at"].dt.to_period("W").dt.start_time
    g = d.groupby("week")
    return pd.DataFrame(
        {
            "orders": g.size(),
            "delivered_share": g["order_status"].apply(lambda s: (s == "delivered").mean()),
            "mean_delivery_days": g["delivery_days"].mean(),
            "late_rate": g["is_late"].mean(),
        }
    ).reset_index()


def plot_censoring(weekly: pd.DataFrame, cutoff: str, out_path: Path) -> None:
    """Weekly volume and mean delivery days with the chosen cutoff marked."""
    w = weekly[weekly["week"] >= "2018-01-01"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].bar(w["week"], w["orders"], width=5, color="#4C78A8")
    ax[0].set_ylabel("Orders approved per week")
    ax[0].set_title("Weekly order volume and delivery time (2018) - where does the data end?")
    ax[1].plot(w["week"], w["mean_delivery_days"], marker="o", color="#F58518")
    ax[1].set_ylabel("Mean days, approval to delivery")
    ax[1].set_xlabel("Week of order approval")
    for a in ax:
        a.axvline(pd.Timestamp(cutoff), color="crimson", ls="--", label=f"cutoff {cutoff}")
    ax[0].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_funnel_report(funnel: pd.DataFrame, weekly: pd.DataFrame, cutoff: str) -> None:
    tail = weekly[weekly["week"] >= "2018-07-23"].copy()
    tail["week"] = tail["week"].dt.strftime("%Y-%m-%d")
    body = "\n".join(
        [
            FUNNEL_START,
            "## Order-table population funnel (generated by `ontime.data.build_orders`)",
            "",
            funnel.to_markdown(index=False),
            "",
            f"### Censoring cutoff: orders approved on/after **{cutoff}** are excluded",
            "Weekly view of the final weeks (all orders with an approval time, any status):",
            "",
            tail.round(3).to_markdown(index=False),
            "",
            "![Censoring diagnostic](figures/censoring_weekly.png)",
            FUNNEL_END,
            "",
        ]
    )
    path = project_path(load_config()["paths"]["reports_dir"]) / "data_quality.md"
    upsert_section(path, FUNNEL_START, FUNNEL_END, body)


def build(validate: bool = True) -> pd.DataFrame:
    """Full Phase 2 pipeline: raw CSVs -> filtered, validated orders.parquet."""
    cfg = load_config()
    cutoff = cfg["splits"]["censoring_cutoff"]
    tables = load_all()
    df_all = build_all_orders(tables)

    weekly = censoring_weekly(df_all)
    plot_censoring(
        weekly, cutoff, project_path(cfg["paths"]["figures_dir"]) / "censoring_weekly.png"
    )

    orders, funnel = apply_population_filters(df_all, cutoff)
    if validate:
        orders = validate_order_table(orders)
    write_funnel_report(funnel, weekly, cutoff)

    # Universe of ALL approved orders with a seller, regardless of final status. Phase 4 uses it
    # for seller-load features, which must not depend on whether an order was later delivered.
    universe = df_all.loc[
        df_all["order_approved_at"].notna() & df_all["primary_seller_id"].notna(),
        ["order_id", "order_approved_at", "primary_seller_id"],
    ]
    uni_out = project_path(cfg["paths"]["order_universe_parquet"])
    universe.to_parquet(uni_out, index=False)
    log.info("Wrote %s (%d rows)", uni_out, len(universe))

    out = project_path(cfg["paths"]["orders_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(out, index=False)
    log.info("Wrote %s (%d rows, %d columns)", out, len(orders), orders.shape[1])
    return orders
