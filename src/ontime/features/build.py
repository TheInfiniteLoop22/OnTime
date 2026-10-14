"""Assemble data/processed/features.parquet and the feature summary report.

The feature list below is the single source of truth for "what the model may see". Tests assert
it contains no forbidden column and that the built table matches it exactly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ontime.config import load_config, project_path
from ontime.features.calendar import add_calendar_features
from ontime.features.geo import add_geo_features
from ontime.features.history import build_history_features
from ontime.features.product import add_product_features
from ontime.models.splits import assign_split

log = logging.getLogger(__name__)

# ----- feature families (names follow docs/FEATURE_SPEC.md) -----------------------------------
GEO_FEATURES = [
    "distance_km",
    "log_distance_km",
    "same_state",
    "customer_region",
    "seller_region",
    "geo_imputed",
    "n_distinct_sellers",
]
CALENDAR_FEATURES = [
    "promised_days",
    "approval_lag_hours",
    "handling_window_days",
    "promise_per_1000km",
    "approve_dow",
    "approve_hour",
    "approve_month",
    "is_weekend_approval",
    "days_to_next_holiday",
    "holidays_in_promise_window",
    "is_black_friday_window",
]
PRODUCT_FEATURES = [
    "n_items",
    "total_price",
    "total_freight",
    "freight_ratio",
    "total_weight_kg",
    "total_volume_l",
    "max_item_weight_kg",
    "category_group",
    "product_photos_qty_mean",
    "payment_type",
    "payment_installments_max",
]
HISTORY_FEATURES = [
    "seller_n_past",
    "seller_late_rate_shrunk",
    "seller_late_rate_90d_shrunk",
    "route_late_rate_shrunk",
    "customer_state_late_rate_shrunk",
    "seller_load_7d",
    "seller_days_since_first_order",
    "global_late_rate_30d",
]
FEATURE_COLUMNS = GEO_FEATURES + CALENDAR_FEATURES + PRODUCT_FEATURES + HISTORY_FEATURES
CATEGORICAL_FEATURES = ["customer_region", "seller_region", "category_group", "payment_type"]

# Non-feature columns carried for splitting, slicing and analysis. NEVER fed to a model.
META_COLUMNS = [
    "order_id",
    "approved_at",
    "split",
    "is_late",
    "delivery_days",
    "primary_seller_id",
    "seller_state",
    "customer_state",
]

# FEATURE_SPEC "Forbidden columns": recorded after approval, or the target itself.
FORBIDDEN_FEATURES = [
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_status",
    "order_estimated_delivery_date",  # raw date: only the derived promised_days is allowed
    "is_late",
    "delivery_days",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
    "review_comment_title",
    "review_comment_message",
]


def assert_feature_list_is_safe(features: list[str] = FEATURE_COLUMNS) -> None:
    """Raise if a forbidden or review-derived name sneaks into the feature list."""
    bad = [f for f in features if f in FORBIDDEN_FEATURES or f.startswith("review_")]
    if bad:
        raise ValueError(f"Forbidden columns in feature list: {bad}")
    if len(set(features)) != len(features):
        raise ValueError("Duplicate names in feature list")


def build_feature_table(
    orders: pd.DataFrame, universe: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    """Orders + universe -> one row per order: meta columns + features (pure, testable)."""
    cfg = cfg or load_config()
    fc = cfg["features"]
    assert_feature_list_is_safe()
    orders = orders.reset_index(drop=True)

    geo = add_geo_features(orders)
    cal = add_calendar_features(
        orders.assign(distance_km=geo["distance_km"]),
        fc["black_friday_dates"],
        fc["black_friday_window_days"],
    )
    prod = add_product_features(orders, fc["category_groups"])
    hist = build_history_features(
        orders,
        universe,
        alpha=fc["shrinkage_alpha"],
        recent_window_days=fc["recent_window_days"],
        load_window_days=fc["load_window_days"],
        global_window_days=fc["global_window_days"],
    )

    meta = pd.DataFrame(
        {
            "order_id": orders["order_id"],
            "approved_at": orders["order_approved_at"],
            "split": assign_split(orders["order_approved_at"], cfg),
            "is_late": orders["is_late"].astype(int),
            "delivery_days": orders["delivery_days"],
            "primary_seller_id": orders["primary_seller_id"],
            "seller_state": orders["seller_state"],
            "customer_state": orders["customer_state"],
        }
    )
    table = pd.concat([meta, geo, cal, prod, hist], axis=1)[META_COLUMNS + FEATURE_COLUMNS]
    for c in CATEGORICAL_FEATURES:
        table[c] = table[c].astype("category")
    return table


# ----- feature summary report ---------------------------------------------------------------
def feature_summary(table: pd.DataFrame, red_flag_auc: float = 0.75) -> pd.DataFrame:
    """Per-feature dtype, null %, per-split mean and univariate train AUC (leak red flag).

    A single feature that separates late from on-time almost perfectly on its own is the classic
    signature of leakage, so |AUC - 0.5| is reported and flagged above `red_flag_auc`.
    """
    train = table[table["split"] == "train"]
    rows = []
    for f in FEATURE_COLUMNS:
        col = table[f]
        row = {"feature": f, "dtype": str(col.dtype), "null_pct": col.isna().mean() * 100}
        if f in CATEGORICAL_FEATURES:
            for s in ["train", "validation", "test"]:
                vc = table.loc[table["split"] == s, f].value_counts(normalize=True)
                row[f"{s}"] = f"{vc.index[0]} ({vc.iloc[0]:.0%})"
            row["univariate_auc"] = np.nan
        else:
            for s in ["train", "validation", "test"]:
                row[f"{s}"] = table.loc[table["split"] == s, f].astype(float).mean()
            x = train[f].astype(float)
            x = x.fillna(x.median())
            row["univariate_auc"] = (
                roc_auc_score(train["is_late"], x) if x.nunique() > 1 else np.nan
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["red_flag"] = (out["univariate_auc"] - 0.5).abs() + 0.5 > red_flag_auc
    return out


def write_feature_summary(summary: pd.DataFrame, table: pd.DataFrame) -> None:
    path = project_path(load_config()["paths"]["reports_dir"]) / "feature_summary.md"
    s = summary.copy()
    for c in ["null_pct", "univariate_auc"]:
        s[c] = s[c].round(3)
    for c in ["train", "validation", "test"]:
        s[c] = s[c].apply(lambda v: f"{v:,.3f}" if isinstance(v, float) else v)
    counts = table.groupby("split")["is_late"].agg(["size", "sum", "mean"])
    counts.columns = ["orders", "late_orders", "late_rate"]
    text = "\n".join(
        [
            "# Feature summary (generated by `ontime.features.build`)",
            "",
            f"{len(FEATURE_COLUMNS)} features. Means for numeric features; most common value "
            "(share) for categorical ones. `univariate_auc` is each numeric feature's ROC-AUC "
            "against `is_late` on TRAIN alone - values far from 0.5 (flag at > 0.75) are leakage "
            "red flags.",
            "",
            "## Rows per split",
            counts.round(4).to_markdown(),
            "",
            "## Features",
            s.to_markdown(index=False),
            "",
            f"Red-flag features: {summary.loc[summary['red_flag'], 'feature'].tolist() or 'none'}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    log.info("Wrote %s", path)


def build() -> pd.DataFrame:
    """Phase 4 stage: read interim tables, build features, validate, write parquet + report."""
    cfg = load_config()
    orders = pd.read_parquet(project_path(cfg["paths"]["orders_parquet"]))
    universe = pd.read_parquet(project_path(cfg["paths"]["order_universe_parquet"]))
    log.info("features | orders in=%d universe=%d", len(orders), len(universe))

    table = build_feature_table(orders, universe, cfg)
    if table["order_id"].duplicated().any():
        raise ValueError("duplicate order_id in feature table")
    if len(table) != len(orders):
        raise ValueError("row count changed while building features")

    out = project_path(cfg["paths"]["features_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    log.info("features | wrote %s rows=%d features=%d", out, len(table), len(FEATURE_COLUMNS))
    write_feature_summary(feature_summary(table), table)

    from ontime.features.leakage_demo import write_leakage_demo

    write_leakage_demo(orders, table)
    return table
