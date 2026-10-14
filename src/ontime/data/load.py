"""Load the raw Olist CSVs into pandas with explicit dtypes.

Why: zip prefixes must stay strings (leading zeros matter and they are identifiers, not numbers),
and timestamps must be parsed once, here, so downstream code never re-parses.
"""

from __future__ import annotations

import logging

import pandas as pd

from ontime.config import load_config, project_path

log = logging.getLogger(__name__)

TIMESTAMP_COLUMNS: dict[str, list[str]] = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}

STRING_COLUMNS: dict[str, list[str]] = {
    "customers": ["customer_zip_code_prefix"],
    "sellers": ["seller_zip_code_prefix"],
    "geolocation": ["geolocation_zip_code_prefix"],
}


def load_table(name: str) -> pd.DataFrame:
    """Load one raw table by logical name (key of `raw_files` in config)."""
    cfg = load_config()
    path = project_path(cfg["paths"]["raw_dir"]) / cfg["raw_files"][name]
    dtype = {c: "string" for c in STRING_COLUMNS.get(name, [])}
    df = pd.read_csv(
        path,
        dtype=dtype or None,
        parse_dates=TIMESTAMP_COLUMNS.get(name),
    )
    # Source typo: product_name_lenght / product_description_lenght. Keep source names raw here;
    # features that use them (if any) reference the typo explicitly.
    log.info("Loaded %-22s rows=%7d cols=%d", name, len(df), df.shape[1])
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Load all nine raw tables."""
    return {name: load_table(name) for name in load_config()["raw_files"]}
