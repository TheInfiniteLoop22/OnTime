"""Pandera schemas for the nine raw tables.

Why: fail loudly if the source data changes shape or violates basic expectations. Schemas encode
types, key uniqueness and plausible ranges. `lazy=True` reports all violations at once.
Nulls are allowed only where the data genuinely has them (e.g., undelivered orders).
"""

from __future__ import annotations

import pandera.pandas as pa

ORDER_STATUSES = [
    "delivered",
    "shipped",
    "canceled",
    "unavailable",
    "invoiced",
    "processing",
    "created",
    "approved",
]
PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card", "not_defined"]

_ts = pa.Column("datetime64[ns]", nullable=True, coerce=True)

SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "orders": pa.DataFrameSchema(
        {
            "order_id": pa.Column(str, unique=True),
            "customer_id": pa.Column(str),
            "order_status": pa.Column(str, pa.Check.isin(ORDER_STATUSES)),
            "order_purchase_timestamp": pa.Column("datetime64[ns]", coerce=True),
            "order_approved_at": _ts,
            "order_delivered_carrier_date": _ts,
            "order_delivered_customer_date": _ts,
            "order_estimated_delivery_date": pa.Column("datetime64[ns]", coerce=True),
        },
        strict=False,
    ),
    "order_items": pa.DataFrameSchema(
        {
            "order_id": pa.Column(str),
            "order_item_id": pa.Column(int, pa.Check.ge(1)),
            "product_id": pa.Column(str),
            "seller_id": pa.Column(str),
            "shipping_limit_date": _ts,
            "price": pa.Column(float, pa.Check.ge(0)),
            "freight_value": pa.Column(float, pa.Check.ge(0)),
        },
        unique=["order_id", "order_item_id"],
        strict=False,
    ),
    "products": pa.DataFrameSchema(
        {
            "product_id": pa.Column(str, unique=True),
            "product_category_name": pa.Column(str, nullable=True),
            "product_photos_qty": pa.Column(float, pa.Check.ge(0), nullable=True),
            "product_weight_g": pa.Column(float, pa.Check.ge(0), nullable=True),
            "product_length_cm": pa.Column(float, pa.Check.ge(0), nullable=True),
            "product_height_cm": pa.Column(float, pa.Check.ge(0), nullable=True),
            "product_width_cm": pa.Column(float, pa.Check.ge(0), nullable=True),
        },
        strict=False,
    ),
    "sellers": pa.DataFrameSchema(
        {
            "seller_id": pa.Column(str, unique=True),
            "seller_zip_code_prefix": pa.Column(str),
            "seller_state": pa.Column(str, pa.Check.str_length(2, 2)),
        },
        strict=False,
    ),
    "customers": pa.DataFrameSchema(
        {
            "customer_id": pa.Column(str, unique=True),
            "customer_unique_id": pa.Column(str),
            "customer_zip_code_prefix": pa.Column(str),
            "customer_state": pa.Column(str, pa.Check.str_length(2, 2)),
        },
        strict=False,
    ),
    "geolocation": pa.DataFrameSchema(
        {
            "geolocation_zip_code_prefix": pa.Column(str),
            "geolocation_lat": pa.Column(float, pa.Check.in_range(-90, 90)),
            "geolocation_lng": pa.Column(float, pa.Check.in_range(-180, 180)),
            "geolocation_state": pa.Column(str, pa.Check.str_length(2, 2)),
        },
        strict=False,
    ),
    "payments": pa.DataFrameSchema(
        {
            "order_id": pa.Column(str),
            "payment_sequential": pa.Column(int, pa.Check.ge(1)),
            "payment_type": pa.Column(str, pa.Check.isin(PAYMENT_TYPES)),
            "payment_installments": pa.Column(int, pa.Check.ge(0)),
            "payment_value": pa.Column(float, pa.Check.ge(0)),
        },
        unique=["order_id", "payment_sequential"],
        strict=False,
    ),
    "reviews": pa.DataFrameSchema(
        {
            "review_id": pa.Column(str),
            "order_id": pa.Column(str),
            "review_score": pa.Column(int, pa.Check.isin([1, 2, 3, 4, 5])),
            "review_creation_date": _ts,
        },
        strict=False,
    ),
    "category_translation": pa.DataFrameSchema(
        {
            "product_category_name": pa.Column(str, unique=True),
            "product_category_name_english": pa.Column(str),
        },
        strict=False,
    ),
}


def order_table_schema() -> pa.DataFrameSchema:
    """Contract for data/interim/orders.parquet: one row per order, no nulls in modelling core."""
    from ontime.config import load_config

    g = load_config()["geo"]
    lat = pa.Column(float, pa.Check.in_range(g["lat_min"], g["lat_max"]))
    lng = pa.Column(float, pa.Check.in_range(g["lng_min"], g["lng_max"]))
    return pa.DataFrameSchema(
        {
            "order_id": pa.Column(str, unique=True),
            "order_status": pa.Column(str, pa.Check.isin(["delivered"])),
            "order_approved_at": pa.Column("datetime64[ns]", coerce=True),
            "order_delivered_customer_date": pa.Column("datetime64[ns]", coerce=True),
            "order_estimated_delivery_date": pa.Column("datetime64[ns]", coerce=True),
            "is_late": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
            "delivery_days": pa.Column(float, pa.Check.ge(0)),
            "primary_seller_id": pa.Column(str),
            "seller_state": pa.Column(str, pa.Check.str_length(2, 2)),
            "customer_state": pa.Column(str, pa.Check.str_length(2, 2)),
            "seller_lat": lat,
            "seller_lng": lng,
            "customer_lat": lat,
            "customer_lng": lng,
            "n_items": pa.Column(int, pa.Check.ge(1), coerce=True),
            "n_distinct_sellers": pa.Column(int, pa.Check.ge(1), coerce=True),
            "total_price": pa.Column(float, pa.Check.ge(0)),
            "total_freight": pa.Column(float, pa.Check.ge(0)),
        },
        strict=False,
    )


def validate_order_table(df):
    """Validate the built order table; raises pandera SchemaErrors on any violation."""
    return order_table_schema().validate(df, lazy=True)


def validate_table(name: str, df):
    """Validate one raw table; raises pandera SchemaErrors listing every violation."""
    return SCHEMAS[name].validate(df, lazy=True)
