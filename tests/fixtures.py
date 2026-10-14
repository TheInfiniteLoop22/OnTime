"""Synthetic mini-Olist generator for integration tests and CI.

Produces the same 9 CSVs (same columns, same quirks such as the `lenght` typo, duplicate geolocation
rows, an out-of-Brazil outlier, missing zip prefixes, cancelled orders) at a tiny scale so the REAL
pipeline code can run end to end in seconds without the Kaggle download.

The data is random: it exercises code paths, it says nothing about the real world.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STATES = ["SP", "RJ", "MG", "BA", "PR", "RS", "SC", "CE"]
CATEGORIES = [
    "bed_bath_table",
    "health_beauty",
    "toys",
    "auto",
    "telephony",
    "baby",
    "garden_tools",
]
ZIP_LAT_LNG = {  # a handful of real-looking prefixes with centroids
    "01000": (-23.55, -46.63, "SP"),
    "20000": (-22.91, -43.17, "RJ"),
    "30000": (-19.92, -43.94, "MG"),
    "40000": (-12.97, -38.50, "BA"),
    "80000": (-25.43, -49.27, "PR"),
    "90000": (-30.03, -51.22, "RS"),
    "88000": (-27.59, -48.55, "SC"),
    "60000": (-3.73, -38.52, "CE"),
}


def make_tiny_olist(out_dir: Path, n_orders: int = 1800, seed: int = 0) -> Path:
    """Write the 9 raw CSVs into `out_dir` and return it."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = list(ZIP_LAT_LNG)

    # geolocation: several noisy points per prefix, exact duplicates, one outlier outside Brazil
    rows = []
    for z, (lat, lng, st) in ZIP_LAT_LNG.items():
        for _ in range(6):
            rows.append((z, lat + rng.normal(0, 0.02), lng + rng.normal(0, 0.02), "city", st))
    rows += rows[:4]  # exact duplicates
    rows.append(("01000", 40.4, -3.7, "madrid", "SP"))  # outlier
    pd.DataFrame(rows, columns=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
                                "geolocation_city", "geolocation_state"]).to_csv(
        out_dir / "olist_geolocation_dataset.csv", index=False)  # fmt: skip

    # sellers (one with a prefix missing from geolocation) and customers
    n_sellers = 25
    seller_zip = rng.choice(zips, n_sellers).tolist()
    seller_zip[0] = "99999"
    sellers = pd.DataFrame(
        {
            "seller_id": [f"s{i:03d}" + "0" * 28 for i in range(n_sellers)],
            "seller_zip_code_prefix": seller_zip,
            "seller_city": "city",
            "seller_state": [ZIP_LAT_LNG.get(z, (0, 0, "SP"))[2] for z in seller_zip],
        }
    )
    sellers.to_csv(out_dir / "olist_sellers_dataset.csv", index=False)

    cust_zip = rng.choice(zips, n_orders)
    customers = pd.DataFrame(
        {
            "customer_id": [f"c{i:05d}" + "0" * 26 for i in range(n_orders)],
            "customer_unique_id": [f"u{i:05d}" + "0" * 26 for i in range(n_orders)],
            "customer_zip_code_prefix": cust_zip,
            "customer_city": "city",
            "customer_state": [ZIP_LAT_LNG[z][2] for z in cust_zip],
        }
    )
    customers.to_csv(out_dir / "olist_customers_dataset.csv", index=False)

    # products + translation
    n_prod = 40
    products = pd.DataFrame(
        {
            "product_id": [f"p{i:03d}" + "0" * 28 for i in range(n_prod)],
            "product_category_name": rng.choice(CATEGORIES, n_prod),
            "product_name_lenght": 30.0,
            "product_description_lenght": 300.0,
            "product_photos_qty": rng.integers(1, 6, n_prod).astype(float),
            "product_weight_g": rng.integers(100, 8000, n_prod).astype(float),
            "product_length_cm": rng.integers(10, 60, n_prod).astype(float),
            "product_height_cm": rng.integers(5, 40, n_prod).astype(float),
            "product_width_cm": rng.integers(5, 40, n_prod).astype(float),
        }
    )
    products.to_csv(out_dir / "olist_products_dataset.csv", index=False)
    pd.DataFrame(
        {"product_category_name": CATEGORIES, "product_category_name_english": CATEGORIES}
    ).to_csv(out_dir / "product_category_name_translation.csv", index=False)

    # orders: approvals spread over a window that covers train / validation / test / censoring
    start, end = pd.Timestamp("2017-06-01"), pd.Timestamp("2018-08-10")
    span_s = int((end - start).total_seconds())
    approved = start + pd.to_timedelta(rng.integers(0, span_s, n_orders), unit="s")
    purchase = approved - pd.to_timedelta(rng.integers(60, 3600 * 12, n_orders), unit="s")
    promised_days = rng.integers(10, 40, n_orders)
    estimated = (approved + pd.to_timedelta(promised_days, unit="D")).normalize()
    # lateness depends a bit on distance-ish and a seasonal bump so models have something to learn
    base = (
        0.05
        + 0.10 * ((approved.month == 3) & (approved.year == 2018))
        + 0.03 * (rng.random(n_orders) < 0.2)
    )
    late = rng.random(n_orders) < base
    actual_days = np.where(late, promised_days + rng.integers(1, 12, n_orders),
                           rng.integers(3, np.maximum(promised_days - 1, 4)))  # fmt: skip
    delivered = (
        approved
        + pd.to_timedelta(actual_days, unit="D")
        + pd.to_timedelta(rng.integers(0, 3600 * 20, n_orders), unit="s")
    )
    status = np.where(rng.random(n_orders) < 0.03, "canceled", "delivered")
    delivered = pd.Series(delivered).where(status == "delivered")
    # undelivered tail: orders approved in the final days are not yet delivered (censoring)
    tail = approved > end - pd.Timedelta(days=5)
    status = np.where(tail & (rng.random(n_orders) < 0.5), "shipped", status)
    delivered = delivered.where(status == "delivered")
    orders = pd.DataFrame(
        {
            "order_id": [f"o{i:05d}" + "0" * 26 for i in range(n_orders)],
            "customer_id": customers["customer_id"],
            "order_status": status,
            "order_purchase_timestamp": purchase,
            "order_approved_at": approved,
            "order_delivered_carrier_date": approved + pd.Timedelta(days=2),
            "order_delivered_customer_date": delivered,
            "order_estimated_delivery_date": estimated,
        }
    )
    orders.loc[orders.index[:3], "order_approved_at"] = pd.NaT  # missing approvals
    orders.to_csv(out_dir / "olist_orders_dataset.csv", index=False)

    # items: 1-3 per order, mostly single seller
    item_rows = []
    for oid in orders["order_id"]:
        k = int(rng.choice([1, 1, 1, 2, 3]))
        sid = sellers["seller_id"].iloc[int(rng.integers(0, n_sellers))]
        for j in range(1, k + 1):
            s_use = (
                sid
                if rng.random() < 0.9
                else sellers["seller_id"].iloc[int(rng.integers(0, n_sellers))]
            )
            item_rows.append(
                {"order_id": oid, "order_item_id": j,
                 "product_id": products["product_id"].iloc[int(rng.integers(0, n_prod))],
                 "seller_id": s_use,
                 "shipping_limit_date": pd.Timestamp("2018-01-01"),  # reset below, relative to approval
                 "price": float(np.round(rng.uniform(10, 400), 2)),
                 "freight_value": float(np.round(rng.uniform(5, 60), 2))}
            )  # fmt: skip
    items = pd.DataFrame(item_rows)
    # make shipping limits relative to approval so handling windows are sensible
    appr_map = dict(zip(orders["order_id"], orders["order_approved_at"], strict=True))
    items["shipping_limit_date"] = [
        (appr_map[o] if pd.notna(appr_map[o]) else pd.Timestamp("2017-07-01"))
        + pd.Timedelta(days=3)
        for o in items["order_id"]
    ]
    items.to_csv(out_dir / "olist_order_items_dataset.csv", index=False)

    # payments and reviews
    pay = pd.DataFrame(
        {
            "order_id": orders["order_id"],
            "payment_sequential": 1,
            "payment_type": rng.choice(
                ["credit_card", "boleto", "voucher"], n_orders, p=[0.7, 0.25, 0.05]
            ),
            "payment_installments": rng.integers(1, 8, n_orders),
            "payment_value": np.round(rng.uniform(20, 500, n_orders), 2),
        }
    )
    pay.to_csv(out_dir / "olist_order_payments_dataset.csv", index=False)
    pd.DataFrame(
        {
            "review_id": [f"r{i:05d}" + "0" * 26 for i in range(n_orders)],
            "order_id": orders["order_id"],
            "review_score": np.where(
                late, rng.integers(1, 3, n_orders), rng.integers(3, 6, n_orders)
            ),
            "review_comment_title": None,
            "review_comment_message": None,
            "review_creation_date": approved + pd.Timedelta(days=12),
            "review_answer_timestamp": approved + pd.Timedelta(days=13),
        }
    ).to_csv(out_dir / "olist_order_reviews_dataset.csv", index=False)
    return out_dir


def write_test_config(tmp: Path, raw_dir: Path) -> Path:
    """Copy configs/config.yaml with every output path redirected into `tmp`."""
    from ontime.config import CONFIG_PATH

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    p = cfg["paths"]
    p.update(
        raw_dir=str(raw_dir),
        interim_dir=str(tmp / "interim"),
        processed_dir=str(tmp / "processed"),
        models_dir=str(tmp / "models"),
        reports_dir=str(tmp / "reports"),
        figures_dir=str(tmp / "reports" / "figures"),
        orders_parquet=str(tmp / "interim" / "orders.parquet"),
        order_universe_parquet=str(tmp / "interim" / "order_universe.parquet"),
        features_parquet=str(tmp / "processed" / "features.parquet"),
    )
    cfg["tuning"]["bootstrap_resamples"] = 50
    for d in ("interim", "processed", "models", "reports"):
        (tmp / d).mkdir(parents=True, exist_ok=True)
    out = tmp / "config.test.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out
