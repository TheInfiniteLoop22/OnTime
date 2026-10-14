"""Phase 2 tests on tiny hand-made fixtures: haversine, primary seller, target, funnel, geo."""

import numpy as np
import pandas as pd
import pytest

from ontime.data.build_orders import apply_population_filters, build_all_orders
from ontime.features.geo import haversine_km

T = pd.Timestamp


def test_haversine_known_distances():
    # Sao Paulo -> Rio de Janeiro is ~360 km straight line
    d = haversine_km(-23.55, -46.63, -22.91, -43.17)
    assert 340 < float(d) < 380
    assert float(haversine_km(-10.0, -50.0, -10.0, -50.0)) == pytest.approx(0.0)


def test_haversine_is_symmetric_and_vectorized():
    a = haversine_km(np.array([-23.5, -3.1]), np.array([-46.6, -60.0]), -15.8, -47.9)
    b = haversine_km(-15.8, -47.9, np.array([-23.5, -3.1]), np.array([-46.6, -60.0]))
    assert a.shape == (2,)
    np.testing.assert_allclose(a, b)


def _tables(delivered, estimated):
    """Two orders: o1 single-seller; o2 two sellers (s2 has the higher item value)."""
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "customer_id": ["c1", "c2"],
            "order_status": ["delivered", "delivered"],
            "order_purchase_timestamp": [T("2018-01-01 08:00")] * 2,
            "order_approved_at": [T("2018-01-01 09:00")] * 2,
            "order_delivered_carrier_date": [T("2018-01-02")] * 2,
            "order_delivered_customer_date": delivered,
            "order_estimated_delivery_date": estimated,
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o2", "o2"],
            "order_item_id": [1, 1, 2, 3],
            "product_id": ["p1", "p1", "p2", "p2"],
            "seller_id": ["s1", "s1", "s2", "s2"],
            "shipping_limit_date": [T("2018-01-03")] * 4,
            "price": [10.0, 5.0, 30.0, 40.0],  # s1 total 5 vs s2 total 70 in o2
            "freight_value": [1.0, 1.0, 1.0, 1.0],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_category_name": ["cat_a", "cat_b"],
            "product_photos_qty": [1.0, 2.0],
            "product_weight_g": [100.0, 200.0],
            "product_length_cm": [10.0, 10.0],
            "product_height_cm": [10.0, 10.0],
            "product_width_cm": [10.0, 10.0],
        }
    )
    sellers = pd.DataFrame(
        {
            "seller_id": ["s1", "s2"],
            "seller_zip_code_prefix": ["01000", "99999"],  # s2 prefix has no geolocation
            "seller_state": ["SP", "RJ"],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "customer_unique_id": ["u1", "u2"],
            "customer_zip_code_prefix": ["01000", "02000"],
            "customer_state": ["SP", "SP"],
        }
    )
    geolocation = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": ["01000", "01000", "01000", "02000", "30000", "30000"],
            "geolocation_lat": [-23.5, -23.6, 40.0, -23.4, -22.9, -22.95],  # 40.0 is an outlier
            "geolocation_lng": [-46.6, -46.7, -3.7, -46.5, -43.2, -43.1],
            "geolocation_state": ["SP", "SP", "SP", "SP", "RJ", "RJ"],
        }
    )
    payments = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o2"],
            "payment_sequential": [1, 1, 2],
            "payment_type": ["credit_card", "voucher", "boleto"],
            "payment_installments": [3, 1, 1],
            "payment_value": [11.0, 10.0, 62.0],
        }
    )
    translation = pd.DataFrame(
        {"product_category_name": ["cat_a"], "product_category_name_english": ["Cat A"]}
    )
    return {
        "orders": orders,
        "order_items": items,
        "products": products,
        "sellers": sellers,
        "customers": customers,
        "geolocation": geolocation,
        "payments": payments,
        "category_translation": translation,
    }


@pytest.fixture
def built():
    delivered = [T("2018-01-10 23:00"), T("2018-01-10 01:00")]
    estimated = [T("2018-01-10 00:00"), T("2018-01-10 00:00")]
    return build_all_orders(_tables(delivered, estimated)).set_index("order_id")


def test_one_row_per_order(built):
    assert len(built) == 2 and built.index.is_unique


def test_primary_seller_is_highest_item_value(built):
    assert built.loc["o1", "primary_seller_id"] == "s1"
    assert built.loc["o2", "primary_seller_id"] == "s2"  # 70 > 5
    assert built.loc["o2", "n_distinct_sellers"] == 2


def test_target_compares_dates_not_timestamps(built):
    # o1 delivered 2018-01-10 23:00, estimate 2018-01-10 -> same DATE -> NOT late.
    assert built.loc["o1", "is_late"] == 0
    assert built.loc["o2", "is_late"] == 0


def test_target_late_when_delivered_date_after_estimate():
    delivered = [T("2018-01-11 00:01"), T("2018-01-09 10:00")]
    estimated = [T("2018-01-10 00:00"), T("2018-01-10 00:00")]
    b = build_all_orders(_tables(delivered, estimated)).set_index("order_id")
    assert b.loc["o1", "is_late"] == 1
    assert b.loc["o2", "is_late"] == 0


def test_geo_outlier_dropped_and_median_centroid(built):
    # outlier (40.0, -3.7) removed -> median of (-23.5, -23.6) = -23.55
    assert built.loc["o1", "customer_lat"] == pytest.approx(-23.55)


def test_missing_zip_falls_back_to_state_centroid_and_is_flagged(built):
    assert bool(built.loc["o2", "seller_geo_imputed"]) is True
    assert bool(built.loc["o1", "seller_geo_imputed"]) is False
    # RJ state centroid = median of the two RJ points
    assert built.loc["o2", "seller_lat"] == pytest.approx(-22.925)


def test_item_and_payment_aggregates(built):
    assert built.loc["o2", "total_price"] == pytest.approx(75.0)
    assert built.loc["o2", "total_weight_g"] == pytest.approx(500.0)
    assert built.loc["o2", "total_volume_cm3"] == pytest.approx(3000.0)
    assert built.loc["o2", "payment_type_dominant"] == "boleto"  # 62 > 10 by value
    assert built.loc["o1", "top_item_category"] == "Cat A"  # translated


def test_forbidden_columns_not_in_table(built):
    assert "order_delivered_carrier_date" not in built.columns
    assert not any(c.startswith("review") for c in built.columns)


def test_funnel_counts_each_exclusion():
    df = pd.DataFrame(
        {
            "order_id": list("abcdef"),
            "order_status": ["delivered"] * 5 + ["canceled"],
            "order_approved_at": [T("2018-01-01")] * 3 + [T("2018-09-01"), pd.NaT, T("2018-01-01")],
            "order_delivered_customer_date": [
                T("2018-01-05"),
                T("2017-12-31"),  # before approval
                T("2018-01-06"),
                T("2018-09-05"),
                T("2018-01-06"),
                pd.NaT,
            ],
            "order_estimated_delivery_date": [T("2018-01-20")] * 6,
            "primary_seller_id": ["s", "s", None, "s", "s", "s"],
        }
    )
    out, funnel = apply_population_filters(df, "2018-08-20")
    assert out["order_id"].tolist() == ["a"]
    assert funnel["rows_remaining"].tolist()[0] == 6
    assert funnel["rows_dropped"].sum() == 5
