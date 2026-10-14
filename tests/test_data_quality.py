import pandas as pd
import pandera.errors
import pytest

from ontime.data.download import missing_files
from ontime.data.quality import duplicate_keys, geolocation_outliers, null_percentages
from ontime.data.schemas import validate_table


def test_missing_files_reports_absent(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    assert missing_files(tmp_path, ["a.csv", "b.csv"]) == ["b.csv"]


def test_null_percentages_only_lists_columns_with_nulls():
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": [1, 2, 3, 4]})
    out = null_percentages({"t": df})
    assert out["column"].tolist() == ["a"]
    assert out["null_pct"].iloc[0] == 25.0


def test_duplicate_keys_counts_repeats():
    tables = {
        "orders": pd.DataFrame({"order_id": ["a", "a", "b"]}),
        "order_items": pd.DataFrame({"order_id": ["a", "a"], "order_item_id": [1, 2]}),
        "products": pd.DataFrame({"product_id": ["p"]}),
        "sellers": pd.DataFrame({"seller_id": ["s"]}),
        "customers": pd.DataFrame({"customer_id": ["c"]}),
        "payments": pd.DataFrame({"order_id": ["a"], "payment_sequential": [1]}),
        "reviews": pd.DataFrame({"review_id": ["r", "r"]}),
        "category_translation": pd.DataFrame({"product_category_name": ["x"]}),
    }
    out = duplicate_keys(tables).set_index("table")["duplicate_rows"]
    assert out["orders"] == 1
    assert out["order_items"] == 0
    assert out["reviews"] == 1


def test_geolocation_outliers_flags_points_outside_brazil():
    geo = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": ["01", "02"],
            "geolocation_lat": [-23.5, 40.0],  # second point is in the northern hemisphere
            "geolocation_lng": [-46.6, -3.7],
            "geolocation_city": ["sp", "madrid"],
            "geolocation_state": ["SP", "SP"],
        }
    )
    assert geolocation_outliers(geo)["outside Brazil bounding box"] == 1


def test_schema_rejects_unknown_order_status():
    df = pd.DataFrame(
        {
            "order_id": ["a"],
            "customer_id": ["c"],
            "order_status": ["teleported"],
            "order_purchase_timestamp": [pd.Timestamp("2018-01-01")],
            "order_approved_at": [pd.NaT],
            "order_delivered_carrier_date": [pd.NaT],
            "order_delivered_customer_date": [pd.NaT],
            "order_estimated_delivery_date": [pd.Timestamp("2018-01-20")],
        }
    )
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_table("orders", df)
