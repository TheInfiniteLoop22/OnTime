"""Feature-family tests on tiny hand-made frames, plus feature-list safety and split tests."""

import numpy as np
import pandas as pd
import pytest

from ontime.config import load_config
from ontime.features import build as fb
from ontime.features.calendar import add_calendar_features, brazil_holidays
from ontime.features.geo import add_geo_features
from ontime.features.product import add_product_features, group_category
from ontime.models.splits import assign_split, split_boundaries

T = pd.Timestamp
CFG = load_config()


# ------------------------------------------------------------------ feature-list safety
def test_no_forbidden_column_in_feature_list():
    fb.assert_feature_list_is_safe()
    assert not set(fb.FEATURE_COLUMNS) & set(fb.FORBIDDEN_FEATURES)
    assert not any(f.startswith("review_") for f in fb.FEATURE_COLUMNS)
    assert len(set(fb.FEATURE_COLUMNS)) == len(fb.FEATURE_COLUMNS)


def test_feature_list_guard_raises_on_bad_name():
    with pytest.raises(ValueError):
        fb.assert_feature_list_is_safe(["distance_km", "order_delivered_customer_date"])
    with pytest.raises(ValueError):
        fb.assert_feature_list_is_safe(["review_score"])


def test_meta_columns_are_not_features():
    assert not set(fb.META_COLUMNS) & set(fb.FEATURE_COLUMNS)


# ------------------------------------------------------------------ splits
def test_config_splits_are_contiguous():
    val_start, test_start = split_boundaries(CFG)
    assert val_start < test_start


def test_assign_split_boundaries():
    a = pd.Series(
        [
            T("2018-01-31 23:59:59"),
            T("2018-02-01 00:00:00"),
            T("2018-04-30 23:59:59"),
            T("2018-05-01 00:00:00"),
        ]
    )
    assert assign_split(a, CFG).tolist() == ["train", "validation", "validation", "test"]


def test_split_gap_in_config_is_rejected():
    bad = {"splits": {**CFG["splits"], "val_start": "2018-02-05"}}
    with pytest.raises(ValueError):
        split_boundaries(bad)


# ------------------------------------------------------------------ geo
def test_geo_features():
    df = pd.DataFrame(
        {
            "seller_lat": [-23.5, -3.1],
            "seller_lng": [-46.6, -60.0],
            "customer_lat": [-23.5, -23.5],
            "customer_lng": [-46.6, -46.6],
            "seller_state": ["SP", "AM"],
            "customer_state": ["SP", "SP"],
            "seller_geo_imputed": [False, True],
            "customer_geo_imputed": [False, False],
            "n_distinct_sellers": [1, 2],
        }
    )
    g = add_geo_features(df)
    assert g.loc[0, "distance_km"] == pytest.approx(0.0, abs=1e-6)
    assert g.loc[1, "distance_km"] > 2500  # Manaus -> Sao Paulo
    assert g["same_state"].tolist() == [1, 0]
    assert g["seller_region"].tolist() == ["SE", "N"]
    assert g["geo_imputed"].tolist() == [0, 1]


# ------------------------------------------------------------------ calendar
def _cal(approved, estimated, purchase=None, limit=None, distance=0.0):
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [T(purchase or approved) - pd.Timedelta(hours=3)],
            "order_approved_at": [T(approved)],
            "order_estimated_delivery_date": [T(estimated)],
            "max_shipping_limit_date": [T(limit or approved) + pd.Timedelta(days=4)],
            "distance_km": [distance],
        }
    )
    return add_calendar_features(
        df, CFG["features"]["black_friday_dates"], CFG["features"]["black_friday_window_days"]
    ).iloc[0]


def test_promise_and_handling_windows():
    r = _cal("2018-03-05 12:00", "2018-03-25", distance=1000.0)
    assert r["promised_days"] == pytest.approx(19.5)
    assert r["approval_lag_hours"] == pytest.approx(3.0)
    assert r["handling_window_days"] == pytest.approx(4.0)
    assert r["promise_per_1000km"] == pytest.approx(19.5 / 2)


def test_holiday_features():
    # 2018-04-30 -> next national holiday is 2018-05-01 (Labour Day): 1 day away
    r = _cal("2018-04-30 10:00", "2018-05-10")
    assert r["days_to_next_holiday"] == 1
    assert r["holidays_in_promise_window"] == 1  # May 1 falls in (Apr 30, May 10]
    # holiday ON the approval day counts as 0 days away but is not "inside" the window
    r2 = _cal("2018-05-01 10:00", "2018-05-10")
    assert r2["days_to_next_holiday"] == 0
    assert r2["holidays_in_promise_window"] == 0


def test_calendar_parts_and_black_friday():
    r = _cal("2017-11-25 23:30", "2017-12-20")  # Saturday, day after Black Friday 2017-11-24
    assert r["approve_dow"] == 5 and r["is_weekend_approval"] == 1
    assert r["approve_hour"] == 23 and r["approve_month"] == 11
    assert r["is_black_friday_window"] == 1
    assert _cal("2017-12-20 10:00", "2018-01-10")["is_black_friday_window"] == 0


def test_brazil_holidays_sorted_and_national_only():
    h = brazil_holidays(2018, 2018)
    assert (np.diff(h.astype("int64")) > 0).all()
    assert np.datetime64("2018-09-07") in h  # Independence Day


# ------------------------------------------------------------------ product
def test_category_grouping_rules():
    groups = {"toys_baby": ["toys", "baby"], "auto": ["auto"]}
    s = pd.Series(["toys", "auto", "something_rare", None])
    assert group_category(s, groups).tolist() == ["toys_baby", "auto", "other", "unknown"]


def test_every_observed_category_group_in_config_is_unique():
    cats = [c for cs in CFG["features"]["category_groups"].values() for c in cs]
    assert len(cats) == len(set(cats)), "a category appears in two groups"


def test_product_features():
    df = pd.DataFrame(
        {
            "n_items": [2, 1],
            "total_price": [100.0, 0.0],
            "total_freight": [25.0, 10.0],
            "total_weight_g": [3000.0, np.nan],
            "total_volume_cm3": [12000.0, np.nan],
            "max_item_weight_g": [2000.0, np.nan],
            "top_item_category": ["toys", None],
            "product_photos_qty_mean": [2.0, np.nan],
            "payment_type_dominant": ["credit_card", None],
            "payment_installments_max": [3, 1],
        }
    )
    p = add_product_features(df, {"toys_baby": ["toys"]})
    assert p.loc[0, "freight_ratio"] == pytest.approx(0.25)
    assert np.isnan(p.loc[1, "freight_ratio"])  # price 0 -> undefined, not inf
    assert p.loc[0, "total_weight_kg"] == pytest.approx(3.0)
    assert p.loc[0, "total_volume_l"] == pytest.approx(12.0)
    assert p["category_group"].tolist() == ["toys_baby", "unknown"]
    assert p["payment_type"].tolist() == ["credit_card", "unknown"]
