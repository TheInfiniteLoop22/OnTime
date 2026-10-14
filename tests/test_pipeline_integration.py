"""End-to-end smoke test on a synthetic mini-Olist dataset (what CI runs; no Kaggle download).

Runs the REAL stages: raw CSV load + pandera validation -> DuckDB order table -> features with the
leak-safe history -> baselines. Asserts the invariants that matter, not particular numbers.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fixtures import make_tiny_olist, write_test_config  # noqa: E402


@pytest.fixture(scope="module")
def pipeline_outputs(tmp_path_factory):
    import os

    tmp = tmp_path_factory.mktemp("tiny_olist")
    raw = make_tiny_olist(tmp / "raw", n_orders=1800, seed=0)
    cfg_path = write_test_config(tmp, raw)
    old = os.environ.get("ONTIME_CONFIG")
    os.environ["ONTIME_CONFIG"] = str(cfg_path)
    try:
        from ontime.config import load_config
        from ontime.data.build_orders import build as build_orders
        from ontime.data.load import load_all
        from ontime.data.schemas import validate_table
        from ontime.features.build import build as build_features
        from ontime.models.baselines import run_baselines

        tables = load_all()
        for name, df in tables.items():
            validate_table(name, df)  # raises on schema violation
        orders = build_orders()
        feats = build_features()
        base = run_baselines(write=False)
        yield {"cfg": load_config(), "orders": orders, "features": feats, "baselines": base,
               "tmp": tmp}  # fmt: skip
    finally:
        if old is None:
            os.environ.pop("ONTIME_CONFIG", None)
        else:
            os.environ["ONTIME_CONFIG"] = old


def test_order_table_is_one_row_per_delivered_order(pipeline_outputs):
    o = pipeline_outputs["orders"]
    assert o["order_id"].is_unique
    assert (o["order_status"] == "delivered").all()
    assert set(o["is_late"].unique()) <= {0, 1}
    assert (o["order_delivered_customer_date"] >= o["order_approved_at"]).all()
    cutoff = pd.Timestamp(pipeline_outputs["cfg"]["splits"]["censoring_cutoff"])
    assert (o["order_approved_at"] < cutoff).all()


def test_geo_fallback_and_outlier_handling(pipeline_outputs):
    o = pipeline_outputs["orders"]
    assert o["seller_lat"].between(-34, 6).all() and o["seller_lng"].between(-74, -34).all()
    assert o["seller_geo_imputed"].any()  # the seller whose zip is missing from geolocation


def test_feature_table_matches_declared_features_and_splits(pipeline_outputs):
    from ontime.features.build import FEATURE_COLUMNS, META_COLUMNS

    f = pipeline_outputs["features"]
    assert list(f.columns) == META_COLUMNS + FEATURE_COLUMNS
    assert len(f) == len(pipeline_outputs["orders"])
    assert set(f["split"].unique()) == {"train", "validation", "test"}
    assert (
        f.groupby("split")["approved_at"]
        .agg(["min", "max"])
        .pipe(
            lambda r: (
                r.loc["train", "max"]
                < r.loc["validation", "min"]
                <= r.loc["validation", "max"]
                < r.loc["test", "min"]
            )
        )
    )


def test_history_features_never_use_the_future(pipeline_outputs):
    """Spot-check against a direct recomputation on the fixture (same rule as the unit tests)."""
    o, f = pipeline_outputs["orders"], pipeline_outputs["features"]
    sample = o.sample(40, random_state=1)
    for i, r in sample.iterrows():
        known = o[
            (o["order_delivered_customer_date"] < r["order_approved_at"])
            & (o["primary_seller_id"] == r["primary_seller_id"])
        ]
        assert f.loc[i, "seller_n_past"] == len(known)


def test_baselines_run_and_beat_or_match_random(pipeline_outputs):
    b = pipeline_outputs["baselines"]
    assert b["evaluated_on"] == "validation"
    for m in ("prior_rate", "rule"):
        assert 0.0 <= b["models"][m]["metrics"]["pr_auc"]["value"] <= 1.0
    # the prior has no ranking skill
    assert b["models"]["prior_rate"]["metrics"]["roc_auc"]["value"] == pytest.approx(0.5)


def test_test_split_is_still_locked(pipeline_outputs):
    from ontime.models.splits import select_split

    with pytest.raises(PermissionError):
        select_split(pipeline_outputs["features"], "test")


def test_stage2_table_builds_with_leak_safe_handover_features(pipeline_outputs):
    from ontime.stage2.features import STAGE2_FEATURES
    from ontime.stage2.run import build_stage2_table

    t = build_stage2_table(write=False)
    assert set(STAGE2_FEATURES) <= set(t.columns)
    assert t["order_id"].is_unique and len(t) <= len(pipeline_outputs["features"])
    assert (t["t1"] >= t["approved_at"]).all()  # decision time is never before approval
    assert (t["days_to_handover"] >= 0).all()
    share = t["dest_backlog_share"].dropna()
    assert share.between(0, 1).all() and len(share) > 0.5 * len(t)
