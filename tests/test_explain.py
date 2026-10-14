"""Phase 8 tests: template coverage, additive decomposition, reason-code logic."""

import joblib
import numpy as np
import pandas as pd
import pytest

from ontime.config import project_path
from ontime.explain.agreement import direction_table, global_importance, rank_agreement
from ontime.explain.contributions import source_feature
from ontime.explain.reason_codes import (
    DISPLAY_FIELDS,
    add_display_fields,
    build_reason_codes,
    feature_to_group,
    group_contributions,
    group_text,
    load_templates,
    template_fields,
    validate_templates,
)
from ontime.features.build import FEATURE_COLUMNS

TEMPLATES = load_templates()


# ------------------------------------------------------------------ template coverage (acceptance)
def test_every_model_feature_maps_to_exactly_one_group_and_template():
    assert validate_templates(TEMPLATES) == []
    f2g = feature_to_group(TEMPLATES)
    assert set(f2g) == set(FEATURE_COLUMNS)  # no orphan, no extra
    for g in set(f2g.values()):
        assert group_text(TEMPLATES, g, "positive") and group_text(TEMPLATES, g, "negative")


def test_template_placeholders_are_known_fields():
    allowed = set(FEATURE_COLUMNS) | DISPLAY_FIELDS
    for g in TEMPLATES["groups"]:
        for kind in ("positive", "negative"):
            assert template_fields(group_text(TEMPLATES, g, kind)) <= allowed, g


def test_feature_in_two_groups_is_rejected():
    bad = {"groups": {"a": {"features": ["x"], "text": "t"}, "b": {"features": ["x"], "text": "t"}}}
    with pytest.raises(ValueError):
        feature_to_group(bad)


# ------------------------------------------------------------------ reason-code logic
def _row_frame():
    """One order whose values exercise every display field."""
    d = {f: [1.0] for f in FEATURE_COLUMNS}
    d.update(
        approve_dow=[2], approve_month=[6], approve_hour=[14], same_state=[0],
        is_black_friday_window=[0], geo_imputed=[0], customer_region=["NE"],
        seller_region=["SE"], category_group=["auto"], payment_type=["boleto"],
        distance_km=[2140.0], promised_days=[12.4], handling_window_days=[3.0],
        seller_late_rate_90d_shrunk=[0.21], seller_n_past=[85.0],
    )  # fmt: skip
    return pd.DataFrame(d)


def test_reason_codes_pick_top_groups_by_sign_and_threshold():
    X = _row_frame()
    groups = list(TEMPLATES["groups"])
    contrib = pd.DataFrame(0.0, index=X.index, columns=groups)
    contrib.loc[0, "distance"] = 0.9
    contrib.loc[0, "seller_record"] = 0.5
    contrib.loc[0, "route_history"] = 0.3
    contrib.loc[0, "promise_window"] = 0.1  # 4th positive: must be cut (n_positive = 3)
    contrib.loc[0, "same_state"] = -0.4
    contrib.loc[0, "payment"] = -0.01  # below min_abs_contribution: dropped
    out = build_reason_codes(contrib, add_display_fields(X), TEMPLATES).iloc[0]
    assert out["risk_groups"] == ["distance", "seller_record", "route_history"]
    assert out["protective_groups"] == ["same_state"]
    assert out["risk_factors"][0].endswith("(adds risk)")
    assert "2,140 km" in out["risk_factors"][0]
    assert out["protective_factors"][0].endswith("(reduces risk)")
    assert "different states" in out["protective_factors"][0]


def test_missing_values_render_without_crashing():
    X = _row_frame()
    X.loc[0, "distance_km"] = np.nan
    contrib = pd.DataFrame(0.0, index=X.index, columns=list(TEMPLATES["groups"]))
    contrib.loc[0, "distance"] = 1.0
    out = build_reason_codes(contrib, add_display_fields(X), TEMPLATES).iloc[0]
    assert "n/a" in out["risk_factors"][0]


def test_group_contributions_sum_features():
    f2g = feature_to_group(TEMPLATES)
    c = pd.DataFrame(0.0, index=[0], columns=FEATURE_COLUMNS)
    c.loc[0, "distance_km"], c.loc[0, "log_distance_km"], c.loc[0, "promise_per_1000km"] = (
        0.2,
        0.3,
        -0.1,
    )
    g = group_contributions(c, f2g)
    assert g.loc[0, "distance"] == pytest.approx(0.4)
    assert g.shape[1] == len(TEMPLATES["groups"])


def test_source_feature_mapping():
    assert source_feature("missingindicator_freight_ratio") == "freight_ratio"
    assert source_feature("distance_km") == "distance_km"
    assert source_feature("category_group_auto") == "category_group"
    assert source_feature("approve_dow_3") == "approve_dow"
    assert source_feature("payment_type_infrequent_sklearn") == "payment_type"
    with pytest.raises(KeyError):
        source_feature("mystery_column")


# ------------------------------------------------------------------ decompositions on real models
MODELS_PRESENT = (
    all((project_path("models") / f"{m}.joblib").exists() for m in ("logreg", "lightgbm", "ebm"))
    and project_path("data/processed/features.parquet").exists()
)


@pytest.mark.skipif(not MODELS_PRESENT, reason="needs trained models and features (make train)")
@pytest.mark.parametrize("name", ["logreg", "lightgbm", "ebm"])
def test_contributions_add_up_to_model_log_odds(name):
    from ontime.explain.contributions import EXTRACTORS
    from ontime.models.data import load_feature_table, prepare_frame
    from ontime.models.splits import select_split

    art = joblib.load(project_path("models") / f"{name}.joblib")
    val = select_split(load_feature_table(), "validation").head(300)
    X = prepare_frame(val, art["levels"])
    contrib, base = EXTRACTORS[name](art["model"], X)
    p = np.clip(art["model"].predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    logit = np.log(p / (1 - p))
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy() + base, logit, atol=1e-6)
    assert list(contrib.columns) == FEATURE_COLUMNS


# ------------------------------------------------------------------ agreement helpers
def test_rank_agreement_perfect_and_reversed():
    a = pd.Series({"f1": 3.0, "f2": 2.0, "f3": 1.0})
    r_same = rank_agreement({"a": a, "b": a * 2}, top_k=2)
    r_rev = rank_agreement({"a": a, "b": a[::-1].set_axis(a.index)}, top_k=2)
    assert r_same["spearman"]["a vs b"] == pytest.approx(1.0)
    assert r_rev["spearman"]["a vs b"] == pytest.approx(-1.0)
    assert r_same["top_k_overlap"]["a vs b"] == 1.0


def test_direction_table_detects_agreement_and_marginal_difference():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"f": rng.normal(size=500)})
    up = pd.DataFrame({"f": 0.5 * x["f"]})
    down = pd.DataFrame({"f": -0.5 * x["f"]})
    y = (x["f"] > 0).astype(int).to_numpy()
    t = direction_table({"m1": up, "m2": up * 2, "m3": down}, x, ["f"], y).iloc[0]
    assert t["marginal"] == "higher value -> more risk"
    assert t["conflict"] and not t["agree"]
    imp = global_importance(up)
    assert imp["f"] > 0
