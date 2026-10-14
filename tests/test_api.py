"""API tests: payload validation always runs; endpoint tests need the trained artifacts."""

import numpy as np
import pandas as pd
import pytest

from ontime.config import load_config, project_path
from ontime.features.build import FEATURE_COLUMNS

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from ontime.api.app import create_app, to_frame  # noqa: E402

CFG = load_config()
HAVE_MODELS = (
    all(
        (project_path(CFG["paths"]["models_dir"]) / f"{m}.joblib").exists()
        for m in ("logreg", "lightgbm", "ebm")
    )
    and project_path(CFG["paths"]["features_parquet"]).exists()
)
needs_artifacts = pytest.mark.skipif(not HAVE_MODELS, reason="trained artifacts not present")


def _good(levels=None):
    f = {c: 1.0 for c in FEATURE_COLUMNS}
    f.update(
        customer_region="SE", seller_region="SE", category_group="auto", payment_type="credit_card"
    )
    return f


def test_to_frame_reports_missing_and_non_numeric():
    f = _good()
    f.pop("distance_km")
    with pytest.raises(ValueError, match="missing features"):
        to_frame(f, FEATURE_COLUMNS, {})
    f = _good()
    f["distance_km"] = "far"
    with pytest.raises(ValueError, match="non-numeric"):
        to_frame(f, FEATURE_COLUMNS, {})


def test_to_frame_nulls_become_nan_and_unknown_levels_warn():
    f = _good()
    f["total_price"] = None
    f["customer_region"] = "MARS"
    df, warns = to_frame(f, FEATURE_COLUMNS, {"customer_region": ["SE", "S"]})
    assert np.isnan(df.loc[0, "total_price"])
    assert any("customer_region" in w for w in warns)


@needs_artifacts
def test_endpoints_end_to_end():
    c = TestClient(create_app())
    assert c.get("/health").json()["status"] == "ok"
    t = pd.read_parquet(project_path(CFG["paths"]["features_parquet"]))
    oid = t[t["split"] == "test"]["order_id"].iloc[0]
    r = c.get(f"/demo/orders/{oid}")
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0 and body["risk_ratio"] > 0
    assert c.get("/demo/orders/does-not-exist").status_code == 404
    assert c.post("/score", json={"features": {"distance_km": 5}}).status_code == 422
    assert c.post("/score", json={"features": _good(), "model": "nope"}).status_code == 422
