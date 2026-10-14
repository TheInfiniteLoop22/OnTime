"""Phase 9 tests: what-if recomputation (pure) and the scoring service (needs trained models)."""

import numpy as np
import pandas as pd
import pytest

from ontime.config import project_path
from ontime.features.build import FEATURE_COLUMNS
from ontime.serving.service import MODELS, RiskService, recompute_dependent_features

READY = (
    all((project_path("models") / f"{m}.joblib").exists() for m in MODELS)
    and project_path("data/processed/features.parquet").exists()
)


def _row():
    return pd.DataFrame(
        {
            "promised_days": [10.0],
            "distance_km": [1000.0],
            "promise_per_1000km": [5.0],
            "total_freight": [20.0],
            "total_price": [100.0],
            "freight_ratio": [0.2],
        }
    )


def test_promise_edit_updates_derived_ratio_only():
    r = recompute_dependent_features(_row(), extra_promise_days=4)
    assert r.loc[0, "promised_days"] == 14.0
    assert r.loc[0, "promise_per_1000km"] == pytest.approx(14.0 / 2.0)  # distance_km/1000 + 1 = 2
    assert r.loc[0, "freight_ratio"] == 0.2 and r.loc[0, "total_freight"] == 20.0


def test_freight_edit_updates_ratio_only():
    r = recompute_dependent_features(_row(), total_freight=50.0)
    assert r.loc[0, "freight_ratio"] == pytest.approx(0.5)
    assert r.loc[0, "promised_days"] == 10.0 and r.loc[0, "promise_per_1000km"] == 5.0


def test_zero_price_gives_nan_ratio_not_inf():
    row = _row()
    row["total_price"] = 0.0
    r = recompute_dependent_features(row, total_freight=10.0)
    assert np.isnan(r.loc[0, "freight_ratio"])


def test_original_row_is_not_mutated():
    row = _row()
    recompute_dependent_features(row, extra_promise_days=3, total_freight=99.0)
    assert row.loc[0, "promised_days"] == 10.0


@pytest.mark.skipif(not READY, reason="needs trained models (make train)")
def test_service_scores_explains_and_whatif_direction():
    svc = RiskService()
    t = pd.read_parquet(project_path("data/processed/features.parquet"))
    row = t[t["split"] == "test"].head(1)[FEATURE_COLUMNS]
    for m in MODELS:
        out = svc.explain_one(row, m)
        assert 0.0 <= out["probability"] <= 1.0
        # contributions are exactly grouped model log-odds contributions
        assert set(out["contributions"].index) == set(svc.group_names)
    sweep = svc.promise_sweep(row, "logreg", 10)
    assert len(sweep) == 11 and sweep["probability"].between(0, 1).all()
    # a longer promise should not INCREASE modelled risk (negative effect in all models)
    assert sweep["probability"].iloc[-1] <= sweep["probability"].iloc[0] + 1e-9
