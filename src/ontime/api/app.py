"""S3 - HTTP scoring service (FastAPI). Thin wrapper: all logic stays in `src/ontime`.

Endpoints
  GET  /health                  liveness + which artifacts are loaded
  GET  /model-info              model names, training window, reported test metrics, disclosures
  POST /score                   stage 1 (at approval): probability + plain-English reasons
  POST /score/handover          stage 2 (at carrier handover): needs the 5 handover features too
  POST /promise                 S1 quantile promise-date model: days to quote at several levels
  GET  /demo/orders/{order_id}  score a known test-period order by id (demo; needs local data)

Run:  uvicorn ontime.api.app:app --port 8000        (docs at /docs)

Honesty notes carried into every response: the stage-1 probability is calibrated on validation and
the base rate moved between periods, so `risk_ratio` (probability / training base rate) is the
number to act on; stage 2 is a ranking-first model; the promise model is a development result.
"""

from __future__ import annotations

import json
import logging
from functools import cached_property
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ontime.config import load_config, project_path
from ontime.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from ontime.stage2.features import STAGE2_FEATURES

log = logging.getLogger(__name__)
API_VERSION = "1.0"


# ------------------------------------------------------------------------------ request/response
class FeaturePayload(BaseModel):
    """One order's features as {name: value}. Missing numeric values may be null."""

    features: dict[str, Any] = Field(
        description="Feature name -> value. Stage 1 needs the 37 approval-time features "
        "(see GET /model-info); /score/handover also needs the 5 handover features."
    )
    model: str | None = Field(
        default=None, description="Stage 1 only: logreg | lightgbm | ebm (default: the primary)"
    )


class ScoreResponse(BaseModel):
    stage: int
    model: str
    probability: float = Field(description="calibrated probability of a late delivery")
    raw_probability: float
    calibration: str
    risk_ratio: float = Field(description="probability / training late rate (relative risk)")
    risk_factors: list[str]
    protective_factors: list[str]
    warnings: list[str] = []


class PromiseResponse(BaseModel):
    quoted_promise_days: float | None
    promise_days_by_level: dict[str, int] = Field(
        description="alpha -> whole days to quote so that ~alpha of orders arrive in time"
    )
    recommended: dict[str, int] = Field(
        description="named operating points chosen on validation (match_olist, cov_0.95, ...)"
    )
    note: str


# ------------------------------------------------------------------------------ validation
def to_frame(features: dict[str, Any], columns: list[str], levels: dict[str, list[str]]):
    """Validate a feature dict and return (1-row DataFrame, warnings). Raises ValueError."""
    missing = [c for c in columns if c not in features]
    if missing:
        raise ValueError(f"missing features: {missing}")
    row: dict[str, Any] = {}
    bad, warns = [], []
    for c in columns:
        v = features[c]
        if c in CATEGORICAL_FEATURES:
            if v is not None and levels.get(c) and str(v) not in levels[c]:
                warns.append(f"{c}={v!r} is not a known level; treated as unknown")
            row[c] = None if v is None else str(v)
        else:
            if v is None:
                row[c] = np.nan
            else:
                try:
                    row[c] = float(v)
                except (TypeError, ValueError):
                    bad.append(c)
    if bad:
        raise ValueError(f"non-numeric values for: {bad}")
    return pd.DataFrame([row]), warns


# ------------------------------------------------------------------------------ state
class State:
    """Loads artifacts lazily so /health works (and tests start fast) before anything is scored."""

    def __init__(self) -> None:
        self.cfg = load_config()
        self.models_dir = project_path(self.cfg["paths"]["models_dir"])
        self.reports_dir = project_path(self.cfg["paths"]["reports_dir"])

    @cached_property
    def results(self) -> dict:
        return json.loads((self.reports_dir / "results.json").read_text(encoding="utf-8"))

    @property
    def primary(self) -> str:
        return self.results["primary_model"]

    @property
    def base_rate(self) -> float:
        return float(self.results["population"]["train_late_rate"])

    @cached_property
    def service(self):  # noqa: ANN201
        from ontime.serving.service import RiskService

        return RiskService()

    @cached_property
    def stage2(self) -> dict:
        return joblib.load(self.models_dir / "stage2.joblib")

    @cached_property
    def eta(self) -> dict:
        return joblib.load(self.models_dir / "eta.joblib")

    def available(self) -> dict[str, bool]:
        return {
            "stage1": all((self.models_dir / f"{m}.joblib").exists() for m in ("logreg",)),
            "stage2": (self.models_dir / "stage2.joblib").exists(),
            "promise": (self.models_dir / "eta.joblib").exists(),
        }


def create_app(state: State | None = None) -> FastAPI:
    st = state or State()
    app = FastAPI(
        title="OnTime late-delivery risk API",
        version=API_VERSION,
        description="Scores orders for late-delivery risk and explains why. See /docs.",
    )

    def _frame(payload: FeaturePayload, columns: list[str], levels: dict):
        try:
            return to_frame(payload.features, columns, levels)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    def _need(name: str, path: str) -> None:
        if not st.available()[name]:
            raise HTTPException(status_code=503, detail=f"artifact missing: {path} (run make all)")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": API_VERSION, "artifacts": st.available()}

    @app.get("/model-info")
    def model_info() -> dict:
        _need("stage1", "models/logreg.joblib")
        res = st.results
        art = joblib.load(st.models_dir / f"{st.primary}.joblib")
        out: dict[str, Any] = {
            "primary_model": st.primary,
            "primary_selection_rule": res.get("primary_selection_rule"),
            "train_range": art["meta"].get("train_range"),
            "git_hash": art["meta"].get("git_hash"),
            "calibration": art.get("method"),
            "stage1_features": FEATURE_COLUMNS,
            "stage2_extra_features": STAGE2_FEATURES,
            "training_late_rate": st.base_rate,
            "test_period": {k: res["population"][k] for k in ("from", "to", "late_rate")},
            "test_metrics_primary": res["models"][st.primary].get("test", None)
            or res["models"][st.primary],
            "disclosure": "Stage 1 was scored on the test months once; stage 2 and the promise "
            "model are second/third looks on the same months (see model card).",
        }
        return out

    @app.post("/score", response_model=ScoreResponse)
    def score(payload: FeaturePayload) -> ScoreResponse:
        _need("stage1", "models/logreg.joblib")
        model = payload.model or st.primary
        if model not in ("logreg", "lightgbm", "ebm"):
            raise HTTPException(status_code=422, detail=f"unknown model {model!r}")
        svc = st.service
        row, warns = _frame(payload, FEATURE_COLUMNS, svc.levels)
        out = svc.explain_one(row, model)
        method = svc.arts[model].get("method", "none")
        return ScoreResponse(
            stage=1, model=model, probability=out["probability"], raw_probability=out["raw"],
            calibration=method, risk_ratio=out["probability"] / st.base_rate,
            risk_factors=out["risk_factors"], protective_factors=out["protective_factors"],
            warnings=warns,
        )  # fmt: skip

    @app.post("/score/handover", response_model=ScoreResponse)
    def score_handover(payload: FeaturePayload) -> ScoreResponse:
        _need("stage2", "models/stage2.joblib")
        from ontime.stage2.explain import ALL_COLUMNS, explain_rows

        art = st.stage2
        row, warns = _frame(payload, ALL_COLUMNS, art["levels"])
        out = explain_rows(art, row)
        from ontime.models.data import prepare_frame

        X = prepare_frame(row, art["levels"])
        for f in STAGE2_FEATURES:
            X[f] = row[f].to_numpy()
        raw = float(art["model"].predict_proba(X[art["columns"]])[:, 1][0])
        prob = float(np.asarray(art["calibrator"].transform(np.array([raw])))[0])
        codes = out["reasons"].iloc[0]
        return ScoreResponse(
            stage=2, model=art["kind"], probability=prob, raw_probability=raw,
            calibration=art["method"], risk_ratio=prob / st.base_rate,
            risk_factors=list(codes["risk_factors"]),
            protective_factors=list(codes["protective_factors"]), warnings=warns,
        )  # fmt: skip

    @app.post("/promise", response_model=PromiseResponse)
    def promise(payload: FeaturePayload) -> PromiseResponse:
        _need("promise", "models/eta.joblib")
        from ontime.eta.quantile import promise_days
        from ontime.models.data import prepare_frame

        art = st.eta
        row, _ = _frame(payload, FEATURE_COLUMNS, art["levels"])
        X = prepare_frame(row, art["levels"])
        by_level = {
            str(a): int(promise_days(m.predict(X))[0]) for a, m in sorted(art["models"].items())
        }
        rec = {name: by_level[str(a)] for name, a in art["chosen"].items()}
        return PromiseResponse(
            quoted_promise_days=float(row["promised_days"].iloc[0]),
            promise_days_by_level=by_level, recommended=rec,
            note="Whole days from approval; higher level = safer (longer) promise. Operating "
            "points were chosen on validation. A shorter promise is not free: demand effects "
            "are not modelled.",
        )  # fmt: skip

    @app.get("/demo/orders/{order_id}")
    def demo_order(order_id: str, stage: int = 1) -> dict:
        if stage not in (1, 2):
            raise HTTPException(status_code=422, detail="stage must be 1 or 2")
        path = project_path(
            st.cfg["paths"]["features_parquet"]
            if stage == 1
            else st.cfg["paths"]["processed_dir"] + "/features_stage2.parquet"
        )
        if not path.exists():
            raise HTTPException(status_code=404, detail="feature table not available in this image")
        t = pd.read_parquet(path)
        hit = t[t["order_id"] == order_id]
        if hit.empty:
            raise HTTPException(status_code=404, detail="unknown order_id")
        r = hit.iloc[0]
        cols = FEATURE_COLUMNS if stage == 1 else FEATURE_COLUMNS + STAGE2_FEATURES
        feats = {c: (None if pd.isna(r[c]) else r[c]) for c in cols}
        feats = {k: (v.item() if hasattr(v, "item") else v) for k, v in feats.items()}
        pl = FeaturePayload(features=feats)
        resp = score(pl) if stage == 1 else score_handover(pl)
        return {"order_id": order_id, "split": str(r["split"]), **resp.model_dump()}

    return app


app = create_app()
