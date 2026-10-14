"""`make monitor`: drift tables for the app and the report. Reads only generated artifacts."""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from ontime.config import load_config, project_path
from ontime.models.data import load_feature_table, prepare_frame
from ontime.monitoring.drift import month_status, monthly_psi, performance_by_month, status_from_psi

log = logging.getLogger(__name__)


def run_monitor(write: bool = True) -> dict:
    cfg = load_config()
    mcfg = cfg.get(
        "monitoring",
        {"psi_bins": 10, "psi_watch": 0.10, "psi_alert": 0.25, "min_month_orders": 500},
    )
    rep = project_path(cfg["paths"]["reports_dir"])
    table = load_feature_table()

    # Performance monitor uses the DEPLOYED model's raw ranking score (rank is what ops uses).
    art = joblib.load(project_path(cfg["paths"]["models_dir"]) / "logreg.joblib")
    X = prepare_frame(table, art["levels"])
    scored = table[["approved_at", "is_late", "split"]].copy()
    scored["score"] = art["model"].predict_proba(X)[:, 1]
    first_eval = pd.Timestamp(cfg["splits"]["val_start"])
    perf = performance_by_month(scored[scored["approved_at"] >= first_eval], seed=cfg["seed"])
    perf["split"] = perf["month"].map(
        lambda m: "test" if m >= pd.Timestamp(cfg["splits"]["test_start"]) else "validation"
    )

    kw = {"min_orders": mcfg["min_month_orders"], "bins": mcfg["psi_bins"]}
    vs_train = monthly_psi(table, reference_mask=table["split"] == "train", **kw)
    vs_train["baseline"] = "training"
    vs_trail = monthly_psi(
        table,
        trailing_months=mcfg["trailing_months"],
        first_month=vs_train["month"].min(),
        **kw,
    )
    vs_trail["baseline"] = "trailing"
    psi = pd.concat([vs_train, vs_trail], ignore_index=True)
    expected = tuple(mcfg["expected_drift_features"])
    psi["expected_drift"] = psi["feature"].isin(expected)
    psi["status"] = psi["psi"].map(
        lambda v: status_from_psi(v, mcfg["psi_watch"], mcfg["psi_alert"])
    )
    status = month_status(psi, perf, mcfg["psi_watch"], mcfg["psi_alert"], exclude=expected)

    # which features drifted most in the test period; does drift relate to decay? (descriptive)
    test_months = status["month"] >= pd.Timestamp(cfg["splits"]["test_start"])
    in_test = psi["month"] >= pd.Timestamp(cfg["splits"]["test_start"])
    top = (
        psi[in_test & (psi["baseline"] == "training") & ~psi["expected_drift"]]
        .groupby("feature")["psi"]
        .mean()
        .sort_values(ascending=False)
    )
    joint = status.dropna(subset=["roc_auc"])
    corr = (
        float(joint["trail_max"].corr(joint["roc_auc"], method="spearman"))
        if len(joint) >= 4 and joint["trail_max"].notna().sum() >= 4
        else None
    )
    summary = {
        "reference": "training period",
        "thresholds": {"watch": mcfg["psi_watch"], "alert": mcfg["psi_alert"]},
        "months_monitored": [str(m.date()) for m in status["month"]],
        "latest_month": str(status["month"].iloc[-1].date()),
        "latest_status": str(status["status"].iloc[-1]),
        "top_drifting_features_test_period": {k: float(v) for k, v in top.head(8).items()},
        "n_features_alert_latest_trailing": int(status["trail_alert"].fillna(0).iloc[-1]),
        "n_features_alert_latest_vs_training": int(status["train_alert"].fillna(0).iloc[-1]),
        "expected_drift_features": list(expected),
        "spearman_trailing_max_psi_vs_roc_auc": corr,
        "n_months_in_correlation": int(len(joint)),
        "test_months_status": status.loc[test_months, ["month", "status"]]
        .assign(month=lambda d: d["month"].dt.strftime("%Y-%m"))
        .to_dict(orient="records"),
    }
    if write:
        psi.to_csv(rep / "drift_psi.csv", index=False)
        perf.to_csv(rep / "drift_performance.csv", index=False)
        status.to_csv(rep / "drift_status.csv", index=False)
        (rep / "drift_summary.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")
        log.info("monitor | wrote drift_* files (%d month x feature rows)", len(psi))
    return summary
