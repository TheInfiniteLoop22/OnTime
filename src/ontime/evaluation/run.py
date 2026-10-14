"""Phase 7: final evaluation. Test is opened here and only here (`final_evaluation=True`).

`run_evaluation(dry_run=True)` executes the identical code path on the VALIDATION split and
writes under reports/dryrun/, so the pipeline is debugged without ever reading test rows.
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from ontime.config import load_config, project_path
from ontime.evaluation import plots
from ontime.evaluation.calibration import expected_calibration_error, rolling_recalibrate
from ontime.evaluation.metrics import (
    classification_metrics,
    evaluate_scores,
    paired_bootstrap_pr_auc_diff,
    topk_metrics,
)
from ontime.evaluation.policy import simulate_policy
from ontime.evaluation.report import render_results_md
from ontime.evaluation.slices import add_slice_columns, slice_table
from ontime.models.baselines import PriorRateBaseline, RuleBaseline
from ontime.models.data import load_feature_table, prepare_frame
from ontime.models.run import MODEL_ORDER
from ontime.models.splits import select_split

log = logging.getLogger(__name__)

SLICE_COLUMNS = [
    "customer_region",
    "seller_region",
    "distance_bucket",
    "category_group",
    "seller_tenure",
    "route_type",
    "payment_type",
    "approval_month",
]
ERROR_COLS = [
    "order_id", "approved_at", "is_late", "customer_state", "seller_state", "distance_km",
    "promised_days", "category_group", "seller_n_past", "seller_late_rate_shrunk",
    "global_late_rate_30d", "delivery_days",
]  # fmt: skip


def primary_model_by_validation(results_path) -> tuple[str, dict]:  # noqa: ANN001
    """Pre-registered rule: best HONEST validation PR-AUC (val_calib) among the four models."""
    v = json.loads(results_path.read_text(encoding="utf-8"))
    scores = {m: v["models"][m]["raw_metrics"]["val_calib"]["pr_auc"] for m in MODEL_ORDER}
    return max(scores, key=scores.get), scores


def _score_frame(artifact: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = prepare_frame(df, artifact["levels"])
    raw = np.asarray(artifact["model"].predict_proba(X)[:, 1], dtype=float)
    return raw, np.asarray(artifact["calibrator"].transform(raw), dtype=float)


def run_evaluation(dry_run: bool = False, write: bool = True) -> dict:
    cfg = load_config()
    seed, n_boot = cfg["seed"], cfg["tuning"]["bootstrap_resamples"]
    k_grid, p_grid = cfg["policy"]["capacity_grid"], cfg["policy"]["intervention_success_grid"]
    split = "validation" if dry_run else "test"
    tag = "dryrun" if dry_run else "test"
    base_dir = project_path(cfg["paths"]["reports_dir"]) / ("dryrun" if dry_run else "")
    fig_dir = base_dir / "figures"
    base_dir.mkdir(parents=True, exist_ok=True)
    log.info("evaluate | split=%s (dry_run=%s)", split, dry_run)

    table = load_feature_table()
    final = not dry_run
    train = select_split(table, "train")
    ev = select_split(table, split, final_evaluation=final).reset_index(drop=True)
    allowed = ["train", "validation"] + (["test"] if final else [])
    pool = pd.concat(
        [select_split(table, s, final_evaluation=final) for s in allowed], ignore_index=True
    )
    orders = pd.read_parquet(
        project_path(cfg["paths"]["orders_parquet"]),
        columns=["order_id", "order_delivered_customer_date"],
    )
    pool = pool.merge(orders, on="order_id", how="left").rename(
        columns={"order_delivered_customer_date": "delivered_at"}
    )

    primary, val_scores = primary_model_by_validation(
        project_path(cfg["paths"]["reports_dir"]) / "validation_results.json"
    )
    log.info("evaluate | primary model (pre-registered rule) = %s", primary)

    # ---- scores ------------------------------------------------------------------------
    arts = {
        m: joblib.load(project_path(cfg["paths"]["models_dir"]) / f"{m}.joblib")
        for m in MODEL_ORDER
    }
    y = ev["is_late"].to_numpy()
    raw, prob, roll = {}, {}, {}
    pool_raw = {}
    for m in MODEL_ORDER:
        raw[m], prob[m] = _score_frame(arts[m], ev)
        pool_raw[m], _ = _score_frame(arts[m], pool)
        sc = pd.DataFrame(
            {
                "approved_at": pool["approved_at"],
                "delivered_at": pool["delivered_at"],
                "y": pool["is_late"],
                "raw": pool_raw[m],
            }
        )
        r = rolling_recalibrate(
            sc, str(ev["approved_at"].min().date()), str(ev["approved_at"].max().date())
        )
        by_order = pd.Series(r.to_numpy(), index=pool["order_id"].to_numpy())
        roll[m] = by_order.reindex(ev["order_id"].to_numpy()).to_numpy()  # aligned to ev rows
    prior = PriorRateBaseline().fit(train["is_late"].to_numpy())
    rule = RuleBaseline().fit(train)
    base_prob = {"prior_rate": prior.predict_proba(ev), "rule": rule.predict_proba(ev)}

    # ---- headline metrics with CIs -----------------------------------------------------
    res: dict = {
        "evaluated_on": split,
        "dry_run": dry_run,
        "primary_model": primary,
        "primary_selection_rule": (
            "Pre-registered before test was opened: highest PR-AUC on val_calib "
            "(validation data never used for tuning)."
        ),
        "validation_pr_auc_used_for_selection": val_scores,
        "population": {
            "orders": int(len(ev)),
            "late_orders": int(y.sum()),
            "late_rate": float(y.mean()),
            "from": str(ev["approved_at"].min().date()),
            "to": str(ev["approved_at"].max().date()),
            "train_late_rate": float(train["is_late"].mean()),
        },
        "models": {},
        "baselines": {},
    }
    dates = ev["approved_at"]
    for m in MODEL_ORDER:
        e = evaluate_scores(dates, y, prob[m], k_grid, n_boot, seed)
        e["calibrator"] = arts[m]["method"]
        e["ece_deployed"] = expected_calibration_error(y, prob[m])
        ok = ~np.isnan(roll[m])
        e["rolling_recalibration"] = {
            **{
                k: v
                for k, v in classification_metrics(y[ok], roll[m][ok]).items()
                if k in ("brier", "log_loss")
            },
            "ece": expected_calibration_error(y[ok], roll[m][ok]),
            "orders": int(ok.sum()),
        }
        res["models"][m] = e
    for b, p in base_prob.items():
        e = evaluate_scores(dates, y, p, k_grid, n_boot, seed)
        e["ece_deployed"] = expected_calibration_error(y, p) if b == "rule" else None
        res["baselines"][b] = e
    res["baselines"]["rule"]["description"] = rule.describe()
    res["constant_oracle_brier"] = float(y.mean() * (1 - y.mean()))

    # ---- model-vs-model gaps (paired bootstrap) --------------------------------------
    pairs = [("ebm", "lightgbm"), ("logreg", "lightgbm"), ("ebm", "logreg"), (primary, "rule")]
    res["paired_pr_auc_diff"] = {}
    for a, b in pairs:
        pa = prob[a]
        pb = prob[b] if b in prob else base_prob[b]
        res["paired_pr_auc_diff"][f"{a}_minus_{b}"] = paired_bootstrap_pr_auc_diff(
            y, pa, pb, n_boot, seed
        )

    # ---- monthly performance -----------------------------------------------------------
    ev["month"] = ev["approved_at"].dt.strftime("%Y-%m")
    rows = []
    for m in MODEL_ORDER:
        for mon, idx in ev.groupby("month").indices.items():
            ym, pm = y[idx], prob[m][idx]
            met = classification_metrics(ym, pm)
            ok = ~np.isnan(roll[m][idx])
            tk = topk_metrics(dates.iloc[idx], ym, pm, 0.10, seed)
            rows.append({
                "model": m, "month": mon, "orders": len(idx), "late_orders": int(ym.sum()),
                "late_rate": met["base_rate"], "pr_auc": met["pr_auc"], "roc_auc": met["roc_auc"],
                "lift_at_10": tk["lift"], "brier_deployed": met["brier"],
                "brier_rolling": float(np.mean((roll[m][idx][ok] - ym[ok]) ** 2)) if ok.any() else None,
                "mean_predicted_deployed": float(pm.mean()),
            })  # fmt: skip
    monthly = pd.DataFrame(rows)
    res["monthly"] = monthly.to_dict(orient="records")

    # ---- slices ------------------------------------------------------------------------
    ev_s = add_slice_columns(ev)
    slice_frames = []
    for m in MODEL_ORDER:
        t = slice_table(ev_s, prob[m], SLICE_COLUMNS)
        t.insert(0, "model", m)
        slice_frames.append(t)
    slices = pd.concat(slice_frames, ignore_index=True)
    slices.to_csv(base_dir / "slices.csv", index=False)
    res["slices_primary_weak"] = slices[(slices["model"] == primary) & slices["weak"]].to_dict(
        orient="records"
    )
    res["slices_file"] = "slices.csv"

    # ---- policy simulation -------------------------------------------------------------
    policy_scores = {m: prob[m] for m in MODEL_ORDER} | {"rule": base_prob["rule"]}
    policy = simulate_policy(dates, y, policy_scores, k_grid, p_grid, seed, n_boot=min(n_boot, 500))
    policy.to_csv(base_dir / "policy_simulation.csv", index=False)
    res["policy"] = policy.to_dict(orient="records")
    impact_path = project_path(cfg["paths"]["reports_dir"]) / "impact.json"
    if impact_path.exists():
        imp = json.loads(impact_path.read_text(encoding="utf-8"))
        res["star_gap_assumptions"] = {
            "full_adjusted_gap_stars": abs(imp["ols"]["adjusted"]["coef"]),
            "conservative_gap_stars": abs(imp["sensitivity_answered_after_delivery"]["coef"]),
            "note": "Illustrative: stars recovered = prevented late orders x gap. Association, not causal.",
        }

    # ---- error analysis exports --------------------------------------------------------
    err_files = []
    for m in MODEL_ORDER:
        d = ev_s.assign(p=prob[m])
        fp = d[d["is_late"] == 0].nlargest(20, "p").assign(error="false_positive")
        fn = d[d["is_late"] == 1].nsmallest(20, "p").assign(error="false_negative")
        out = pd.concat([fp, fn])[["error", "p", *ERROR_COLS]]
        name = f"error_analysis_{m}.csv"
        out.to_csv(base_dir / name, index=False)
        err_files.append(name)
    res["error_analysis_files"] = err_files

    # ---- figures -----------------------------------------------------------------------
    figs = [
        plots.pr_curves(y, prob | {"rule": base_prob["rule"]}, fig_dir, tag),
        plots.calibration_plot(y, prob, (primary, prob[primary], roll[primary]), fig_dir, tag),
        plots.gains_and_lift(y, prob, fig_dir, tag),
        plots.monthly_pr_auc(monthly, fig_dir, tag),
        plots.policy_curves(policy, p_grid, fig_dir, tag),
    ]
    res["figures"] = figs

    if write:
        (base_dir / "results.json").write_text(
            json.dumps(res, indent=2, default=float), encoding="utf-8"
        )
        (base_dir / "results.md").write_text(render_results_md(res), encoding="utf-8")
        log.info("evaluate | wrote %s and results.md", base_dir / "results.json")
    return res
