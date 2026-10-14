"""Stage 2 train + evaluate: re-score each order at carrier handover.

PROTOCOL (fixed before the test period was scored by this model):
  * population  : stage-1 modelling population with a valid handover timestamp;
  * splits      : identical to stage 1 (by approval time) so every order is in the same split;
  * models      : logistic regression (C by validation PR-AUC) and LightGBM (Optuna on validation);
  * selection   : the better of the two by VALIDATION PR-AUC is the stage-2 model (pre-registered);
  * test        : only the selected model is scored on test, once, next to the deployed stage-1
                  model on the SAME orders (paired bootstrap for the difference);
  * ranking only: no calibration is fitted at this stage (use rank/bands).

DISCLOSURE: the test months were already used once to evaluate the stage-1 models. This is a second,
separately pre-declared model evaluated on the same months; it does not change any stage-1 result,
but its test numbers are not a first-look hold-out.
"""

from __future__ import annotations

import json
import logging

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ontime.config import load_config, project_path  # noqa: E402
from ontime.data.build_orders import build_all_orders  # noqa: E402
from ontime.data.load import load_all  # noqa: E402
from ontime.evaluation.metrics import (  # noqa: E402
    daily_topk_table,
    evaluate_scores,
    paired_bootstrap_pr_auc_diff,
    topk_metrics,
)
from ontime.features.build import FEATURE_COLUMNS  # noqa: E402
from ontime.models.data import category_levels, prepare_frame  # noqa: E402
from ontime.models.rolling import paired_day_bootstrap_lift  # noqa: E402
from ontime.models.splits import select_split  # noqa: E402
from ontime.models.train import fit_lgbm, fit_logreg  # noqa: E402
from ontime.models.tune import tune_lgbm  # noqa: E402
from ontime.stage2.features import STAGE2_FEATURES, make_events, stage2_features  # noqa: E402

log = logging.getLogger(__name__)
C_GRID = [0.01, 0.1, 1.0, 10.0]


def build_stage2_table(write: bool = True) -> pd.DataFrame:
    cfg = load_config()
    tables = load_all()
    orders_all = build_all_orders(tables)
    carrier = tables["orders"].set_index("order_id")["order_delivered_carrier_date"]
    events = make_events(orders_all, carrier)
    s1 = pd.read_parquet(project_path(cfg["paths"]["features_parquet"]))
    extra = orders_all[
        [
            "order_id",
            "max_shipping_limit_date",
            "order_estimated_delivery_date",
            "order_approved_at",
        ]
    ]
    q = s1.merge(extra, on="order_id", how="left")
    q["handover_at"] = q["order_id"].map(carrier)
    q = q[q["order_id"].isin(set(events["order_id"]))].reset_index(drop=True)  # valid handover only
    s2 = stage2_features(q, events, alpha=cfg["features"]["shrinkage_alpha"])
    table = pd.concat([q[list(s1.columns)], s2], axis=1)
    log.info("stage2 | population %d of %d stage-1 orders (valid handover)", len(table), len(s1))
    if write:
        out = project_path(cfg["paths"]["processed_dir"]) / "features_stage2.parquet"
        table.to_parquet(out, index=False)
    return table


def _frame(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    X = prepare_frame(df, levels)
    for f in STAGE2_FEATURES:
        X[f] = df[f].to_numpy()
    return X


def run_stage2(write: bool = True) -> dict:
    cfg = load_config()
    seed, n_boot = cfg["seed"], cfg["tuning"]["bootstrap_resamples"]
    k_grid = cfg["policy"]["capacity_grid"]
    table = build_stage2_table(write=write)
    levels = category_levels(pd.read_parquet(project_path(cfg["paths"]["features_parquet"])))
    cols = FEATURE_COLUMNS + STAGE2_FEATURES

    train = select_split(table, "train")
    val = select_split(table, "validation")
    Xtr, ytr = _frame(train, levels), train["is_late"].to_numpy()
    Xva, yva = _frame(val, levels), val["is_late"].to_numpy()

    # ---- stage-1 comparator: the deployed v1 logistic regression, scored on the same orders
    art = joblib.load(project_path(cfg["paths"]["models_dir"]) / "logreg.joblib")

    def s1_score(df: pd.DataFrame) -> np.ndarray:
        return art["model"].predict_proba(prepare_frame(df, art["levels"]))[:, 1]

    # ---- candidates, tuned on validation only
    cand: dict = {}
    best_c, best_ap = C_GRID[0], -1.0
    for c in C_GRID:
        m = fit_logreg(Xtr, ytr, {"C": c, "class_weight": "none"}, seed, cols)
        ap = average_precision_score(yva, m.predict_proba(Xva[cols])[:, 1])
        if ap > best_ap:
            best_c, best_ap, best_lr = c, ap, m
    cand["logreg"] = {"model": best_lr, "val_pr_auc": float(best_ap), "params": {"C": best_c}}

    params, study = tune_lgbm(Xtr[cols], ytr, Xva[cols], yva, 25, seed,
                              cfg["tuning"]["lgbm_early_stopping_rounds"])  # fmt: skip
    lgb = fit_lgbm(
        Xtr[cols], ytr, params, seed, Xva[cols], yva, cfg["tuning"]["lgbm_early_stopping_rounds"]
    )
    cand["lightgbm"] = {"model": lgb, "val_pr_auc": float(study.best_value), "params": params}

    chosen = max(cand, key=lambda k: cand[k]["val_pr_auc"])
    model = cand[chosen]["model"]
    log.info("stage2 | validation PR-AUC: %s -> chosen = %s",
             {k: round(v["val_pr_auc"], 4) for k, v in cand.items()}, chosen)  # fmt: skip

    def s2_score(df: pd.DataFrame) -> np.ndarray:
        X = _frame(df, levels)
        return model.predict_proba(X[cols])[:, 1]

    # ---- validation comparison (development numbers)
    val_cmp = {
        "orders": int(len(val)),
        "late_rate": float(yva.mean()),
        "stage1_pr_auc": float(average_precision_score(yva, s1_score(val))),
        "stage2_pr_auc": float(average_precision_score(yva, s2_score(val))),
        "stage1_roc_auc": float(roc_auc_score(yva, s1_score(val))),
        "stage2_roc_auc": float(roc_auc_score(yva, s2_score(val))),
    }

    # ---- test: scored once (disclosed second look)
    test = select_split(table, "test", final_evaluation=True).reset_index(drop=True)
    y = test["is_late"].to_numpy()
    p1, p2 = s1_score(test), s2_score(test)
    dates = test["t1"]
    ev2 = evaluate_scores(dates, y, p2, k_grid, n_boot, seed)
    ev1 = evaluate_scores(dates, y, p1, k_grid, n_boot, seed)
    pair = paired_bootstrap_pr_auc_diff(y, p2, p1, n_boot, seed)
    t1_tbl = daily_topk_table(dates, y, p1, 0.10, seed)
    t2_tbl = daily_topk_table(dates, y, p2, 0.10, seed)
    lift_d, lift_ci = paired_day_bootstrap_lift(t2_tbl, t1_tbl, n_boot, seed)

    test = test.assign(month=test["approved_at"].dt.strftime("%Y-%m"))
    monthly = []
    for mon, g in test.groupby("month"):
        yg, i = g["is_late"].to_numpy(), g.index.to_numpy()
        monthly.append({
            "month": mon, "orders": len(g), "late_rate": float(yg.mean()),
            "stage1_roc_auc": float(roc_auc_score(yg, p1[i])), "stage2_roc_auc": float(roc_auc_score(yg, p2[i])),
            "stage1_lift_at_10": topk_metrics(g["t1"], yg, p1[i], 0.10, seed)["lift"],
            "stage2_lift_at_10": topk_metrics(g["t1"], yg, p2[i], 0.10, seed)["lift"],
        })  # fmt: skip

    late = test[test["is_late"] == 1]
    window = {
        "late_orders": int(len(late)),
        "median_days_left_at_handover": float(late["promise_left_at_handover_days"].median()),
        "share_with_3plus_days_left": float((late["promise_left_at_handover_days"] >= 3).mean()),
        "share_with_7plus_days_left": float((late["promise_left_at_handover_days"] >= 7).mean()),
        "median_days_to_handover": float(test["days_to_handover"].median()),
    }

    # univariate value of the new signals on validation (development evidence)
    uni = {f: float(roc_auc_score(yva, val[f].fillna(val[f].median()))) for f in STAGE2_FEATURES}

    result = {
        "protocol": "Pre-registered: select by validation PR-AUC, score on test once, same orders as stage 1.",
        "disclosure": "Test months were already used once for stage-1 evaluation; this is a second, "
        "separately declared model on the same months. Stage-1 results are unchanged.",  # fmt: skip
        "population": {
            "orders_total": int(len(table)),
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },  # fmt: skip
        "chosen_model": chosen,
        "validation_pr_auc_by_candidate": {k: v["val_pr_auc"] for k, v in cand.items()},
        "chosen_params": cand[chosen]["params"],
        "validation_comparison": val_cmp,
        "univariate_validation_auc_of_new_features": uni,
        "test": {
            "late_rate": float(y.mean()),
            "stage1": ev1,
            "stage2": ev2,
            "pr_auc_diff_stage2_minus_stage1": pair,
            "lift_at_10_diff_stage2_minus_stage1": {"diff": lift_d, "ci95": list(lift_ci)},
            "monthly": monthly,
            "action_window_for_late_orders": window,
        },
        "ranking_only": True,
    }
    if write:
        rep = project_path(cfg["paths"]["reports_dir"])
        (rep / "stage2_results.json").write_text(
            json.dumps(result, indent=2, default=float), "utf-8"
        )
        _plot(result, project_path(cfg["paths"]["figures_dir"]) / "stage2_vs_stage1.png")
        log.info("stage2 | wrote stage2_results.json")
    return result


def _plot(r: dict, path) -> None:  # noqa: ANN001
    t = r["test"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    names = ["Stage 1\n(at approval)", "Stage 2\n(at handover)"]
    vals = [t["stage1"]["metrics"]["pr_auc"]["value"], t["stage2"]["metrics"]["pr_auc"]["value"]]
    ci = [t["stage1"]["metrics"]["pr_auc"]["ci95"], t["stage2"]["metrics"]["pr_auc"]["ci95"]]
    err = [
        [v - c[0] for v, c in zip(vals, ci, strict=True)],
        [c[1] - v for v, c in zip(vals, ci, strict=True)],
    ]
    ax[0].bar(names, vals, yerr=err, color=["#4C78A8", "#EB6834"], capsize=6)
    ax[0].axhline(t["late_rate"], color="grey", ls="--")
    ax[0].set_title("Test PR-AUC (95% CI); dashed = base rate")
    m = pd.DataFrame(t["monthly"])
    ax[1].plot(m["month"], m["stage1_lift_at_10"], "o-", color="#4C78A8", label="Stage 1")
    ax[1].plot(m["month"], m["stage2_lift_at_10"], "o-", color="#EB6834", label="Stage 2")
    ax[1].axhline(1.0, color="grey", lw=1)
    ax[1].set_title("Lift at 10% daily capacity by month")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
