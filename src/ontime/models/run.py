"""Phase 6 orchestration: tune -> fit -> calibrate -> evaluate on validation -> save -> verify.

Visible data: train and validation only. The test split is never loaded here (CLAUDE.md rule 4).
"""

from __future__ import annotations

import json
import logging
import subprocess
import warnings
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from ontime.config import load_config, project_path
from ontime.evaluation.metrics import (
    classification_metrics,
    evaluate_scores,
    paired_bootstrap_pr_auc_diff,
)
from ontime.features.build import FEATURE_COLUMNS
from ontime.models.baselines import PriorRateBaseline, RuleBaseline
from ontime.models.calibrate import CALIBRATORS, choose_calibrator
from ontime.models.data import category_levels, load_feature_table, modelling_frames, prepare_frame
from ontime.models.train import fit_ebm, fit_lgbm, fit_logreg, fit_random_forest
from ontime.models.tune import tune_ebm, tune_lgbm, tune_logreg, weight_experiment

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*Missing values detected.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
logging.getLogger("interpret").setLevel(logging.WARNING)  # EBM logs allocation noise at INFO

MODEL_ORDER = ["logreg", "random_forest", "lightgbm", "ebm"]
SUBSETS = ["train", "val_tune", "val_calib", "calib_fit", "calib_eval", "validation"]


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=project_path("."), text=True
        ).strip()
    except Exception:  # noqa: BLE001 - metadata only; never fail training over it
        return "unknown"


def _predict(model, X: pd.DataFrame) -> np.ndarray:  # noqa: ANN001
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def run_training(write: bool = True) -> dict:
    cfg = load_config()
    seed, tcfg = cfg["seed"], cfg["tuning"]
    table = load_feature_table()
    levels = category_levels(table)
    frames = modelling_frames(table, cfg)
    X = {k: prepare_frame(v, levels) for k, v in frames.items()}
    y = {k: v["is_late"].to_numpy() for k, v in frames.items()}
    for k in SUBSETS:
        log.info("train | %-10s n=%6d late=%5d rate=%.3f", k, len(y[k]), y[k].sum(), y[k].mean())

    # ---- 1. tune (sees only train + val_tune) ------------------------------------------
    es = tcfg["lgbm_early_stopping_rounds"]
    lr_p, lr_s = tune_logreg(
        X["train"], y["train"], X["val_tune"], y["val_tune"], tcfg["optuna_trials_logreg"], seed
    )
    lg_p, lg_s = tune_lgbm(
        X["train"], y["train"], X["val_tune"], y["val_tune"], tcfg["optuna_trials"], seed, es
    )
    eb_p, eb_s = tune_ebm(
        X["train"], y["train"], frames["train"]["approved_at"], X["val_tune"], y["val_tune"],
        tcfg["optuna_trials_ebm"], seed, tcfg["ebm_early_stop_fraction"],
    )  # fmt: skip
    best = {"logreg": lr_p, "lightgbm": lg_p, "ebm": eb_p}

    # ---- 2. final fits on TRAIN only ---------------------------------------------------
    models = {
        "logreg": fit_logreg(X["train"], y["train"], lr_p, seed),
        "random_forest": fit_random_forest(X["train"], y["train"], seed),
        "lightgbm": fit_lgbm(X["train"], y["train"], lg_p, seed, X["val_tune"], y["val_tune"], es),
        "ebm": fit_ebm(
            X["train"], y["train"], frames["train"]["approved_at"], eb_p, seed,
            tcfg["ebm_early_stop_fraction"],
        ),
    }  # fmt: skip

    # ---- 3. predict, calibrate, evaluate ----------------------------------------------
    raw = {m: {s: _predict(models[m], X[s]) for s in SUBSETS} for m in MODEL_ORDER}
    results: dict = {"models": {}, "baselines": {}, "sub_splits": {}}
    for s in SUBSETS:
        f = frames[s]
        results["sub_splits"][s] = {
            "orders": int(len(f)),
            "late_rate": float(f["is_late"].mean()),
            "from": str(f["approved_at"].min().date()),
            "to": str(f["approved_at"].max().date()),
        }

    k_grid, n_boot = cfg["policy"]["capacity_grid"], tcfg["bootstrap_resamples"]
    artifacts: dict = {}
    for m in MODEL_ORDER:
        method, cal_table = choose_calibrator(
            raw[m]["calib_fit"], y["calib_fit"], raw[m]["calib_eval"], y["calib_eval"]
        )
        # honest calibrated predictions on calib_eval: calibrator fit on calib_fit only
        honest = (
            CALIBRATORS[method]()
            .fit(raw[m]["calib_fit"], y["calib_fit"])
            .transform(raw[m]["calib_eval"])
        )
        # deployed calibrator: refit on ALL of val_calib (more data, still strictly after tuning)
        calibrator = CALIBRATORS[method]().fit(raw[m]["val_calib"], y["val_calib"])
        results["models"][m] = {
            "best_params": best.get(m, {"n_estimators": 300, "min_samples_leaf": 50}),
            "raw_metrics": {s: classification_metrics(y[s], raw[m][s]) for s in SUBSETS},
            "val_calib_with_ci": evaluate_scores(
                frames["val_calib"]["approved_at"],
                y["val_calib"],
                raw[m]["val_calib"],
                k_grid,
                n_boot,
                seed,
            ),
            "calibration": {
                "chosen": method,
                "scores_on_calib_eval": cal_table,
                "calib_eval_raw": classification_metrics(y["calib_eval"], raw[m]["calib_eval"]),
                "calib_eval_calibrated": classification_metrics(y["calib_eval"], honest),
            },
        }
        artifacts[m] = {"model": models[m], "calibrator": calibrator, "method": method}
    results["class_weight_experiment"] = {
        "logreg": weight_experiment(lr_s),
        "lightgbm": weight_experiment(lg_s),
    }
    results["paired_pr_auc_diff_val_calib"] = {
        f"{a}_minus_{b}": paired_bootstrap_pr_auc_diff(
            y["val_calib"], raw[a]["val_calib"], raw[b]["val_calib"], n_boot, seed
        )
        for a, b in [("ebm", "lightgbm"), ("logreg", "lightgbm"), ("ebm", "logreg")]
    }

    # baselines scored on the same sub-splits, for a like-for-like table
    prior = PriorRateBaseline().fit(y["train"])
    rule = RuleBaseline().fit(frames["train"])
    for name, mdl in {"prior_rate": prior, "rule": rule}.items():
        results["baselines"][name] = {
            s: classification_metrics(y[s], mdl.predict_proba(frames[s])) for s in SUBSETS
        }

    # ---- 4. persist + verify reload ---------------------------------------------------
    meta = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_hash": _git_hash(),
        "seed": seed,
        "features": FEATURE_COLUMNS,
        "train_range": [
            str(frames["train"]["approved_at"].min()),
            str(frames["train"]["approved_at"].max()),
        ],
        "train_orders": int(len(frames["train"])),
    }
    models_dir = project_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    for m in MODEL_ORDER:
        art = {
            **artifacts[m],
            "levels": levels,
            "meta": {**meta, "name": m, "params": results["models"][m]["best_params"]},
        }
        joblib.dump(art, models_dir / f"{m}.joblib")
        reloaded = joblib.load(models_dir / f"{m}.joblib")
        again = _predict(reloaded["model"], prepare_frame(frames["val_calib"], reloaded["levels"]))
        diff = float(np.max(np.abs(again - raw[m]["val_calib"])))
        results["models"][m]["reload_max_abs_diff"] = diff
        if diff > 1e-9:
            raise RuntimeError(f"{m}: reloaded model does not reproduce validation scores ({diff})")
    results["meta"] = meta

    table_md = comparison_table(results)
    log.info("\n%s", table_md)
    if write:
        rep = project_path(cfg["paths"]["reports_dir"])
        (rep / "validation_results.json").write_text(
            json.dumps(results, indent=2, default=float), encoding="utf-8"
        )
        (rep / "model_comparison.md").write_text(
            "# Model comparison on validation (generated by `ontime.models.run`)\n\n"
            + table_md
            + "\n",
            encoding="utf-8",
        )
    return results


def comparison_table(results: dict) -> str:
    """Markdown table: one row per model/baseline, honest columns first."""
    rows = []
    for name, b in results["baselines"].items():
        rows.append(
            {
                "model": name,
                "PR-AUC val_tune*": b["val_tune"]["pr_auc"],
                "PR-AUC val_calib": b["val_calib"]["pr_auc"],
                "ROC-AUC val_calib": b["val_calib"]["roc_auc"],
                "lift@10% val_calib": np.nan,
                "calibration": "-",
                "Brier calib_eval raw": b["calib_eval"]["brier"],
                "Brier calib_eval cal.": np.nan,
            }
        )
    for m in MODEL_ORDER:
        r = results["models"][m]
        k10 = next(t for t in r["val_calib_with_ci"]["top_k"] if abs(t["k"] - 0.10) < 1e-9)
        rows.append(
            {
                "model": m,
                "PR-AUC val_tune*": r["raw_metrics"]["val_tune"]["pr_auc"],
                "PR-AUC val_calib": r["raw_metrics"]["val_calib"]["pr_auc"],
                "ROC-AUC val_calib": r["raw_metrics"]["val_calib"]["roc_auc"],
                "lift@10% val_calib": k10["lift"],
                "calibration": r["calibration"]["chosen"],
                "Brier calib_eval raw": r["calibration"]["calib_eval_raw"]["brier"],
                "Brier calib_eval cal.": r["calibration"]["calib_eval_calibrated"]["brier"],
            }
        )
    df = pd.DataFrame(rows).set_index("model")
    note = (
        "\n\n*val_tune is where hyperparameters were tuned, so it is optimistic; val_calib is "
        "later in time and untouched by tuning (the honest column). Brier columns are on "
        "calib_eval, scored by a calibrator fitted on calib_fit only."
    )
    return df.round(4).to_markdown() + note
