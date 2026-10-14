"""Hyperparameter search with Optuna. Objective = PR-AUC on `val_tune` (time-ordered).

Only `train` and `val_tune` are visible here. `val_calib` (later in time) and `test` are never
passed in, so tuning cannot overfit the rows that calibration and final evaluation rely on.

Every study records the class-weighting option of each trial, so the report can answer
"did re-weighting the minority class help?" with data.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from ontime.models.train import fit_ebm, fit_lgbm, fit_logreg

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _run_study(objective: Callable, n_trials: int, seed: int, name: str) -> optuna.Study:
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed), study_name=name
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info(
        "tune | %s best val_tune PR-AUC=%.4f after %d trials", name, study.best_value, n_trials
    )
    return study


def weight_experiment(study: optuna.Study) -> dict[str, dict[str, float]]:
    """Best and mean trial PR-AUC per class-weight option."""
    rows: dict[str, list[float]] = {}
    for t in study.trials:
        if t.value is not None:
            rows.setdefault(t.params.get("class_weight", "n/a"), []).append(t.value)
    return {
        k: {"best_pr_auc": float(max(v)), "mean_pr_auc": float(np.mean(v)), "trials": len(v)}
        for k, v in rows.items()
    }


def tune_logreg(Xtr, ytr, Xtu, ytu, n_trials: int, seed: int):  # noqa: ANN001
    def objective(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced"]),
        }
        m = fit_logreg(Xtr, ytr, params, seed)
        return float(average_precision_score(ytu, m.predict_proba(Xtu)[:, 1]))

    study = _run_study(objective, n_trials, seed, "logreg")
    return study.best_params, study


def tune_lgbm(Xtr, ytr, Xtu, ytu, n_trials: int, seed: int, es_rounds: int):  # noqa: ANN001
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["none", "sqrt", "balanced"]),
        }
        m = fit_lgbm(Xtr, ytr, params, seed, Xtu, ytu, es_rounds)
        trial.set_user_attr("best_iteration", int(m.best_iteration_ or m.n_estimators_))
        return float(average_precision_score(ytu, m.predict_proba(Xtu)[:, 1]))

    study = _run_study(objective, n_trials, seed, "lightgbm")
    return study.best_params, study


def tune_ebm(  # noqa: ANN201
    Xtr: pd.DataFrame,
    ytr,
    approved_tr: pd.Series,
    Xtu,
    ytu,
    n_trials: int,
    seed: int,
    es_frac: float,
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_bins": trial.suggest_categorical("max_bins", [64, 128, 256, 512]),
            "interactions": trial.suggest_categorical("interactions", [0, 5, 10]),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_leaves": trial.suggest_int("max_leaves", 2, 4),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [10, 50, 100]),
        }
        m = fit_ebm(Xtr, ytr, approved_tr, params, seed, es_frac)
        return float(average_precision_score(ytu, m.predict_proba(Xtu)[:, 1]))

    study = _run_study(objective, n_trials, seed, "ebm")
    return study.best_params, study
