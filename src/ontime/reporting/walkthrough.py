"""Tables for notebooks/03_modeling_walkthrough.ipynb (all read from generated result files)."""

from __future__ import annotations

import json

import pandas as pd

from ontime.config import load_config, project_path

MODELS = ["logreg", "random_forest", "lightgbm", "ebm"]
LABEL = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
    "ebm": "EBM",
    "prior_rate": "Guess the training rate",
    "rule": "Hand-written rule",
}


def _reports() -> str:
    return project_path(load_config()["paths"]["reports_dir"])


def validation() -> dict:
    return json.loads((_reports() / "validation_results.json").read_text("utf-8"))


def results() -> dict:
    return json.loads((_reports() / "results.json").read_text("utf-8"))


def split_table() -> pd.DataFrame:
    """Orders, late rate and date range of every sub-split (train, validation pieces)."""
    v = validation()["sub_splits"]
    t = results()["population"]
    rows = [{"split": k, "from": s["from"], "to": s["to"], "orders": s["orders"], "late_rate": s["late_rate"]}
            for k, s in v.items() if k != "validation"]  # fmt: skip
    rows.append({"split": "test", "from": t["from"], "to": t["to"], "orders": t["orders"],
                 "late_rate": t["late_rate"]})  # fmt: skip
    return pd.DataFrame(rows)


def tuned_params() -> pd.DataFrame:
    v = validation()["models"]
    rows = []
    for m in MODELS:
        p = v[m]["best_params"]
        rows.append(
            {
                "model": LABEL[m],
                "chosen hyperparameters": ", ".join(f"{k}={_fmt(x)}" for k, x in p.items()),
            }
        )
    return pd.DataFrame(rows)


def _fmt(x) -> str:  # noqa: ANN001
    return f"{x:.3g}" if isinstance(x, float) else str(x)


def class_weight_table() -> pd.DataFrame:
    rows = []
    for model, d in validation()["class_weight_experiment"].items():
        for opt, s in d.items():
            rows.append({"model": LABEL[model], "class weighting": opt, "best PR-AUC (val_tune)": s["best_pr_auc"],
                         "mean PR-AUC": s["mean_pr_auc"], "trials": s["trials"]})  # fmt: skip
    return pd.DataFrame(rows)


def tuned_vs_honest() -> pd.DataFrame:
    """PR-AUC on the slice tuned on vs on the later untouched validation slice."""
    v = validation()
    rows = []
    for m in MODELS:
        r = v["models"][m]["raw_metrics"]
        rows.append({"model": LABEL[m], "train": r["train"]["pr_auc"], "val_tune (tuned on)": r["val_tune"]["pr_auc"],
                     "val_calib (honest)": r["val_calib"]["pr_auc"]})  # fmt: skip
    for b in ("prior_rate", "rule"):
        r = v["baselines"][b]
        rows.append({"model": LABEL[b], "train": r["train"]["pr_auc"], "val_tune (tuned on)": r["val_tune"]["pr_auc"],
                     "val_calib (honest)": r["val_calib"]["pr_auc"]})  # fmt: skip
    df = pd.DataFrame(rows)
    base = {"val_tune (tuned on)": v["sub_splits"]["val_tune"]["late_rate"],
            "val_calib (honest)": v["sub_splits"]["val_calib"]["late_rate"]}  # fmt: skip
    return df, base


def calibration_table() -> pd.DataFrame:
    v = validation()["models"]
    rows = []
    for m in MODELS:
        c = v[m]["calibration"]
        s = c["scores_on_calib_eval"]
        rows.append({"model": LABEL[m], "chosen": c["chosen"], "Brier none": s["none"]["brier"],
                     "Brier sigmoid": s["sigmoid"]["brier"], "Brier isotonic": s["isotonic"]["brier"]})  # fmt: skip
    return pd.DataFrame(rows)


def test_table() -> pd.DataFrame:
    r = results()
    rows = []
    for k in ("prior_rate", "rule", *MODELS):
        e = r["baselines"].get(k) or r["models"][k]
        m = e["metrics"]
        k10 = next(t for t in e["top_k"] if abs(t["k"] - 0.10) < 1e-9)
        rows.append({"key": k, "model": LABEL[k], "PR-AUC": m["pr_auc"]["value"], "lo": m["pr_auc"]["ci95"][0],
                     "hi": m["pr_auc"]["ci95"][1], "ROC-AUC": m["roc_auc"]["value"], "Brier": m["brier"]["value"],
                     "lift@10%": k10["lift"]})  # fmt: skip
    return pd.DataFrame(rows)


def paired_differences() -> pd.DataFrame:
    d = results()["paired_pr_auc_diff"]
    return pd.DataFrame([{"comparison": k.replace("_minus_", " minus "), "PR-AUC difference": v["diff"],
                          "CI low": v["ci95"][0], "CI high": v["ci95"][1]} for k, v in d.items()])  # fmt: skip


def monthly() -> pd.DataFrame:
    return pd.DataFrame(results()["monthly"])
