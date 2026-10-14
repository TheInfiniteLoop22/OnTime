"""Baselines: the floor every learned model must clear.

* PriorRateBaseline - predicts the training late rate for everyone. Its PR-AUC equals the base
  rate of whatever it is scored on; anything below "model PR-AUC / base rate = lift" is noise.
* RuleBaseline - what an ops analyst would write by hand. Candidate rules are fixed in advance
  (named in the spec) and their threshold is chosen on TRAIN ONLY by maximizing F1:
    A. inter-state AND promised_days < X
    B. distance_km > D
  Output is a two-level probability (late rate among flagged / unflagged train orders), so it
  can be scored with the same probability metrics as the models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ontime.config import load_config, project_path
from ontime.evaluation.metrics import evaluate_scores
from ontime.models.splits import assert_no_time_overlap, select_split

log = logging.getLogger(__name__)


@dataclass
class PriorRateBaseline:
    """Predict the training-set late rate for every order."""

    rate_: float = float("nan")

    def fit(self, y_train: np.ndarray) -> PriorRateBaseline:
        self.rate_ = float(np.mean(y_train))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.rate_)


def _f1(y: np.ndarray, flag: np.ndarray) -> float:
    tp = float((flag & (y == 1)).sum())
    fp = float((flag & (y == 0)).sum())
    fn = float((~flag & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


@dataclass
class RuleBaseline:
    """Hand-written rule with a threshold fitted on training data only."""

    family: str = ""  # "interstate_tight_promise" | "far_distance"
    threshold: float = float("nan")
    p_flagged: float = float("nan")
    p_unflagged: float = float("nan")
    train_f1: float = float("nan")

    def flag(self, df: pd.DataFrame) -> np.ndarray:
        if self.family == "interstate_tight_promise":
            return ((df["same_state"] == 0) & (df["promised_days"] < self.threshold)).to_numpy()
        if self.family == "far_distance":
            return (df["distance_km"] > self.threshold).to_numpy()
        raise ValueError(f"unfitted or unknown rule family {self.family!r}")

    def fit(self, train: pd.DataFrame) -> RuleBaseline:
        y = train["is_late"].to_numpy()
        best = (-1.0, "", float("nan"))
        grid = np.arange(5, 96, 5)
        candidates = {
            "interstate_tight_promise": np.percentile(train["promised_days"], grid),
            "far_distance": np.percentile(train["distance_km"], grid),
        }
        for family, thresholds in candidates.items():
            for thr in np.unique(thresholds):
                self.family, self.threshold = family, float(thr)
                flag = self.flag(train)
                if flag.mean() < 0.01 or flag.mean() > 0.99:  # degenerate rule
                    continue
                score = _f1(y, flag)
                if score > best[0]:
                    best = (score, family, float(thr))
        self.train_f1, self.family, self.threshold = best
        flag = self.flag(train)
        self.p_flagged = float(y[flag].mean())
        self.p_unflagged = float(y[~flag].mean())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.where(self.flag(df), self.p_flagged, self.p_unflagged)

    def describe(self) -> str:
        if self.family == "interstate_tight_promise":
            return (
                f"seller and customer in different states AND promised_days < {self.threshold:.1f}"
            )
        return f"distance_km > {self.threshold:,.0f}"


def run_baselines(write: bool = True) -> dict:
    """Fit both baselines on TRAIN, evaluate on VALIDATION (test stays locked)."""
    cfg = load_config()
    table = pd.read_parquet(project_path(cfg["paths"]["features_parquet"]))
    assert_no_time_overlap(table)
    train = select_split(table, "train")
    val = select_split(table, "validation")
    seed = cfg["seed"]
    n_boot = cfg["tuning"]["bootstrap_resamples"]
    k_grid = cfg["policy"]["capacity_grid"]
    log.info("baselines | train=%d validation=%d (test locked)", len(train), len(val))

    prior = PriorRateBaseline().fit(train["is_late"].to_numpy())
    rule = RuleBaseline().fit(train)
    log.info(
        "baselines | prior rate=%.4f | rule: %s (train F1=%.3f)",
        prior.rate_,
        rule.describe(),
        rule.train_f1,
    )

    result = {
        "evaluated_on": "validation",
        "note": "Test set not touched. Rule thresholds and prior rate come from train only.",
        "train": {
            "orders": int(len(train)),
            "late_rate": float(train["is_late"].mean()),
        },
        "validation": {
            "orders": int(len(val)),
            "late_rate": float(val["is_late"].mean()),
        },
        "models": {},
    }
    for name, model in {"prior_rate": prior, "rule": rule}.items():
        p = model.predict_proba(val)
        entry = evaluate_scores(
            val["approved_at"], val["is_late"].to_numpy(), p, k_grid, n_boot, seed
        )
        if name == "prior_rate":
            entry["description"] = (
                f"predicts the training late rate ({prior.rate_:.4f}) for every order"
            )
        else:
            entry["description"] = rule.describe()
            entry["fitted"] = {
                "family": rule.family,
                "threshold": rule.threshold,
                "p_flagged": rule.p_flagged,
                "p_unflagged": rule.p_unflagged,
                "train_f1": rule.train_f1,
                "share_flagged_validation": float(rule.flag(val).mean()),
            }
        result["models"][name] = entry
    if write:
        out = project_path(cfg["paths"]["reports_dir"]) / "baselines.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("baselines | wrote %s", out)
    return result
