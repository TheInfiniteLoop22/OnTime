"""Probability calibration: map raw model scores to honest probabilities.

Why calibrate: the app will tell a human "34% chance this order is late". That sentence is only
true if, among orders scored 0.34, about 34% really are late. Ranking quality (PR-AUC) is
unaffected by a monotone map; Brier score and log loss are.

Three options are compared by Brier score on data the calibrator did not see:
  none      - use raw scores
  sigmoid   - Platt scaling: logistic regression on the logit of the score (2 parameters)
  isotonic  - monotone step function (flexible; can overfit small samples, creates ties)
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class IdentityCalibrator:
    name = "none"

    def fit(self, p, y):  # noqa: ANN001
        return self

    def transform(self, p):  # noqa: ANN001
        return np.asarray(p, dtype=float)


class SigmoidCalibrator:
    name = "sigmoid"

    def fit(self, p, y):  # noqa: ANN001
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def transform(self, p):  # noqa: ANN001
        return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    name = "isotonic"

    def fit(self, p, y):  # noqa: ANN001
        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p):  # noqa: ANN001
        return self.iso_.predict(np.asarray(p, dtype=float))


CALIBRATORS = {
    "none": IdentityCalibrator,
    "sigmoid": SigmoidCalibrator,
    "isotonic": IsotonicCalibrator,
}


def choose_calibrator(
    p_fit: np.ndarray, y_fit: np.ndarray, p_eval: np.ndarray, y_eval: np.ndarray
) -> tuple[str, dict[str, dict[str, float]]]:
    """Fit each option on (p_fit, y_fit), score Brier/log-loss on (p_eval, y_eval); pick the best.

    The evaluation rows come strictly AFTER the fit rows in time, so the comparison mimics
    deployment (calibrate on the past, use on the future) and cannot favour the more flexible
    isotonic map just because it can memorize its own fitting data.
    """
    table: dict[str, dict[str, float]] = {}
    for name, cls in CALIBRATORS.items():
        p = np.clip(cls().fit(p_fit, y_fit).transform(p_eval), 1e-15, 1 - 1e-15)
        brier = float(brier_score_loss(y_eval, p))
        ll = float(-np.mean(y_eval * np.log(p) + (1 - y_eval) * np.log(1 - p)))
        table[name] = {"brier": brier, "log_loss": ll}
    best = min(table, key=lambda k: table[k]["brier"])
    return best, table
