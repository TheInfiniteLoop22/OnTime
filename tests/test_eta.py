"""S1 promise-date model: target definition, scoring rule and level selection."""

import numpy as np
import pandas as pd

from ontime.eta.quantile import (
    calendar_targets,
    check_lateness_definition,
    coverage,
    paired_bootstrap,
    pick_level,
    pinball_loss,
    promise_days,
)


def _table():
    appr = pd.to_datetime(["2018-02-17 16:15:34", "2018-03-01 08:00:00", "2018-03-05 23:59:00"])
    est = pd.to_datetime(["2018-03-20", "2018-03-10", "2018-03-12"])
    deliv = pd.to_datetime(["2018-03-20 00:59:25", "2018-03-11 09:00:00", "2018-03-12 23:00:00"])
    return pd.DataFrame(
        {
            "approved_at": appr,
            "delivery_days": (deliv - appr).total_seconds() / 86400,
            "promised_days": (est - appr).total_seconds() / 86400,
            # the dataset rule: late iff delivered DATE is after the promised DATE
            "is_late": (deliv.normalize() > est.normalize()).astype(int),
        }
    )


def test_integer_day_target_reproduces_is_late_exactly():
    t = calendar_targets(_table())
    assert check_lateness_definition(t) == 0.0
    # regression: float days used to land just before midnight and floor to the previous date
    assert t["promise_olist"].tolist() == [31, 9, 7]


def test_pinball_loss_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.exponential(5, 20_000)
    alpha = 0.9
    grid = np.linspace(1, 25, 120)
    best = grid[np.argmin([pinball_loss(y, np.full_like(y, c), alpha) for c in grid])]
    assert abs(best - np.quantile(y, alpha)) < 0.5


def test_promise_rounds_up_and_is_at_least_one_day():
    assert promise_days(np.array([0.2, 3.0, 3.01])).tolist() == [1, 3, 4]


def test_coverage_and_pick_level():
    assert coverage(np.array([1, 5, 9]), np.array([2, 5, 8])) == 2 / 3
    grid = {0.8: {"coverage": 0.88}, 0.9: {"coverage": 0.94}, 0.95: {"coverage": 0.97}}
    assert pick_level(grid, 0.93) == 0.9
    assert pick_level(grid, 0.999) == 0.95  # unreachable target -> the largest level


def test_paired_bootstrap_difference_matches_point_estimate():
    y = np.array([5, 8, 10, 12, 20] * 40)
    new, old = np.array([9, 9, 12, 12, 21] * 40), np.array([6, 7, 11, 13, 20] * 40)
    r = paired_bootstrap(y, new, old, 200, 1)
    assert abs(r["length_diff"] - (new - old).mean()) < 1e-12
    lo, hi = r["length_diff_ci"]
    assert lo <= r["length_diff"] <= hi
