"""Capacity-aware policy simulation: "which orders should ops act on today?"

Assumptions (all explicit, all in config, all sensitivity-tested):
* Ops can act on at most the top k% of EACH DAY's orders (ceil, at least one).
* An intervention prevents a would-be-late order from being late with probability p. Nothing in
  this dataset tells us p; it is swept over `policy.intervention_success_grid`.
* Interventions on orders that would not have been late are wasted (no benefit, no harm).

Strategies compared: the model's ranking, the hand-written rule baseline, random selection
(analytic expectation: each day's picks catch that day's late share), and an oracle that picks
only truly late orders (upper bound given the capacity).

This is a SIMULATION with stated assumptions, not a measured business result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ontime.evaluation.metrics import daily_topk_table


def _bootstrap_extra(table: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float]:
    """95% CI (percentile, days resampled) of  captured - expected_random  in late orders."""
    cols = table[["captured", "expected_random"]].to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            (lambda s: s[0] - s[1])(cols[rng.integers(0, len(cols), len(cols))].sum(axis=0))
            for _ in range(n_boot)
        ]
    )
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def simulate_policy(
    dates: pd.Series,
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    k_grid: list[float],
    p_grid: list[float],
    seed: int,
    n_boot: int = 500,
) -> pd.DataFrame:
    """Long table: strategy x k x p -> expected late orders prevented and gain over random."""
    y = np.asarray(y)
    total_late = float(y.sum())
    rows = []
    for k in k_grid:
        # random + oracle depend only on the daily quota, so derive them from any strategy's table
        base = daily_topk_table(dates, y, next(iter(scores.values())), k, seed)
        base["expected_random"] = base["selected"] * base["late"] / base["orders"]
        oracle_captured = float(np.minimum(base["selected"], base["late"]).sum())
        random_captured = float(base["expected_random"].sum())
        strategies = {name: s for name, s in scores.items()}
        captured: dict[str, float] = {}
        ci: dict[str, tuple[float, float]] = {}
        for name, s in strategies.items():
            t = daily_topk_table(dates, y, s, k, seed)
            t["expected_random"] = t["selected"] * t["late"] / t["orders"]
            captured[name] = float(t["captured"].sum())
            ci[name] = _bootstrap_extra(t, n_boot, seed)
        captured["random"] = random_captured
        captured["oracle"] = oracle_captured
        ci["random"] = (0.0, 0.0)
        ci["oracle"] = (oracle_captured - random_captured,) * 2
        selected = float(base["selected"].sum())
        for name, cap in captured.items():
            for p in p_grid:
                rows.append(
                    {
                        "strategy": name,
                        "k": k,
                        "p": p,
                        "selected": selected,
                        "captured_late": cap,
                        "precision": cap / selected,
                        "recall": cap / total_late,
                        "prevented_expected": p * cap,
                        "prevented_share_of_late": p * cap / total_late,
                        "extra_vs_random": p * (cap - random_captured),
                        "extra_vs_random_ci_low": p * (ci[name][0]),
                        "extra_vs_random_ci_high": p * (ci[name][1]),
                    }
                )
    return pd.DataFrame(rows)
