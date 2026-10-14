"""S1 - promise-date model: quantile regression of delivery time (LightGBM, classical ML).

Question: could the platform quote a *shorter* promise at the same late-delivery risk, or a
*safer* promise at the same length, than the promise Olist actually quoted?

Target (integer calendar days, to match how lateness is defined):

    Y = date(delivered) - date(approved)         [days]
    Olist promise P_olist = date(estimated) - date(approved)
    late  <=>  Y > P        (this reproduces the dataset's `is_late` exactly; asserted)

A model promise is `ceil(q_alpha(x))` where q_alpha is the alpha-quantile of Y given the order's
approval-time features. Coverage = share of orders with Y <= promise. Only approval-time
features are used; no delivery information. Levels (alpha) are chosen on a validation slice,
never on test.
"""

from __future__ import annotations

import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ontime.config import load_config, project_path
from ontime.features.build import FEATURE_COLUMNS
from ontime.models.data import category_levels, load_feature_table, modelling_frames, prepare_frame
from ontime.models.splits import select_split

log = logging.getLogger(__name__)
ALPHAS = [0.5, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
PARAMS = {
    "n_estimators": 800, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "verbose": -1,
}  # fmt: skip


def calendar_targets(table: pd.DataFrame) -> pd.DataFrame:
    """Integer-day target `y_days` and Olist's integer-day promise `promise_olist`."""
    appr = pd.to_datetime(table["approved_at"])
    day0 = appr.dt.normalize()
    # round to the second: float days would otherwise land 1e-5 s before midnight and floor wrongly
    delivered = (appr + pd.to_timedelta(table["delivery_days"], unit="D")).dt.round("s")
    promised = (appr + pd.to_timedelta(table["promised_days"], unit="D")).dt.round("s")
    out = table.copy()
    out["y_days"] = (delivered.dt.normalize() - day0).dt.days.astype(int)
    out["promise_olist"] = (promised.dt.normalize() - day0).dt.days.astype(int)
    return out


def check_lateness_definition(t: pd.DataFrame) -> float:
    """Share of orders where `y_days > promise_olist` disagrees with `is_late` (must be ~0)."""
    return float(((t["y_days"] > t["promise_olist"]).astype(int) != t["is_late"]).mean())


def pinball_loss(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    """Mean pinball loss: the proper scoring rule whose minimiser is the alpha-quantile."""
    d = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def fit_quantile(Xtr, ytr, Xes, yes, alpha: float, seed: int, early: int = 50):  # noqa: ANN001
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=seed, **PARAMS)
    m.fit(
        Xtr, ytr, eval_set=[(Xes, yes)], eval_metric="quantile",
        callbacks=[lgb.early_stopping(early, verbose=False)],
    )  # fmt: skip
    return m


def promise_days(q: np.ndarray) -> np.ndarray:
    """What the customer is quoted: whole days, rounded up, at least 1."""
    return np.maximum(1, np.ceil(np.asarray(q, dtype=float) - 1e-9)).astype(int)


def coverage(y_days: np.ndarray, promise: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_days) <= np.asarray(promise)))


def pick_level(grid: dict[float, dict], target: float) -> float:
    """Smallest alpha whose validation coverage reaches `target` (else the largest alpha)."""
    ok = [a for a, v in sorted(grid.items()) if v["coverage"] >= target]
    return ok[0] if ok else max(grid)


def paired_bootstrap(y, p_new, p_old, n_boot: int, seed: int) -> dict:  # noqa: ANN001
    """Order-bootstrap CIs for (mean promise length, coverage) differences, new minus old."""
    rng = np.random.default_rng(seed)
    n = len(y)
    ok_new, ok_old = (y <= p_new).astype(float), (y <= p_old).astype(float)
    ln = p_new.astype(float) - p_old.astype(float)
    dl, dc = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        dl.append(ln[i].mean())
        dc.append((ok_new[i] - ok_old[i]).mean())
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {
        "length_diff": float(ln.mean()), "length_diff_ci": ci(dl),
        "coverage_diff": float((ok_new - ok_old).mean()), "coverage_diff_ci": ci(dc),
    }  # fmt: skip


def run_eta(write: bool = True) -> dict:
    cfg = load_config()
    seed, n_boot = cfg["seed"], cfg["tuning"]["bootstrap_resamples"]
    table = calendar_targets(load_feature_table())
    mismatch = check_lateness_definition(table)
    if mismatch > 0.01:
        raise RuntimeError(
            f"integer-day lateness disagrees with is_late for {mismatch:.2%} of orders"
        )
    levels = category_levels(table)
    fr = {k: v for k, v in modelling_frames(table, cfg).items()}
    train, tune, calib = fr["train"], fr["val_tune"], fr["val_calib"]
    Xtr, Xtu, Xca = (prepare_frame(d, levels) for d in (train, tune, calib))
    ytr, ytu, yca = (d["y_days"].to_numpy() for d in (train, tune, calib))

    # ---- fit one model per level; early stopping on val_tune, level choice on val_calib
    models, grid = {}, {}
    for a in ALPHAS:
        m = fit_quantile(Xtr, ytr, Xtu, ytu, a, seed)
        models[a] = m
        q = m.predict(Xca)
        pr = promise_days(q)
        grid[a] = {
            "coverage": coverage(yca, pr), "mean_promise_days": float(pr.mean()),
            "pinball": pinball_loss(yca, q, a), "best_iteration": int(m.best_iteration_ or 0),
        }  # fmt: skip
    olist_val = {
        "coverage": coverage(yca, calib["promise_olist"].to_numpy()),
        "mean_promise_days": float(calib["promise_olist"].mean()),
    }
    # targets: match Olist's own validation coverage, then two safer levels
    targets = {
        "match_olist": olist_val["coverage"],
        "cov_0.95": 0.95,
        "cov_0.97": 0.97,
    }
    chosen = {k: pick_level(grid, v) for k, v in targets.items()}
    log.info("eta | olist val coverage %.3f; chosen alphas %s", olist_val["coverage"], chosen)

    # ---- refit chosen levels on train + val_tune (more recent outcomes), no early stopping change
    # (kept simple and explainable: models above are used as fitted; see DECISIONS_LOG)
    # ---- test: scored once, disclosed
    test = select_split(table, "test", final_evaluation=True).reset_index(drop=True)
    yte = test["y_days"].to_numpy()
    Xte = prepare_frame(test, levels)
    p_olist = test["promise_olist"].to_numpy()
    olist_test = {"coverage": coverage(yte, p_olist), "mean_promise_days": float(p_olist.mean())}
    frontier = []
    for a in ALPHAS:
        q = models[a].predict(Xte)
        pr = promise_days(q)
        frontier.append({
            "alpha": a, "coverage": coverage(yte, pr), "mean_promise_days": float(pr.mean()),
            "pinball": pinball_loss(yte, q, a),
        })  # fmt: skip
    # baseline: Olist's own promise plus a constant number of days
    shift = [
        {"shift_days": s, "coverage": coverage(yte, p_olist + s),
         "mean_promise_days": float((p_olist + s).mean())}
        for s in range(0, 9)
    ]  # fmt: skip
    shift_val = {s: coverage(yca, calib["promise_olist"].to_numpy() + s) for s in range(0, 9)}
    pin_olist = {  # Olist's promise scored as if it were each alpha-quantile (reference only)
        a: pinball_loss(yte, p_olist, a) for a in ALPHAS
    }
    results = {}
    for name, a in chosen.items():
        pr = promise_days(models[a].predict(Xte))
        results[name] = {
            "alpha": a, "target_validation_coverage": targets[name],
            "test_coverage": coverage(yte, pr), "test_mean_promise_days": float(pr.mean()),
            "vs_olist": paired_bootstrap(yte, pr, p_olist, n_boot, seed),
        }  # fmt: skip
        # same-coverage-target baseline: smallest constant shift reaching the validation target
        ok = [s for s, c in shift_val.items() if c >= targets[name]]
        if ok:
            s = min(ok)
            results[name]["shift_baseline"] = {
                "shift_days": s, "test_coverage": coverage(yte, p_olist + s),
                "test_mean_promise_days": float((p_olist + s).mean()),
            }  # fmt: skip
    # descriptive (uses test, not for selection): promise length at Olist's test coverage, and
    # on-time share at Olist's test length, read off the model frontier by interpolation
    f = pd.DataFrame(frontier).sort_values("coverage")
    interp = {
        "length_at_olist_test_coverage": float(
            np.interp(olist_test["coverage"], f["coverage"], f["mean_promise_days"])
        ),
        "coverage_at_olist_test_length": float(
            np.interp(olist_test["mean_promise_days"], f["mean_promise_days"], f["coverage"])
        ),
    }
    out = {
        "disclosure": "Third look at the test months (after stage 1 and stage 2); levels chosen on "
        "validation only. Development evidence, not a first-look hold-out.",
        "definition": "Y = date(delivered) - date(approved) in whole days; "
        "promise = ceil(q_alpha); late <=> Y > promise (matches Olist is_late).",
        "lateness_definition_mismatch": mismatch,
        "orders": {
            "train": int(len(train)),
            "val_tune": int(len(tune)),
            "val_calib": int(len(calib)),
            "test": int(len(test)),
        },  # fmt: skip
        "validation": {"olist": olist_val, "grid": {str(a): v for a, v in grid.items()}},
        "test": {
            "olist": olist_test,
            "frontier": frontier,
            "shift_baseline": shift,
            "olist_pinball_by_alpha": {str(a): v for a, v in pin_olist.items()},
            "interpolated_same_level": interp,
            "chosen": results,
        },  # fmt: skip
        "features": FEATURE_COLUMNS,
        "params": PARAMS,
    }
    if write:
        rep = project_path(cfg["paths"]["reports_dir"])
        (rep / "eta_results.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8", newline="\n"
        )
        mdir = project_path(cfg["paths"]["models_dir"])
        mdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"models": models, "levels": levels, "chosen": chosen, "alphas": ALPHAS,
             "columns": FEATURE_COLUMNS},
            mdir / "eta.joblib",
        )  # fmt: skip
        _plot(out, rep / "figures" / "eta_frontier.png")
    return out


def _plot(r: dict, path) -> None:  # noqa: ANN001
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = r["test"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    f = pd.DataFrame(t["frontier"])
    ax.plot(f["mean_promise_days"], f["coverage"] * 100, "o-", color="#4C78A8",
            label="Quantile model (alpha 0.5 ... 0.99)")  # fmt: skip
    s = pd.DataFrame(t["shift_baseline"])
    ax.plot(s["mean_promise_days"], s["coverage"] * 100, "s--", color="#9C9C9C",
            label="Olist promise + constant days")  # fmt: skip
    o = t["olist"]
    ax.scatter([o["mean_promise_days"]], [o["coverage"] * 100], s=90, color="#E45756",
               zorder=5, label="Olist promise as quoted")  # fmt: skip
    ax.set_xlabel("Average promise length (days from approval)")
    ax.set_ylabel("Orders delivered on or before promise (%)")
    ax.set_title("Promise length vs on-time share (test months)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
