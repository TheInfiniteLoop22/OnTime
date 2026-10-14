"""Rolling-origin backtest: does RETRAINING every month, and dropping the calendar proxies, help?

Motivation (Phase 7): the static model's ranking quality decayed over the months after training.
Two cheap, standard remedies are tested head to head against the deployed static model:

  A. retrain monthly on all outcomes known at the start of each month (expanding window);
  B. drop calendar features that mostly encode "which part of the year the data covered".

For each evaluation month s:
  * training rows = orders approved AND DELIVERED before s (outcomes known at s, no look-ahead);
  * logistic regression: C chosen by validating on the last N days before s (inner forward split);
  * LightGBM: fixed generic parameters, early stopping on the latest 15% of training rows by time;
  * score the month with each variant; the static v1 model (trained on data to 2018-01-31) is scored
    on the same rows.

HONEST STATUS: this is a DEVELOPMENT backtest. February-April were v1's validation period, and
May-August were v1's test period, which has already been looked at once. The design question
("does retraining help?") was motivated by that look, so these months are no longer a clean hold-out
for the variants. Nothing here changes the Phase 7 headline numbers.
"""

from __future__ import annotations

import json
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from ontime.config import load_config, project_path  # noqa: E402
from ontime.evaluation.metrics import daily_topk_table  # noqa: E402
from ontime.features.build import FEATURE_COLUMNS  # noqa: E402
from ontime.models.data import category_levels, prepare_frame  # noqa: E402
from ontime.models.train import fit_lgbm, fit_logreg  # noqa: E402

log = logging.getLogger(__name__)

VARIANTS = {
    "static_v1": "Deployed v1 (trained once, to Jan 2018)",
    "retrain_logreg_full": "Logistic regression, monthly retrain, all features",
    "retrain_logreg_noproxy": "Logistic regression, monthly retrain, no calendar proxies",
    "retrain_lgbm_full": "LightGBM, monthly retrain, all features",
    "retrain_lgbm_noproxy": "LightGBM, monthly retrain, no calendar proxies",
}


def _pooled_lift(tbl: pd.DataFrame) -> float:
    sel, cap, late, n = (tbl[c].sum() for c in ("selected", "captured", "late", "orders"))
    return float((cap / sel) / (late / n)) if sel and late else float("nan")


def paired_day_bootstrap_lift(
    a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int
) -> tuple[float, tuple[float, float]]:
    """Lift(a) - Lift(b) pooled over days, with a paired CI from resampling whole days."""
    m = a.merge(b, on="day", suffixes=("_a", "_b"))
    cols = ["orders_a", "late_a", "selected_a", "captured_a", "selected_b", "captured_b"]
    arr = m[cols].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    def diff(x: np.ndarray) -> float:
        n, late, sa, ca, sb, cb = x.sum(axis=0)
        base = late / n
        return (ca / sa) / base - (cb / sb) / base

    point = diff(arr)
    draws = [diff(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n_boot)]
    return float(point), (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def _choose_c(Xtr, ytr, Xva, yva, c_grid, seed, feats):  # noqa: ANN001
    best_c, best = c_grid[0], -1.0
    for c in c_grid:
        m = fit_logreg(Xtr, ytr, {"C": c, "class_weight": "none"}, seed, feats)
        ap = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        if ap > best:
            best_c, best = c, ap
    return best_c


def run_rolling(write: bool = True, n_boot: int | None = None) -> dict:
    cfg = load_config()
    rc, seed = cfg["rolling"], cfg["seed"]
    n_boot = n_boot or cfg["tuning"]["bootstrap_resamples"]
    table = pd.read_parquet(project_path(cfg["paths"]["features_parquet"]))
    deliv = pd.read_parquet(
        project_path(cfg["paths"]["orders_parquet"]),
        columns=["order_id", "order_delivered_customer_date"],
    ).rename(columns={"order_delivered_customer_date": "delivered_at"})
    table = (
        table.merge(deliv, on="order_id", how="left")
        .sort_values("approved_at")
        .reset_index(drop=True)
    )
    levels = category_levels(table)
    X_all = prepare_frame(table, levels)
    y_all = table["is_late"].to_numpy()
    proxies = set(rc["proxy_features"])
    feats_full = list(FEATURE_COLUMNS)
    feats_np = [f for f in FEATURE_COLUMNS if f not in proxies]

    art = joblib.load(project_path(cfg["paths"]["models_dir"]) / "logreg.joblib")
    static_score = art["model"].predict_proba(prepare_frame(table, art["levels"]))[:, 1]

    months = pd.date_range(rc["first_month"], rc["last_month"], freq="MS")
    inner = pd.Timedelta(days=rc["inner_validation_days"])
    scores: dict[str, np.ndarray] = {k: np.full(len(table), np.nan) for k in VARIANTS}
    scores["static_v1"] = static_score
    chosen_c: dict[str, dict[str, float]] = {}
    sizes = {}

    for s in months:
        e = s + pd.offsets.MonthBegin(1)
        known = (table["approved_at"] < s) & (table["delivered_at"] < s)
        test_idx = np.flatnonzero((table["approved_at"] >= s) & (table["approved_at"] < e))
        tr_idx = np.flatnonzero(known)
        if len(tr_idx) < rc["min_train_orders"] or len(test_idx) == 0:
            continue
        sizes[str(s.date())] = {"train_orders": int(len(tr_idx)), "eval_orders": int(len(test_idx))}
        Xtr, ytr = X_all.iloc[tr_idx], y_all[tr_idx]
        t_tr = table["approved_at"].iloc[tr_idx]
        fit_mask = (t_tr < s - inner).to_numpy()
        chosen_c[str(s.date())] = {}
        for name, feats in (
            ("retrain_logreg_full", feats_full),
            ("retrain_logreg_noproxy", feats_np),
        ):
            c = _choose_c(
                Xtr[fit_mask],
                ytr[fit_mask],
                Xtr[~fit_mask],
                ytr[~fit_mask],
                rc["c_grid"],
                seed,
                feats,
            )
            m = fit_logreg(Xtr, ytr, {"C": c, "class_weight": "none"}, seed, feats)
            scores[name][test_idx] = m.predict_proba(X_all.iloc[test_idx])[:, 1]
            chosen_c[str(s.date())][name] = c
        es_cut = t_tr.quantile(0.85)
        es = (t_tr > es_cut).to_numpy()
        for name, feats in (("retrain_lgbm_full", feats_full), ("retrain_lgbm_noproxy", feats_np)):
            m = fit_lgbm(Xtr.loc[~es, feats], ytr[~es], rc["lgbm"], seed, Xtr.loc[es, feats], ytr[es],
                         cfg["tuning"]["lgbm_early_stopping_rounds"])  # fmt: skip
            scores[name][test_idx] = m.predict_proba(X_all.iloc[test_idx][feats])[:, 1]
        log.info("rolling | %s: train=%d eval=%d", s.date(), len(tr_idx), len(test_idx))

    evaluated = np.isfinite(scores["retrain_logreg_full"])
    ev = table[evaluated].reset_index(drop=True)
    ev["month"] = ev["approved_at"].dt.strftime("%Y-%m")
    rows, daily = [], {}
    for name in VARIANTS:
        sc = scores[name][evaluated]
        for mon, g in ev.groupby("month"):
            y, p = g["is_late"].to_numpy(), sc[g.index]
            tbl = daily_topk_table(g["approved_at"], y, p, 0.10, seed)
            rows.append({"variant": name, "month": mon, "orders": len(g), "late_rate": float(y.mean()),
                         "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                         "lift_at_10": _pooled_lift(tbl)})  # fmt: skip
        daily[name] = daily_topk_table(ev["approved_at"], ev["is_late"].to_numpy(), sc, 0.10, seed)
    monthly = pd.DataFrame(rows)

    summary = {}
    for name in VARIANTS:
        m = monthly[monthly["variant"] == name]
        entry = {
            "label": VARIANTS[name],
            "macro_roc_auc": float(m["roc_auc"].mean()),
            "pooled_lift_at_10": _pooled_lift(daily[name]),
            "last_month_roc_auc": float(m.sort_values("month")["roc_auc"].iloc[-1]),
            "last_month_lift_at_10": float(m.sort_values("month")["lift_at_10"].iloc[-1]),
        }
        if name != "static_v1":
            d, (lo, hi) = paired_day_bootstrap_lift(daily[name], daily["static_v1"], n_boot, seed)
            entry["lift_diff_vs_static"] = {"diff": d, "ci95": [lo, hi]}
        summary[name] = entry
    # the two ablation comparisons that answer "do the proxies matter?"
    ablation = {}
    for fam in ("logreg", "lgbm"):
        d, (lo, hi) = paired_day_bootstrap_lift(
            daily[f"retrain_{fam}_noproxy"], daily[f"retrain_{fam}_full"], n_boot, seed
        )
        ablation[fam] = {"noproxy_minus_full_lift": d, "ci95": [lo, hi]}

    result = {
        "status": "DEVELOPMENT BACKTEST - months were already seen (validation Feb-Apr, test May-Aug of v1); "
        "not a clean hold-out.",  # fmt: skip
        "months": sorted(ev["month"].unique().tolist()),
        "training_sizes": sizes,
        "chosen_C": chosen_c,
        "summary": summary,
        "proxy_ablation": ablation,
        "monthly": monthly.round(4).to_dict(orient="records"),
        "proxy_features_dropped": sorted(proxies),
    }
    if write:
        rep = project_path(cfg["paths"]["reports_dir"])
        (rep / "rolling_backtest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        _plot(monthly, project_path(cfg["paths"]["figures_dir"]) / "rolling_backtest.png")
        log.info("rolling | wrote rolling_backtest.json + figure")
    return result


def _plot(monthly: pd.DataFrame, path) -> None:  # noqa: ANN001
    colors = {"static_v1": "#222222", "retrain_logreg_full": "#4C78A8", "retrain_logreg_noproxy": "#1BAF7A",
              "retrain_lgbm_full": "#EB6834", "retrain_lgbm_noproxy": "#6250D6"}  # fmt: skip
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, label in VARIANTS.items():
        g = monthly[monthly["variant"] == name].sort_values("month")
        ls = "--" if name == "static_v1" else "-"
        ax[0].plot(g["month"], g["roc_auc"], ls, marker="o", color=colors[name], label=label)
        ax[1].plot(g["month"], g["lift_at_10"], ls, marker="o", color=colors[name], label=label)
    ax[0].axhline(0.5, color="grey", lw=1)
    ax[1].axhline(1.0, color="grey", lw=1)
    ax[0].set_title("ROC-AUC by month")
    ax[1].set_title("Lift at 10% daily capacity by month")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Retraining study (development backtest, not a clean hold-out)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
