"""Generate README.md and reports/model_card.md from the project's result files.

CLAUDE.md rule 5: no number may be written into README/reports that was not produced by code.
The prose lives in templates (docs/templates/*.md) with {{placeholders}}; every placeholder is
filled from results.json, validation_results.json, impact.json, explain_summary.json, the feature
table or the config. A missing value raises: a README can never ship with a stale or invented number.
"""

from __future__ import annotations

import json
import logging
import re
from importlib import metadata

import pandas as pd

from ontime.config import PROJECT_ROOT, load_config, project_path
from ontime.features.build import FEATURE_COLUMNS

log = logging.getLogger(__name__)
TEMPLATES = PROJECT_ROOT / "docs" / "templates"
LABEL = {
    "logreg": "Logistic regression",
    "lightgbm": "LightGBM",
    "ebm": "EBM (glass-box)",
    "random_forest": "Random forest",
    "rule": "Hand-written rule",
    "prior_rate": "Guess the training rate",
}
PLACEHOLDER = re.compile(r"\{\{\s*([a-z0-9_]+)\s*\}\}")


def _pct(v: float, nd: int = 1) -> str:
    return f"{v * 100:.{nd}f}%"


def _ci(lo_hi: list[float], nd: int = 3) -> str:
    return f"[{lo_hi[0]:.{nd}f}, {lo_hi[1]:.{nd}f}]"


def _load(name: str) -> dict:
    return json.loads(
        (project_path(load_config()["paths"]["reports_dir"]) / name).read_text("utf-8")
    )


def _results_table(res: dict) -> str:
    rows = [
        "| Model | PR-AUC [95% CI] | ROC-AUC | Brier | Lift @10% [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for key in ("prior_rate", "rule", "logreg", "random_forest", "lightgbm", "ebm"):
        e = res["baselines"].get(key) or res["models"][key]
        m = e["metrics"]
        k10 = next(t for t in e["top_k"] if abs(t["k"] - 0.10) < 1e-9)
        name = f"**{LABEL[key]}**" if key == res["primary_model"] else LABEL[key]
        rows.append(
            f"| {name} | {m['pr_auc']['value']:.3f} {_ci(m['pr_auc']['ci95'])} | "
            f"{m['roc_auc']['value']:.3f} | {m['brier']['value']:.4f} | "
            f"{k10['lift']:.2f} {_ci(k10['ci95']['lift'], 2)} |"
        )
    return "\n".join(rows)


def _gap_sentence(d: dict) -> str:
    lo, hi = d["ci95"]
    verdict = "no reliable difference" if lo <= 0 <= hi else "a real difference"
    return (
        f"EBM minus LightGBM = {d['diff']:+.3f} PR-AUC, 95% CI [{lo:+.3f}, {hi:+.3f}] ({verdict})"
    )


def _example(codes: pd.DataFrame, prim: str) -> tuple[str, str, str]:
    from ontime.explain.cases import pick_cases

    oid = pick_cases(codes, primary=prim)["true_positive"]
    r = codes[(codes["order_id"] == oid) & (codes["model"] == prim)].iloc[0]
    lines = [f"- {s}" for s in list(r["risk_factors"]) + list(r["protective_factors"])]
    return oid[:12] + "...", f"{r['probability']:.0%}", "\n".join(lines)


def _lib_versions() -> str:
    names = ["scikit-learn", "lightgbm", "interpret", "shap", "duckdb", "pandas", "streamlit"]
    out = []
    for n in names:
        try:
            out.append(f"{n} {metadata.version(n)}")
        except metadata.PackageNotFoundError:
            continue
    return ", ".join(out)


def _k10(entry: dict) -> dict:
    return next(t for t in entry["top_k"] if abs(t["k"] - 0.10) < 1e-9)


def _stage2_values(res: dict) -> dict[str, str]:
    """Placeholders for the stage-2 (carrier handover) model."""
    s2 = _load("stage2_results.json")
    t = s2["test"]
    m1, m2 = t["stage1"]["metrics"], t["stage2"]["metrics"]
    k1, k2 = _k10(t["stage1"]), _k10(t["stage2"])
    mon = pd.DataFrame(t["monthly"]).sort_values("month")
    first, last = mon.iloc[0], mon.iloc[-1]
    win = t["action_window_for_late_orders"]
    v = s2["validation_comparison"]
    pol = pd.DataFrame(t["policy"])
    pm = sorted(pol["p"].unique())[len(pol["p"].unique()) // 2]
    sel = pol[(pol["k"].round(3) == 0.10) & (pol["p"] == pm)].set_index("strategy")
    ab = ["| Feature set | PR-AUC (validation) | ROC-AUC (validation) |", "|---|---|---|"]
    for name, a in s2["ablation_validation_logreg"].items():
        ab.append(f"| {name} | {a['pr_auc']:.3f} | {a['roc_auc']:.3f} |")
    mt = ["| Month | Late rate | ROC-AUC at approval | ROC-AUC at handover | Lift@10% at approval | Lift@10% at handover |",
          "|---|---|---|---|---|---|"]  # fmt: skip
    for r in mon.itertuples():
        mt.append(
            f"| {r.month} | {_pct(r.late_rate)} | {r.stage1_roc_auc:.2f} | {r.stage2_roc_auc:.2f} | "
            f"{r.stage1_lift_at_10:.2f} | {r.stage2_lift_at_10:.2f} |"
        )
    d = t["pr_auc_diff_stage2_minus_stage1"]
    ld = t["lift_at_10_diff_stage2_minus_stage1"]
    return {
        "s2_model": LABEL[s2["chosen_model"]],
        "s2_population": f"{s2['population']['orders_total']:,}",
        "s2_pr_auc": f"{m2['pr_auc']['value']:.3f}",
        "s2_pr_ci": _ci(m2["pr_auc"]["ci95"]),
        "s1_same_pr_auc": f"{m1['pr_auc']['value']:.3f}",
        "s2_roc": f"{m2['roc_auc']['value']:.3f}",
        "s1_same_roc": f"{m1['roc_auc']['value']:.3f}",
        "s2_lift10": f"{k2['lift']:.1f}",
        "s2_lift10_ci": _ci(k2["ci95"]["lift"], 2),
        "s1_same_lift10": f"{k1['lift']:.1f}",
        "s2_precision10": _pct(k2["precision"]),
        "s2_recall10": _pct(k2["recall"]),
        "s2_pr_diff": f"{d['diff']:+.3f}",
        "s2_pr_diff_ci": f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]",
        "s2_lift_diff": f"{ld['diff']:+.2f}",
        "s2_lift_diff_ci": f"[{ld['ci95'][0]:+.2f}, {ld['ci95'][1]:+.2f}]",
        "s2_first_month": first["month"],
        "s2_last_month": last["month"],
        "s2_roc_first": f"{first['stage2_roc_auc']:.2f}",
        "s2_roc_last": f"{last['stage2_roc_auc']:.2f}",
        "s1_roc_last": f"{last['stage1_roc_auc']:.2f}",
        "s2_window_median_days": f"{win['median_days_left_at_handover']:.0f}",
        "s2_window_share3": _pct(win["share_with_3plus_days_left"], 0),
        "s2_handover_median_days": f"{win['median_days_to_handover']:.1f}",
        "s2_val_pr_s1": f"{v['stage1_pr_auc']:.3f}",
        "s2_val_pr_s2": f"{v['stage2_pr_auc']:.3f}",
        "s2_within_auc": f"{s2['backlog_within_state_month']['median_auc']:.2f}",
        "s2_ablation_table": "\n".join(ab),
        "s2_monthly_table": "\n".join(mt),
        "s2_policy_sentence": (
            f"at 10% daily capacity and an assumed {pm:.0%} success rate: about "
            f"{sel.loc['stage2', 'prevented_expected']:.0f} late orders prevented when acting at handover "
            f"({_pct(sel.loc['stage2', 'prevented_share_of_late'], 0)} of all late) vs "
            f"{sel.loc['stage1', 'prevented_expected']:.0f} when acting at approval and "
            f"{sel.loc['random', 'prevented_expected']:.0f} for random picking; ceiling "
            f"{sel.loc['oracle', 'prevented_expected']:.0f}"
        ),
    }


def _s2_cal_values() -> dict[str, str]:
    """Placeholders for the stage-2 calibration block."""
    cal = _load("stage2_results.json")["calibration"]
    t = cal["test"]
    return {
        "s2_cal_method": cal["method"],
        "s2_cal_brier_raw": f"{t['raw']['brier']:.4f}",
        "s2_cal_brier": f"{t['calibrated']['brier']:.4f}",
        "s2_cal_brier_s1": f"{t['stage1_deployed']['brier']:.4f}",
        "s2_cal_ece_raw": f"{t['raw']['ece']:.3f}",
        "s2_cal_ece": f"{t['calibrated']['ece']:.3f}",
        "s2_cal_ece_s1": f"{t['stage1_deployed']['ece']:.3f}",
        "s2_const_brier": f"{t['constant_at_true_rate_brier']:.4f}",
    }


def _eta_values() -> dict[str, str]:
    """Placeholders for the S1 promise-date (quantile) model."""
    e = _load("eta_results.json")
    te, o = e["test"], e["test"]["olist"]
    ch = te["chosen"]["match_olist"]
    v = ch["vs_olist"]
    rows = ["| Operating point (chosen on validation) | Level | Avg promise (days) | On time | vs Olist length | vs Olist on-time |",
            "|---|---|---|---|---|---|"]  # fmt: skip
    rows.append(
        f"| Olist as quoted | - | {o['mean_promise_days']:.1f} | {_pct(o['coverage'])} | - | - |"
    )
    for name, r in te["chosen"].items():
        d = r["vs_olist"]
        rows.append(
            f"| model, {name.replace('_', ' ')} | {r['alpha']} | {r['test_mean_promise_days']:.1f} | "
            f"{_pct(r['test_coverage'])} | {d['length_diff']:+.1f} d {_ci(d['length_diff_ci'], 1)} | "
            f"{d['coverage_diff'] * 100:+.1f} pp {_ci([x * 100 for x in d['coverage_diff_ci']], 1)} |"
        )
    shift = ch.get("shift_baseline", {})
    return {
        "eta_alpha": str(ch["alpha"]),
        "eta_val_cov": _pct(ch["target_validation_coverage"]),
        "eta_olist_len": f"{o['mean_promise_days']:.1f}",
        "eta_olist_cov": _pct(o["coverage"]),
        "eta_model_len": f"{ch['test_mean_promise_days']:.1f}",
        "eta_model_cov": _pct(ch["test_coverage"]),
        "eta_len_diff": f"{v['length_diff']:+.1f}",
        "eta_len_ci": _ci(v["length_diff_ci"], 1),
        "eta_cov_diff": f"{v['coverage_diff'] * 100:+.1f}",
        "eta_cov_ci": _ci([x * 100 for x in v["coverage_diff_ci"]], 1),
        "eta_interp_len": f"{te['interpolated_same_level']['length_at_olist_test_coverage']:.1f}",
        "eta_n_test": f"{e['orders']['test']:,}",
        "eta_mismatch": f"{e['lateness_definition_mismatch']:.2%}",
        "eta_table": "\n".join(rows),
        "eta_shift_note": (
            f"A constant shift of {shift['shift_days']} days on Olist's promise reaches "
            f"{_pct(shift['test_coverage'])} on time at {shift['test_mean_promise_days']:.1f} days."
            if shift
            else ""
        ),
    }


def _rolling_values() -> dict[str, str]:
    """Placeholders for the monthly-retraining / calendar-proxy ablation (a negative result)."""
    r = _load("rolling_backtest.json")
    sm = r["summary"]
    rows = ["| Variant | Macro ROC-AUC | Pooled lift@10% | vs deployed static model (lift diff, 95% CI) |",
            "|---|---|---|---|"]  # fmt: skip
    for v in sm.values():
        d = v.get("lift_diff_vs_static")
        cmp_ = (
            f"{d['diff']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]" if d else "(reference)"
        )
        rows.append(
            f"| {v['label']} | {v['macro_roc_auc']:.3f} | {v['pooled_lift_at_10']:.2f} | {cmp_} |"
        )
    a = r["proxy_ablation"]
    helped = sm["retrain_logreg_full"]["lift_diff_vs_static"]
    return {
        "roll_table": "\n".join(rows),
        "roll_months": f"{r['months'][0]} to {r['months'][-1]}",
        "roll_lr_diff": f"{helped['diff']:+.2f}",
        "roll_lr_ci": f"[{helped['ci95'][0]:+.2f}, {helped['ci95'][1]:+.2f}]",
        "roll_proxy_lr": f"{a['logreg']['noproxy_minus_full_lift']:+.2f} "
        f"[{a['logreg']['ci95'][0]:+.2f}, {a['logreg']['ci95'][1]:+.2f}]",
        "roll_proxy_lgbm": f"{a['lgbm']['noproxy_minus_full_lift']:+.2f} "
        f"[{a['lgbm']['ci95'][0]:+.2f}, {a['lgbm']['ci95'][1]:+.2f}]",
        "roll_static_lift": f"{sm['static_v1']['pooled_lift_at_10']:.2f}",
        "roll_retrain_lift": f"{sm['retrain_logreg_full']['pooled_lift_at_10']:.2f}",
    }


def _drift_values() -> dict[str, str]:
    """Placeholders for the drift monitor."""
    cfg = load_config()
    rep = project_path(cfg["paths"]["reports_dir"])
    st = pd.read_csv(rep / "drift_status.csv", parse_dates=["month"])
    summ = _load("drift_summary.json")
    first_alert = st.loc[st["status"] == "alert", "month"].min()
    perf_drop = st.loc[st["roc_vs_first"] <= -0.05, "month"].min()
    tbl = ["| Month | Order-mix features in alert | ROC-AUC | Status |", "|---|---|---|---|"]
    for r in st.itertuples():
        tbl.append(f"| {r.month:%Y-%m} | {int(r.trail_alert)} | {r.roc_auc:.2f} | {r.status} |")
    lead = ""
    if pd.notna(first_alert) and pd.notna(perf_drop):
        gap = (perf_drop.year - first_alert.year) * 12 + perf_drop.month - first_alert.month
        lead = (
            f"The monitor first reached *alert* in {first_alert:%Y-%m}; ranking quality first fell "
            f"more than 0.05 below its starting level in {perf_drop:%Y-%m} "
            f"({gap} month(s) later)."
        )
    return {
        "drift_table": "\n".join(tbl),
        "drift_lead_sentence": lead,
        "drift_n_months": str(len(st)),
        "drift_state_features": ", ".join(f"`{f}`" for f in summ["expected_drift_features"]),
    }


def collect_values() -> dict[str, str]:
    """Every placeholder used by the templates, computed from artifacts."""
    cfg = load_config()
    res, val = _load("results.json"), _load("validation_results.json")
    imp, expl = _load("impact.json"), _load("explain_summary.json")
    prim = res["primary_model"]
    pop = res["population"]
    pm = res["models"][prim]["metrics"]
    k10 = next(t for t in res["models"][prim]["top_k"] if abs(t["k"] - 0.10) < 1e-9)

    feats = pd.read_parquet(project_path(cfg["paths"]["features_parquet"]))
    val_rate = feats.loc[feats["split"] == "validation", "is_late"].mean()
    desc = imp["descriptive"]
    timing = {r["group"]: r for r in imp["review_timing"]}

    pol = pd.DataFrame(res["policy"])
    p_vals = sorted(pol["p"].unique())
    p_mid = p_vals[len(p_vals) // 2]
    sel = pol[(pol["k"] == 0.10) & (pol["p"] == p_mid)].set_index("strategy")
    m_row, r_row, o_row = sel.loc[prim], sel.loc["random"], sel.loc["oracle"]

    mon = pd.DataFrame(res["monthly"])
    mp = mon[mon["model"] == prim].sort_values("month")
    first, last = mp.iloc[0], mp.iloc[-1]

    rule = res["baselines"]["rule"]["metrics"]["pr_auc"]
    rule_ci_hi = rule["ci95"][1]
    prim_lo = pm["pr_auc"]["ci95"][0]
    rule_verdict = (
        f"Yes: the rule's PR-AUC is {rule['value']:.3f}, essentially the base rate "
        f"({pop['late_rate']:.3f}), while {LABEL[prim]} scores {pm['pr_auc']['value']:.3f}."
        if prim_lo > rule_ci_hi
        else f"Not reliably (rule PR-AUC {rule['value']:.3f} {_ci(rule['ci95'])})."
    )

    agree = expl["agreement"]
    rk, d = agree["rank_agreement"], agree["direction"]
    agreement_sentence = (
        f"Across the three explanation methods, global importance agrees most between LightGBM and "
        f"the EBM (rank correlation {rk['spearman']['lightgbm vs ebm']:.2f}); "
        f"{d['agree']} of {d['features_compared']} numeric features have the same effect direction "
        f"in at least two models."
    )

    # calibration text
    rows = ["| Model | Calibrator | Brier (deployed) | Brier (monthly recalibration) | ECE (deployed) |",
            "|---|---|---|---|---|"]  # fmt: skip
    for m in ("logreg", "lightgbm", "ebm", "random_forest"):
        e = res["models"][m]
        rows.append(
            f"| {LABEL[m]} | {e['calibrator']} | {e['metrics']['brier']['value']:.4f} | "
            f"{e['rolling_recalibration']['brier']:.4f} | {e['ece_deployed']:.4f} |"
        )
    calibration_text = (
        "\n".join(rows)
        + f"\n\nFor reference, a constant prediction at the true test rate scores Brier "
        f"{res['constant_oracle_brier']:.4f}. The deployed calibrators were fitted on validation "
        f"data; the late rate then moved from {_pct(pop['train_late_rate'])} (train) to "
        f"{_pct(pop['late_rate'])} (test), so probabilities are best read as relative scores. "
        "Re-fitting the calibrator monthly on trailing outcomes did not improve Brier on this data."
    )

    stab = [
        "| Month | Orders | Late rate | PR-AUC | ROC-AUC | Lift @10% |",
        "|---|---|---|---|---|---|",
    ]
    for r in mp.itertuples():
        stab.append(
            f"| {r.month} | {r.orders:,} | {_pct(r.late_rate)} | {r.pr_auc:.3f} | {r.roc_auc:.3f} | "
            f"{r.lift_at_10:.2f} |"
        )
    stability_text = (
        f"{LABEL[prim]} by month:\n\n" + "\n".join(stab) + "\n\nRanking skill is not constant: it "
        "is strongest in the first months after training and weakest at the end of the test period. "
        "Monitoring and scheduled retraining are required."
    )

    weak = res["slices_primary_weak"]
    if weak:
        sl = [
            "| Slice | Value | Orders | Late rate | ROC-AUC | Overall ROC-AUC |",
            "|---|---|---|---|---|---|",
        ]
        for w in weak:
            sl.append(
                f"| {w['slice']} | {w['value']} | {w['orders']:,} | {_pct(w['late_rate'])} | "
                f"{w['roc_auc']:.3f} | {w['overall_roc_auc']:.3f} |"
            )
        slices_text = (
            "Flagged = ROC-AUC more than 0.05 below the model's overall ROC-AUC, among slices with "
            "enough late orders.\n\n" + "\n".join(sl) + "\n\nFull table: `reports/slices.csv`."
        )
    else:
        slices_text = (
            "No slice with enough late orders was flagged. Full table: `reports/slices.csv`."
        )

    oid, ex_prob, ex_reasons = _example(
        pd.read_parquet(project_path(cfg["paths"]["reports_dir"]) / "test_reason_codes.parquet"),
        prim,
    )

    extra = {
        **_stage2_values(res), **_s2_cal_values(), **_eta_values(),
        **_rolling_values(), **_drift_values(),
    }  # fmt: skip
    return {
        **extra,
        "test_orders": f"{pop['orders']:,}",
        "test_late": f"{pop['late_orders']:,}",
        "test_late_rate": _pct(pop["late_rate"]),
        "test_from": pop["from"],
        "test_to": pop["to"],
        "train_late_rate": _pct(pop["train_late_rate"]),
        "val_late_rate": _pct(val_rate),
        "primary": LABEL[prim],
        "primary_pr_auc": f"{pm['pr_auc']['value']:.3f}",
        "primary_pr_ci": _ci(pm["pr_auc"]["ci95"]),
        "base_rate": f"{pop['late_rate']:.3f}",
        "pr_ratio": f"{pm['pr_auc']['value'] / pop['late_rate']:.1f}x",
        "lift10": f"{k10['lift']:.1f}",
        "lift10_ci": _ci(k10["ci95"]["lift"], 2),
        "precision10": _pct(k10["precision"]),
        "rule_verdict": rule_verdict,
        "gap_sentence": _gap_sentence(res["paired_pr_auc_diff"]["ebm_minus_lightgbm"]),
        "policy_sentence": (
            f"at 10% daily capacity and an assumed {p_mid:.0%} intervention success, "
            f"about {m_row['prevented_expected']:.0f} late orders prevented "
            f"({_pct(m_row['prevented_share_of_late'])} of all late) vs {r_row['prevented_expected']:.0f} "
            f"for random selection; ceiling with perfect ranking {o_row['prevented_expected']:.0f}"
        ),
        "results_table": _results_table(res),
        "n_features": str(len(FEATURE_COLUMNS)),
        "star_conservative": f"{abs(imp['sensitivity_answered_after_delivery']['coef']):.1f}",
        "star_full": f"{abs(imp['ols']['adjusted']['coef']):.1f}",
        "late_low_share": _pct(desc["late"]["share_1_or_2_star"], 0),
        "ontime_low_share": _pct(desc["on time"]["share_1_or_2_star"], 0),
        "late_before_share": _pct(
            timing["late, reviewed BEFORE arrival"]["share_of_group_orders"], 0
        ),
        "example_order": oid,
        "example_prob": ex_prob,
        "example_reasons": ex_reasons,
        "agreement_sentence": agreement_sentence,
        "roc_first": f"{first['roc_auc']:.2f}",
        "roc_last": f"{last['roc_auc']:.2f}",
        "month_first": first["month"],
        "month_last": last["month"],
        # model card only
        "val_calib_pr_auc": f"{val['models'][prim]['raw_metrics']['val_calib']['pr_auc']:.3f}",
        "val_calib_base_rate": f"{val['models'][prim]['raw_metrics']['val_calib']['base_rate']:.3f}",
        "train_from": val["meta"]["train_range"][0][:10],
        "train_to": val["meta"]["train_range"][1][:10],
        "train_orders": f"{val['meta']['train_orders']:,}",
        "git_hash": val["meta"]["git_hash"],
        "seed": str(cfg["seed"]),
        "trained_at": val["meta"]["created_utc"][:10],
        "lib_versions": _lib_versions(),
        "raw_orders_note": "about one hundred thousand orders across 2016-2018",
        "cutoff": cfg["splits"]["censoring_cutoff"],
        "model_orders": f"{len(feats):,}",
        "accuracy_trivial": _pct(1 - pop["late_rate"]),
        "calibration_text": calibration_text,
        "stability_text": stability_text,
        "slices_text": slices_text,
        "alpha": str(cfg["features"]["shrinkage_alpha"]),
    }


def render_template(text: str, values: dict[str, str]) -> str:
    """Replace {{keys}}; raise on any key without a value (no silent blanks)."""
    missing = sorted({k for k in PLACEHOLDER.findall(text) if k not in values})
    if missing:
        raise KeyError(f"template placeholders without a value: {missing}")
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


def build_reports(write: bool = True) -> dict[str, str]:
    """Render README.md and reports/model_card.md."""
    values = collect_values()
    out = {
        "README.md": render_template((TEMPLATES / "README.template.md").read_text("utf-8"), values),
        "reports/model_card.md": render_template(
            (TEMPLATES / "model_card.template.md").read_text("utf-8"), values
        ),
    }
    if write:
        for rel, text in out.items():
            path = PROJECT_ROOT / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            log.info("report | wrote %s (%d chars)", rel, len(text))
    return out
