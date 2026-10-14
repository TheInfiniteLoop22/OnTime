"""Phase 8 orchestration: global explanations, agreement report, reason codes for every test order.

Global analyses use the VALIDATION set (so explanations are not fitted to test behaviour);
per-order reason codes are produced for the TEST set after the final evaluation, to read
alongside its error analysis. Nothing here changes a model or a decision.
"""

from __future__ import annotations

import json
import logging
import warnings

import joblib
import numpy as np
import pandas as pd

from ontime.config import load_config, project_path
from ontime.explain import agreement, ebm_explain, shap_explain
from ontime.explain.contributions import EXTRACTORS
from ontime.explain.reason_codes import (
    add_display_fields,
    build_reason_codes,
    feature_to_group,
    group_contributions,
    load_templates,
    validate_templates,
)
from ontime.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from ontime.models.data import load_feature_table, prepare_frame
from ontime.models.splits import select_split

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*constant.*")
warnings.filterwarnings("ignore", message=".*TreeExplainer shap values output has changed.*")
MODELS = ["logreg", "lightgbm", "ebm"]


def _raw_logit(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    p = np.clip(artifact["model"].predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _contribs(artifacts: dict, X: pd.DataFrame, check: bool = True) -> dict:
    """{model: contributions DataFrame}, verifying sum + intercept == raw log-odds."""
    out = {}
    for m in MODELS:
        c, base = EXTRACTORS[m](artifacts[m]["model"], X)
        if check:
            err = np.abs(c.sum(axis=1).to_numpy() + base - _raw_logit(artifacts[m], X)).max()
            log.info("explain | %-8s additive identity max error = %.2e", m, err)
            if err > 1e-4:
                raise RuntimeError(f"{m}: contributions do not add up to the model output ({err})")
        out[m] = c
    return out


def run_explain(write: bool = True) -> dict:
    cfg = load_config()
    rep = project_path(cfg["paths"]["reports_dir"])
    fig = rep / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    table = load_feature_table()
    arts = {
        m: joblib.load(project_path(cfg["paths"]["models_dir"]) / f"{m}.joblib") for m in MODELS
    }
    levels = arts["logreg"]["levels"]

    templates = load_templates()
    problems = validate_templates(templates)
    if problems:
        raise ValueError("reason template problems:\n" + "\n".join(problems))
    f2g = feature_to_group(templates)

    # ---------------- global explanations on validation ----------------
    val = select_split(table, "validation")
    Xv = prepare_frame(val, levels)
    cv = _contribs(arts, Xv)
    numeric = [f for f in FEATURE_COLUMNS if f not in CATEGORICAL_FEATURES]

    imps = {m: agreement.global_importance(cv[m]) for m in MODELS}
    imp_tbl = pd.DataFrame(imps)
    imp_tbl.to_csv(rep / "global_importance_features.csv")
    grp_tbl = pd.DataFrame({m: group_contributions(cv[m], f2g).abs().mean() for m in MODELS})
    grp_tbl.sort_values("logreg", ascending=False).to_csv(rep / "global_importance_groups.csv")

    # EBM
    ebm = arts["ebm"]["model"]
    terms = ebm_explain.term_importance_table(ebm)
    terms.to_csv(rep / "ebm_term_importance.csv", index=False)
    ebm_explain.plot_importance(terms, fig / "explain_ebm_importance.png")
    Xtr = prepare_frame(select_split(table, "train"), levels)  # x-range from the TRAINING domain
    multi_valued = [f for f in numeric if Xtr[f].nunique() > 2]  # skip binary flags in the grid
    main_cont = [t for t in terms["term"] if " & " not in t and t in multi_valued][:6]
    ebm_explain.plot_shapes(ebm, main_cont, Xtr, fig / "explain_ebm_shapes.png")
    top_int = terms[terms["kind"] == "interaction"].iloc[0]["term"]
    ebm_explain.plot_interaction(ebm, top_int, fig / "explain_ebm_interaction.png")

    # SHAP (LightGBM) on a seeded sample for readable plots
    samp = Xv.sample(min(5000, len(Xv)), random_state=seed)
    shap_vals = cv["lightgbm"].loc[samp.index].to_numpy()
    shap_explain.beeswarm(shap_vals, samp, fig / "explain_shap_beeswarm.png")
    shap_explain.bar(shap_vals, samp, fig / "explain_shap_bar.png")
    top_shap_num = [f for f in imps["lightgbm"].index if f in numeric][:4]
    shap_explain.dependence(shap_vals, samp, top_shap_num, fig / "explain_shap_dependence.png")

    # Agreement
    ranks = agreement.rank_agreement(imps, top_k=10)
    dirs = agreement.direction_table(cv, Xv, numeric, val["is_late"].to_numpy())
    dirs.to_csv(rep / "explain_direction_agreement.csv", index=False)
    used = dirs[dirs["models_with_clear_direction"] >= 2]
    agree_summary = {
        "rank_agreement": ranks,
        "direction": {
            "features_compared": int(len(used)),
            "agree": int(used["agree"].sum()),
            "conflict": int(used["conflict"].sum()),
            "conflicting_features": used.loc[used["conflict"], "feature"].tolist(),
            "differs_from_marginal": _differs_from_marginal(used, MODELS),
        },
        "top10_by_model": {m: imps[m].head(10).index.tolist() for m in MODELS},
        "ebm_top_interaction": top_int,
        "ebm_interaction_share_of_importance": float(
            terms.loc[terms["kind"] == "interaction", "share_of_total"].sum()
        ),
    }

    # ---------------- reason codes on test ----------------
    test = select_split(table, "test", final_evaluation=True).reset_index(drop=True)
    Xt = prepare_frame(test, levels)
    ct = _contribs(arts, Xt)
    display = add_display_fields(Xt)
    frames = []
    for m in MODELS:
        grouped = group_contributions(ct[m], f2g)
        rc = build_reason_codes(grouped, display, templates)
        raw = arts[m]["model"].predict_proba(Xt)[:, 1]
        rc.insert(0, "probability", arts[m]["calibrator"].transform(raw))
        rc.insert(0, "model", m)
        rc.insert(0, "is_late", test["is_late"].to_numpy())
        rc.insert(0, "order_id", test["order_id"].to_numpy())
        frames.append(rc.reset_index(drop=True))
        log.info("explain | reason codes built for %s (%d orders)", m, len(rc))
    codes = pd.concat(frames, ignore_index=True)
    codes.to_parquet(rep / "test_reason_codes.parquet", index=False)

    # 10 readable samples of the primary model for owner review (seeded, stratified by outcome)
    prim = codes[codes["model"] == "logreg"].reset_index(drop=True)
    pick = pd.concat(
        [
            prim[prim["is_late"] == 1].nlargest(3, "probability"),
            prim[prim["is_late"] == 0].nlargest(2, "probability"),
            prim[prim["is_late"] == 1].nsmallest(2, "probability"),
            prim.sample(3, random_state=seed),
        ]
    )
    lines = ["# Reason-code samples for review (logistic regression, test set)", ""]
    for _, r in pick.iterrows():
        lines += [
            f"### Order {r['order_id']} - predicted {r['probability']:.1%} - actually "
            f"{'LATE' if r['is_late'] else 'on time'}",
            *(f"- {s}" for s in r["risk_factors"]),
            *(f"- {s}" for s in r["protective_factors"]),
            "",
        ]
    (rep / "reason_code_samples.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "agreement": agree_summary,
        "reason_codes": {
            "orders": int(len(test)),
            "models": MODELS,
            "share_with_at_least_one_risk_factor": {
                m: float((codes[codes["model"] == m]["risk_factors"].map(len) > 0).mean())
                for m in MODELS
            },
        },
    }
    if write:
        (rep / "explain_summary.json").write_text(
            json.dumps(summary, indent=2, default=float), encoding="utf-8"
        )
        _write_agreement_md(rep / "explain_agreement.md", summary, dirs, terms)
    return summary


def _differs_from_marginal(dirs: pd.DataFrame, models: list[str]) -> list[str]:
    """Features where some model has a clear direction opposite to the marginal direction."""
    out = []
    for _, r in dirs.iterrows():
        if r["marginal"] == "flat/non-monotone":
            continue
        for m in models:
            if (
                r[m] in ("higher value -> more risk", "higher value -> less risk")
                and r[m] != r["marginal"]
            ):
                out.append(r["feature"])
                break
    return out


def _write_agreement_md(path, summary: dict, dirs: pd.DataFrame, terms: pd.DataFrame) -> None:  # noqa: ANN001
    a = summary["agreement"]
    L = ["# Explanation agreement (generated by `ontime.explain.run`)", ""]
    L += [
        "## Global importance: rank agreement (validation set)",
        "",
        "| pair | Spearman | top-10 overlap |",
        "|---|---|---|",
    ]
    for k, v in a["rank_agreement"]["spearman"].items():
        L.append(f"| {k} | {v:.2f} | {a['rank_agreement']['top_k_overlap'][k]:.0%} |")
    L += ["", "## Top-10 features by model", ""]
    for m, f in a["top10_by_model"].items():
        L.append(f"- **{m}**: " + ", ".join(f))
    d = a["direction"]
    L += [
        "",
        "## Direction of effect (numeric features)",
        f"Among {d['features_compared']} numeric features with a clear direction in at least two "
        f"models, **{d['agree']}** agree and **{d['conflict']}** conflict.",
        "",
        "Conflicting features: " + (", ".join(d["conflicting_features"]) or "none"),
        "Differs from marginal: " + (", ".join(d["differs_from_marginal"]) or "none"),
        "",
        f"Features where at least one model's direction differs from the one-feature-at-a-time "
        f"(marginal) direction: **{d['differs_from_marginal']}**. These are typically corrections "
        "for correlated features (e.g. several late-rate features computed from the same events), "
        "not stand-alone effects, and should not be read causally.",
        "",
        dirs[dirs["models_with_clear_direction"] >= 2][
            ["feature", "marginal", "logreg", "lightgbm", "ebm", "agree"]
        ].to_markdown(index=False),
        "",
        "## EBM interactions",
        f"Pairwise terms carry {a['ebm_interaction_share_of_importance']:.1%} of EBM importance; "
        f"the strongest is `{a['ebm_top_interaction']}`.",
    ]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
