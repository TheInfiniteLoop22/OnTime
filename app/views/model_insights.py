"""Model insights: how good, how honest, where weak - with intervals and without hype."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ontime.evaluation.calibration import reliability_table
from ontime.serving.service import MODEL_LABELS, MODELS
from ontime_ui import components as c
from ontime_ui import data, theme

theme.inject_css()
data.require_artifacts()

res = data.read_json("results.json")
val = data.read_json("validation_results.json")
prim = res["primary_model"]
pop = res["population"]
LABEL = {
    **MODEL_LABELS,
    "random_forest": "Random forest",
    "rule": "Hand-written rule",
    "prior_rate": "Guess the training rate",
}
ALL = ["prior_rate", "rule", "logreg", "random_forest", "lightgbm", "ebm"]


def entry(name: str) -> dict:
    return res["baselines"][name] if name in res["baselines"] else res["models"][name]


c.hero(
    "Model insights",
    "Everything here was measured on a later period the models never saw. Intervals are 95% "
    "bootstrap intervals. The primary model was fixed by validation results before the test set was opened.",
)

tabs = st.tabs(
    [
        "Performance",
        "Stage 2: at handover",
        "Calibration",
        "Stability",
        "Explainability",
        "Where it is weak",
        "Limits",
    ]
)

# ------------------------------------------------------------------ performance
with tabs[0]:
    rows = []
    for n in ALL:
        e = entry(n)
        m = e["metrics"]
        k10 = next(t for t in e["top_k"] if abs(t["k"] - 0.10) < 1e-9)
        rows.append(
            {
                "key": n,
                "Model": LABEL[n],
                "PR-AUC": m["pr_auc"]["value"],
                "lo": m["pr_auc"]["ci95"][0],
                "hi": m["pr_auc"]["ci95"][1],
                "ROC-AUC": m["roc_auc"]["value"],
                "Brier": m["brier"]["value"],
                "Log loss": m["log_loss"]["value"],
                "Lift @10%": k10["lift"],
                "Precision @10%": k10["precision"],
                "Recall @10%": k10["recall"],
            }
        )
    perf = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=perf["PR-AUC"],
            y=perf["Model"],
            mode="markers",
            name="PR-AUC (95% CI)",
            marker={"size": 12, "color": [theme.MODEL_COLORS[k] for k in perf["key"]]},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": perf["hi"] - perf["PR-AUC"],
                "arrayminus": perf["PR-AUC"] - perf["lo"],
                "color": theme.INK2,
            },
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=pop["late_rate"],
        line_dash="dash",
        line_color=theme.MUTED,
        annotation_text=f"Random ranking = base rate {pop['late_rate']:.3f}",
        annotation_position="top right",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="PR-AUC on the test period (higher is better)", rangemode="tozero")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(theme.style_fig(fig, 360, legend=False), width="stretch")
    c.caption(
        "Look at the distance from the dashed line: that is skill over guessing. "
        "Overlapping bars are not reliably different."
    )

    show = perf.drop(columns=["key", "lo", "hi"]).copy()
    show.insert(
        2, "95% CI", [f"[{a:.3f}, {b:.3f}]" for a, b in zip(perf["lo"], perf["hi"], strict=True)]
    )
    for col, nd in [("PR-AUC", 3), ("ROC-AUC", 3), ("Lift @10%", 2)]:
        show[col] = show[col].map(lambda v, nd=nd: f"{v:.{nd}f}")
    show["Brier"] = show["Brier"].map(lambda v: f"{v:.4f}")
    show["Log loss"] = show["Log loss"].map(lambda v: f"{v:.3f}")
    show["Precision @10%"] = show["Precision @10%"].map(c.pct)
    show["Recall @10%"] = show["Recall @10%"].map(c.pct)
    st.dataframe(show, hide_index=True, width="stretch")

    st.markdown("**Glass-box vs black-box**")
    for key, label in [
        ("ebm_minus_lightgbm", "EBM minus LightGBM"),
        ("logreg_minus_lightgbm", "Logistic regression minus LightGBM"),
        ("ebm_minus_logreg", "EBM minus logistic regression"),
    ]:
        d = res["paired_pr_auc_diff"][key]
        lo, hi = d["ci95"]
        verdict = "a real difference" if (lo > 0 or hi < 0) else "no reliable difference"
        st.markdown(
            f"- **{label}:** {d['diff']:+.3f} PR-AUC, 95% CI [{lo:+.3f}, {hi:+.3f}] - {verdict}"
        )
    g = res["paired_pr_auc_diff"]["ebm_minus_lightgbm"]
    if g["ci95"][0] <= 0 <= g["ci95"][1]:
        best = perf.loc[perf["PR-AUC"].idxmax(), "Model"]
        c.callout(
            "<b>Verdict:</b> choosing the interpretable EBM over LightGBM costs no reliable accuracy "
            f"on this data. The highest PR-AUC belongs to <b>{best}</b>."
        )
    else:
        c.callout("<b>Verdict:</b> the EBM and LightGBM differ reliably; see the interval above.")

# ------------------------------------------------------------------ stage 2
with tabs[1]:
    s2_path = data.REPORTS / "stage2_results.json"
    if not s2_path.exists():
        st.info("Stage-2 results are missing. Run `make stage2` and reload.")
    else:
        s2 = data.read_json("stage2_results.json")
        t2 = s2["test"]
        m1, m2 = t2["stage1"]["metrics"], t2["stage2"]["metrics"]
        k1 = next(t for t in t2["stage1"]["top_k"] if abs(t["k"] - 0.10) < 1e-9)
        k2 = next(t for t in t2["stage2"]["top_k"] if abs(t["k"] - 0.10) < 1e-9)
        win = t2["action_window_for_late_orders"]
        st.markdown(
            "**Idea:** re-score each order when the seller **hands it to the carrier** (usually within a "
            "couple of days of approval). Everything known at approval is still used; the handover "
            "timestamp and a destination *backlog* signal (earlier orders still in transit) are added. "
            "The customer delivery date is still never used."
        )
        a, b, c3, d = st.columns(4)
        with a:
            c.kpi(
                "PR-AUC at approval",
                f"{m1['pr_auc']['value']:.3f}",
                f"base rate {t2['late_rate']:.3f}",
            )
        with b:
            c.kpi(
                "PR-AUC at handover",
                f"{m2['pr_auc']['value']:.3f}",
                f"95% CI [{m2['pr_auc']['ci95'][0]:.3f}, {m2['pr_auc']['ci95'][1]:.3f}]",
            )
        with c3:
            c.kpi("Lift at 10% capacity", f"{k2['lift']:.1f}x", f"vs {k1['lift']:.1f}x at approval")
        with d:
            c.kpi(
                "Time left to act",
                f"{win['median_days_left_at_handover']:.0f} days",
                f"median before the promised date for late orders; {win['share_with_3plus_days_left']:.0%} "
                "have 3+ days left",
            )
        st.write("")
        mm = pd.DataFrame(t2["monthly"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=mm["month"],
                y=mm["stage1_roc_auc"],
                mode="lines+markers",
                name="At approval (stage 1)",
                line={"color": theme.BLUE, "width": 2},
                marker={"size": 8},
                hovertemplate="%{y:.2f}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mm["month"],
                y=mm["stage2_roc_auc"],
                mode="lines+markers",
                name="At handover (stage 2)",
                line={"color": theme.ORANGE, "width": 2},
                marker={"size": 8},
                hovertemplate="%{y:.2f}",
            )
        )
        fig.add_hline(
            y=0.5, line_dash="dash", line_color=theme.MUTED, annotation_text="random = 0.5"
        )
        fig.update_yaxes(title_text="ROC-AUC by month", range=[0.4, 0.9])
        fig.update_xaxes(type="category")
        st.plotly_chart(theme.style_fig(fig, 340), width="stretch")
        c.caption(
            "Look at the last months: the approval-time model decays while the handover-time model "
            "holds, because it can see the backlog building before outcomes arrive."
        )

        ab = pd.DataFrame(
            [
                {
                    "Feature set": k,
                    "PR-AUC (validation)": v["pr_auc"],
                    "ROC-AUC (validation)": v["roc_auc"],
                }
                for k, v in s2["ablation_validation_logreg"].items()
            ]
        )
        st.markdown("**What each block adds (logistic regression, validation period only)**")
        st.dataframe(ab.round(3), hide_index=True, width="stretch")
        w = s2["backlog_within_state_month"]
        c.caption(
            f"The backlog says *when and where* trouble is, not which order: its median AUC within a single "
            f"state-month is only {w['median_auc']:.2f} (over {w['groups']} groups), so it is not encoding "
            "each order's own outcome."
        )

        pol = pd.DataFrame(t2["policy"])
        p_opts = sorted(pol["p"].unique())
        pv = st.select_slider(
            "Assumed intervention success",
            options=p_opts,
            value=p_opts[len(p_opts) // 2],
            format_func=lambda v: f"{v:.0%}",
            key="s2_p",
        )
        sel = pol[(pol["p"] == pv) & (pol["k"].round(3) == 0.10)].set_index("strategy")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Prevented: random picking", f"{sel.loc['random', 'prevented_expected']:.0f}")
        cB.metric("Prevented: at approval", f"{sel.loc['stage1', 'prevented_expected']:.0f}")
        cC.metric(
            "Prevented: at handover",
            f"{sel.loc['stage2', 'prevented_expected']:.0f}",
            f"{sel.loc['stage2', 'prevented_share_of_late']:.0%} of all late",
        )
        cD.metric("Ceiling (perfect ranking)", f"{sel.loc['oracle', 'prevented_expected']:.0f}")
        c.caption(
            "Simulation at 10% daily capacity; the success rate is an assumption. Acting at handover "
            "leaves fewer days than acting at approval, and some interventions may no longer be possible."
        )
        c.callout(
            "<b>Read with care.</b> The test months were already used once to evaluate the approval-time "
            "model, so this is a second, separately pre-declared model on the same months (selected by "
            "validation, scored once), not a first-look hold-out. It outputs a ranking only (no calibration "
            "fitted); use rank and bands.",
            kind="warn",
        )

# ------------------------------------------------------------------ calibration
with tabs[2]:
    sc = data.test_scored()
    y = sc["is_late"].to_numpy()
    fig = go.Figure()
    hi = 0.0
    for m in MODELS:
        r = reliability_table(y, sc[f"p_{m}"].to_numpy())
        hi = max(hi, r["mean_predicted"].max(), r["observed"].max())
        fig.add_trace(
            go.Scatter(
                x=r["mean_predicted"],
                y=r["observed"],
                mode="lines+markers",
                name=MODEL_LABELS[m],
                line={"color": theme.MODEL_COLORS[m], "width": 2},
                marker={"size": 8},
                hovertemplate="predicted %{x:.1%} → observed %{y:.1%}",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, hi],
            y=[0, hi],
            mode="lines",
            name="Perfect calibration",
            line={"color": theme.MUTED, "dash": "dash", "width": 1.5},
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title_text="Predicted probability of being late", tickformat=".0%")
    fig.update_yaxes(title_text="Observed share late", tickformat=".0%")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(theme.style_fig(fig, 420), width="stretch")
    c.caption(
        "Points on the dashed line are honest probabilities; points below it are overconfident "
        "(predict more late than happens)."
    )
    cal = pd.DataFrame(
        [
            {
                "Model": MODEL_LABELS[m],
                "Calibrator": res["models"][m]["calibrator"],
                "Brier (deployed)": f"{res['models'][m]['metrics']['brier']['value']:.4f}",
                "Brier (monthly recalibration)": f"{res['models'][m]['rolling_recalibration']['brier']:.4f}",
                "ECE (deployed)": f"{res['models'][m]['ece_deployed']:.4f}",
                "ECE (monthly recalibration)": f"{res['models'][m]['rolling_recalibration']['ece']:.4f}",
            }
            for m in MODELS
        ]
    )
    st.dataframe(cal, hide_index=True, width="stretch")
    c.callout(
        f"<b>Reading this honestly:</b> the late rate was {c.pct(pop['train_late_rate'])} in training and "
        f"{c.pct(pop['late_rate'])} in this period, and it moved month to month. A calibrator fitted on one "
        "period cannot follow that, and refitting monthly did not help here. Use the <b>rank and risk band</b> "
        "for decisions; treat the percentage as a relative score.",
        kind="warn",
    )

# ------------------------------------------------------------------ stability
with tabs[3]:
    mon = pd.DataFrame(res["monthly"])
    fig = go.Figure()
    for m in MODELS:
        g = mon[mon["model"] == m].sort_values("month")
        fig.add_trace(
            go.Scatter(
                x=g["month"],
                y=g["roc_auc"],
                mode="lines+markers",
                name=MODEL_LABELS[m],
                line={"color": theme.MODEL_COLORS[m], "width": 2},
                marker={"size": 8},
                hovertemplate="%{y:.2f}",
            )
        )
    fig.add_hline(y=0.5, line_dash="dash", line_color=theme.MUTED, annotation_text="random = 0.5")
    fig.update_yaxes(title_text="ROC-AUC by month", range=[0.4, 0.85])
    fig.update_xaxes(type="category")
    st.plotly_chart(theme.style_fig(fig, 360), width="stretch")
    c.caption(
        "Look for the slide toward the dashed line in the later months: ranking skill decays as "
        "time passes after training."
    )
    tbl = (
        mon[mon["model"] == prim]
        .sort_values("month")[
            [
                "month",
                "orders",
                "late_rate",
                "pr_auc",
                "roc_auc",
                "lift_at_10",
                "mean_predicted_deployed",
            ]
        ]
        .copy()
    )
    tbl["late_rate"] = tbl["late_rate"].map(c.pct)
    tbl["mean_predicted_deployed"] = tbl["mean_predicted_deployed"].map(c.pct)
    tbl.columns = [
        "Month",
        "Orders",
        "Late rate",
        "PR-AUC",
        "ROC-AUC",
        "Lift @10%",
        "Mean predicted",
    ]
    st.markdown(f"**{LABEL[prim]} by month**")
    st.dataframe(tbl.round(3), hide_index=True, width="stretch")
    c.callout(
        "<b>Implication:</b> schedule retraining and monitor drift; a model scored on the first "
        "months after training looks better than it will later.",
        kind="warn",
    )

# ------------------------------------------------------------------ explainability
with tabs[4]:
    summ = data.read_json("explain_summary.json")["agreement"]
    rk = summ["rank_agreement"]
    d = summ["direction"]
    a, b, e = st.columns(3)
    with a:
        c.kpi(
            "LightGBM vs EBM",
            f"{rk['spearman']['lightgbm vs ebm']:.2f}",
            f"rank correlation of global importance; top-10 overlap {rk['top_k_overlap']['lightgbm vs ebm']:.0%}",
        )
    with b:
        c.kpi(
            "Linear vs trees",
            f"{rk['spearman']['logreg vs lightgbm']:.2f}",
            f"rank correlation; top-10 overlap {rk['top_k_overlap']['logreg vs lightgbm']:.0%}",
        )
    with e:
        c.kpi(
            "Direction agreement",
            f"{d['agree']} of {d['features_compared']}",
            "numeric features whose effect direction the models agree on",
        )
    st.write("")
    img1, img2 = st.columns(2)
    with img1:
        st.image(str(data.FIGURES / "explain_ebm_shapes.png"), width="stretch")
        c.caption("EBM shape functions: each curve is the model's whole story for that feature.")
    with img2:
        st.image(str(data.FIGURES / "explain_shap_beeswarm.png"), width="stretch")
        c.caption(
            "SHAP summary for LightGBM: red dots are high feature values; right of centre means more risk."
        )
    c.callout(
        "<b>Not causal.</b> Explanations show how the models use information available at approval time. "
        f"Calendar features ({', '.join(d['differs_from_marginal'][:3])}) mostly reflect the part of the "
        "year the training data covered.",
        kind="warn",
    )

# ------------------------------------------------------------------ slices
with tabs[5]:
    sl = data.slices()
    which = st.selectbox(
        "Model",
        MODELS,
        index=MODELS.index(prim),
        format_func=lambda m: MODEL_LABELS[m],
        key="slice_m",
    )
    s = sl[sl["model"] == which].copy()
    dim = st.selectbox("Slice by", sorted(s["slice"].unique()))
    v = s[s["slice"] == dim].dropna(subset=["roc_auc"]).sort_values("roc_auc")
    overall = s["overall_roc_auc"].iloc[0]
    fig = go.Figure(
        go.Bar(
            x=v["roc_auc"],
            y=v["value"],
            orientation="h",
            marker_color=[theme.CRITICAL if w else theme.BLUE for w in v["weak"]],
            customdata=v[["orders", "late_orders"]],
            hovertemplate="%{y}: ROC-AUC %{x:.3f} · %{customdata[0]:,} orders · %{customdata[1]} late<extra></extra>",
        )
    )
    fig.add_vline(
        x=overall,
        line_dash="dash",
        line_color=theme.MUTED,
        annotation_text=f"overall {overall:.2f}",
    )
    fig.update_xaxes(range=[0.4, 0.85], title_text="ROC-AUC within the slice")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(theme.style_fig(fig, max(260, 38 * len(v) + 90), legend=False), width="stretch")
    c.caption(
        "Red bars are more than 0.05 below the model's overall ROC-AUC. "
        "Slices with fewer than 30 late orders are not shown."
    )
    st.dataframe(
        s[s["slice"] == dim][
            [
                "value",
                "orders",
                "late_orders",
                "late_rate",
                "mean_predicted",
                "roc_auc",
                "pr_auc",
                "weak",
            ]
        ].round(3),
        hide_index=True,
        width="stretch",
    )

# ------------------------------------------------------------------ limits
with tabs[6]:
    vm = val["models"][prim]["raw_metrics"]
    st.markdown(
        f"""
**What this system is.** A ranking aid for a capacity-limited operations team, built only from
information known at order approval.

**What it is not**
- *Not a precise probability.* Rates moved from {c.pct(pop["train_late_rate"])} (training) to {c.pct(pop["late_rate"])}
  (test period), and month to month; percentages are relative scores.
- *Not stable forever.* Ranking quality fell over the months after training (see Stability).
- *Not causal.* Explanations describe model behaviour, and the review-score gap is an adjusted association.
- *Not a measured business result.* The policy benefit is a simulation with an assumed success rate.

**Known weak points**
- Orders delayed for reasons invisible at approval (carrier incidents, stock-outs) are unpredictable
  from this data; see the false-negative case study.
- Sellers with little history are judged mostly on the platform average (by design).
- Regions and months flagged in *Where it is weak*.

**How honest evaluation was kept**
- Time-based splits only; the test period was opened once, after code and primary model were fixed.
- History features use only outcomes delivered before each order's approval time, with unit tests.
- Validation-period PR-AUC of the primary model on its untouched slice: {vm["val_calib"]["pr_auc"]:.3f}
  (base rate {vm["val_calib"]["base_rate"]:.3f}).
"""
    )
