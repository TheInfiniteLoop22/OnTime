"""Ops queue & policy simulator: which orders should operations act on today?"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ontime.evaluation.metrics import topk_metrics
from ontime.evaluation.policy import simulate_policy
from ontime.models.baselines import RuleBaseline
from ontime.serving.service import MODEL_LABELS, MODELS
from ontime_ui import components as c
from ontime_ui import data, theme

theme.inject_css()
data.require_artifacts()

cfg = data.CFG
scored = data.test_scored()
res = data.read_json("results.json")
prim = data.primary_model()
seed = cfg["seed"]
p_grid = cfg["policy"]["intervention_success_grid"]
K_CURVE = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]


@st.cache_data(show_spinner="Simulating the policy over the test period...")
def curves(model: str) -> pd.DataFrame:
    """Policy simulation at p = 1 (benefits are linear in p, so scale later)."""
    t = data.features_all()
    rule = RuleBaseline().fit(t[t["split"] == "train"])
    test = t[t["split"] == "test"].reset_index(drop=True)
    scores = {model: scored[f"p_{model}"].to_numpy(), "rule": rule.predict_proba(test)}
    return simulate_policy(
        scored["approved_at"],
        scored["is_late"].to_numpy(),
        scores,
        K_CURVE,
        [1.0],
        seed,
        n_boot=150,
    )


c.hero(
    "Ops queue & policy simulator",
    "Choose a day and how many orders the team can act on. The queue lists the riskiest orders "
    "first, with the main reason; the simulator shows what that capacity is worth.",
)

# ------------------------------------------------------------------ controls
with st.sidebar:
    st.subheader("Queue settings")
    model = st.selectbox(
        "Model", MODELS, index=MODELS.index(prim), format_func=lambda m: MODEL_LABELS[m]
    )
    days = sorted(scored["day"].dt.date.unique())
    busiest = scored.groupby(scored["day"].dt.date).size().idxmax()
    day = st.date_input("Day", value=busiest, min_value=days[0], max_value=days[-1])
    k = st.slider("Ops capacity (% of the day's orders)", 1, 30, 10) / 100
    p = st.select_slider(
        "Assumed intervention success",
        options=p_grid,
        value=p_grid[len(p_grid) // 2],
        format_func=lambda v: f"{v:.0%}",
        help="Chance an intervention prevents a would-be-late order. An assumption.",
    )
    reveal = st.toggle("Reveal outcomes (back-test)", value=False)

dfd = scored[scored["day"].dt.date == day]
if dfd.empty:
    st.info("No orders on that day in the test period.")
    st.stop()
n = len(dfd)
quota = max(1, math.ceil(k * n))
queue = dfd.nsmallest(quota, f"rank_{model}").copy()

# back-test precision at this capacity over the WHOLE test period (not just today)
bt = topk_metrics(
    scored["approved_at"], scored["is_late"].to_numpy(), scored[f"p_{model}"].to_numpy(), k, seed
)
exp_late = quota * bt["precision"]

k1, k2, k3, k4 = st.columns(4)
with k1:
    c.kpi("Orders approved", f"{n:,}", f"{day:%A %d %B %Y}")
with k2:
    c.kpi("Queue size", f"{quota:,}", f"Top {k:.0%} of the day by risk ({MODEL_LABELS[model]})")
with k3:
    c.kpi(
        "Expected late in queue",
        f"{exp_late:.1f}",
        f"Back-test precision at {k:.0%} capacity: {c.pct(bt['precision'])} "
        f"({bt['lift']:.1f}x the {c.pct(res['population']['late_rate'])} base rate)",
    )
with k4:
    c.kpi(
        "Expected prevented",
        f"{p * exp_late:.1f}",
        f"If {p:.0%} of interventions work (assumption)",
    )
st.write("")

# ------------------------------------------------------------------ the queue
st.subheader("Today's queue")
codes_m = data.reason_codes().query("model == @model").set_index("order_id")
reasons = codes_m["risk_factors"]
groups = codes_m["risk_groups"]
queue["Top reason"] = queue["order_id"].map(
    lambda o: reasons.get(o, [""])[0].rsplit(" (", 1)[0] if len(reasons.get(o, [])) else "-"
)
queue["Main driver"] = queue["order_id"].map(
    lambda o: groups.get(o, [])[0].replace("_", " ").capitalize() if len(groups.get(o, [])) else "-"
)
out = pd.DataFrame(
    {
        "Rank": queue[f"rank_{model}"].astype(int),
        "Band": [data.band_for(v) for v in queue[f"pct_{model}"]],
        "Risk estimate": (queue[f"p_{model}"] * 100).round(0),
        "Main driver": queue["Main driver"],
        "Route": queue["seller_state"] + " → " + queue["customer_state"],
        "Distance km": queue["distance_km"].round(0),
        "Promised days": queue["promised_days"].round(1),
        "Seller orders so far": queue["seller_n_past"].astype(int),
        "Order ID": queue["order_id"],
    }
)
if reveal:
    out["Outcome"] = queue["is_late"].map({1: "LATE", 0: "on time"}).to_numpy()
    st.caption(
        f"Back-test: {int(queue['is_late'].sum())} of {quota} queued orders were actually late."
    )
st.dataframe(
    out,
    hide_index=True,
    width="stretch",
    height=min(40 + 35 * len(out), 560),
    column_config={
        "Risk estimate": st.column_config.ProgressColumn(
            "Risk estimate", format="%d%%", min_value=0, max_value=100
        ),
        "Distance km": st.column_config.NumberColumn(format="%d"),
        "Order ID": st.column_config.TextColumn(width="large"),
    },
)
full_csv = out.assign(**{"Top reason (full sentence)": queue["Top reason"].to_numpy()})
st.download_button(
    "Download queue (CSV)",
    full_csv.to_csv(index=False).encode("utf-8"),
    file_name=f"ontime_queue_{day}.csv",
    mime="text/csv",
)
c.caption(
    "Orders are ranked by risk within the day; the band comes from rank, not from the exact "
    "percentage, because absolute probabilities drift when the base rate moves."
)

# ------------------------------------------------------------------ policy simulator
st.subheader("What is this capacity worth? (simulation over the whole test period)")
cv = curves(model).copy()
for col in [
    "prevented_expected",
    "prevented_share_of_late",
    "extra_vs_random",
    "extra_vs_random_ci_low",
    "extra_vs_random_ci_high",
]:
    cv[col] = cv[col] * p
names = {
    model: MODEL_LABELS[model],
    "rule": "Hand-written rule",
    "random": "Random selection",
    "oracle": "Perfect ranking (upper bound)",
}
styles = {
    model: (theme.MODEL_COLORS[model], "solid"),
    "rule": (theme.MUTED, "solid"),
    "random": (theme.BASELINE, "dash"),
    "oracle": (theme.INK, "dot"),
}
fig = go.Figure()
for s, (col, dash) in styles.items():
    g = cv[cv["strategy"] == s].sort_values("k")
    fig.add_trace(
        go.Scatter(
            x=g["k"] * 100,
            y=g["prevented_share_of_late"],
            mode="lines+markers",
            name=names[s],
            line={"color": col, "width": 2, "dash": dash},
            marker={"size": 8},
            hovertemplate="%{y:.1%} of late orders<extra>" + names[s] + "</extra>",
        )
    )
fig.add_vline(x=k * 100, line_dash="dot", line_color=theme.MUTED)
fig.update_xaxes(title_text="Ops capacity: % of each day's orders", ticksuffix="%")
fig.update_yaxes(title_text="Late orders prevented (% of all late)", tickformat=".0%")
st.plotly_chart(theme.style_fig(fig, 420), width="stretch")
c.caption(
    f"Look at the gap between the model and random selection at your capacity (dotted line): "
    f"it is the value of ranking. Assumes {p:.0%} of interventions succeed."
)

one = simulate_policy(
    scored["approved_at"],
    scored["is_late"].to_numpy(),
    {model: scored[f"p_{model}"].to_numpy()},
    [k],
    [p],
    seed,
    n_boot=200,
)
m_row = one[one["strategy"] == model].iloc[0]
o_row = one[one["strategy"] == "oracle"].iloc[0]
a, b, d = st.columns(3)
a.metric(
    "Late orders prevented (model)",
    f"{m_row['prevented_expected']:.0f}",
    f"{m_row['prevented_share_of_late']:.1%} of all late",
)
b.metric(
    "Gain over random selection",
    f"{m_row['extra_vs_random']:+.0f}",
    f"95% CI [{m_row['extra_vs_random_ci_low']:+.0f}, {m_row['extra_vs_random_ci_high']:+.0f}]",
    delta_color="off",
)
d.metric(
    "Ceiling with perfect ranking",
    f"{o_row['prevented_expected']:.0f}",
    f"model reaches {m_row['prevented_expected'] / o_row['prevented_expected']:.0%} of it",
)
c.callout(
    "<b>Read this as a simulation.</b> The success rate of an intervention is an assumption "
    "(sliders above); the interval reflects day-to-day sampling noise only. Interventions must "
    "happen <b>before the promised date</b> to matter: most of the review damage of a late order "
    "happens while the customer is still waiting.",
    kind="warn",
)
