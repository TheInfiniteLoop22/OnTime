"""Overview: the business case in one screen."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ontime_ui import components as c
from ontime_ui import data, theme

theme.inject_css()
data.require_artifacts()

res = data.read_json("results.json")
imp = data.read_json("impact.json")
cfg = data.CFG
prim = res["primary_model"]
pop = res["population"]

c.hero(
    "OnTime",
    "Predict, at the moment an order is approved, whether it will miss the date promised to the "
    "customer - explain why in plain language - and decide which orders operations should act on.",
)

# ---------------------------------------------------------------- headline numbers
k10 = next(t for t in res["models"][prim]["top_k"] if abs(t["k"] - 0.10) < 1e-9)
pol = pd.DataFrame(res["policy"])
p_values = sorted(pol["p"].unique())
p_mid = p_values[len(p_values) // 2]
sel = pol[(pol["k"] == 0.10) & (pol["p"] == p_mid)].set_index("strategy")
star = res["star_gap_assumptions"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    c.kpi(
        "Late-delivery rate",
        c.pct(pop["late_rate"]),
        f"Test period {pop['from']} to {pop['to']} · training period {c.pct(pop['train_late_rate'])}",
    )
with c2:
    c.kpi(
        "Ranking power",
        f"{k10['lift']:.1f}x lift",
        f"Of the riskiest 10% of each day's orders, {c.pct(k10['precision'])} were late "
        f"(vs {c.pct(pop['late_rate'])} overall)",
        "Lift = precision in the flagged group divided by the overall late rate.",
    )
with c3:
    c.kpi(
        "Review stars lost per late order",
        f"-{star['conservative_gap_stars']:.1f} to -{star['full_adjusted_gap_stars']:.1f}",
        "Lower review score vs on-time orders (adjusted association, not causal)",
    )
with c4:
    c.kpi(
        "Late orders prevented (simulated)",
        f"{c.pct(sel.loc[prim, 'prevented_share_of_late'])}",
        f"prevented by acting on the top 10% daily, if {p_mid:.0%} of interventions work "
        f"(random picking: {c.pct(sel.loc['random', 'prevented_share_of_late'])})",
        "A simulation: the success rate of an intervention is an assumption, not a measurement.",
    )

st.write("")

# ---------------------------------------------------------------- how it works
st.subheader("How it works")
c.steps(
    [
        (
            "Predict",
            "At approval time, a model scores every order using only information known then.",
        ),
        (
            "Explain",
            "Each score comes with plain-English reasons: what adds risk and what reduces it.",
        ),
        ("Decide", "Operations act on the top-k% riskiest orders each day, within their capacity."),
        (
            "Measure",
            "Every number is checked on a later period the model never saw, with intervals.",
        ),
    ]
)
st.write("")

# ---------------------------------------------------------------- charts
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Lateness comes in episodes")
    t = data.features_all()
    m = (
        t.assign(month=t["approved_at"].dt.to_period("M").dt.start_time)
        .groupby("month")
        .agg(orders=("is_late", "size"), late_rate=("is_late", "mean"))
        .reset_index()
    )
    m = m[m["orders"] >= 500]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(
            x=m["month"],
            y=m["late_rate"],
            mode="lines+markers",
            name="Late rate",
            line={"color": theme.BLUE, "width": 2},
            marker={"size": 8},
            hovertemplate="%{y:.1%}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=m["month"],
            y=m["orders"],
            name="Orders",
            marker_color=theme.BASELINE,
            hovertemplate="%{y:,} orders",
        ),
        row=2,
        col=1,
    )
    val_start, test_start = cfg["splits"]["val_start"], cfg["splits"]["test_start"]
    for x, label in [(val_start, "validation"), (test_start, "test")]:
        fig.add_vline(x=pd.Timestamp(x).timestamp() * 1000, line_dash="dot", line_color=theme.MUTED)
        fig.add_annotation(
            x=pd.Timestamp(x),
            y=1,
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"color": theme.MUTED, "size": 12},
        )
    fig.update_yaxes(
        tickformat=".0%",
        row=1,
        col=1,
        title_text="Late deliveries",
        range=[0, m["late_rate"].max() * 1.22],
    )
    fig.update_yaxes(row=2, col=1, title_text="Orders")
    st.plotly_chart(theme.style_fig(fig, 400, legend=False), width="stretch")
    peaks = m.nlargest(2, "late_rate").sort_values("month")
    spike_txt = ", ".join(f"{r.month:%b %Y} at {r.late_rate:.0%}" for r in peaks.itertuples())
    c.caption(
        f"Look for the spikes ({spike_txt}): the late rate moves far more than sampling noise, "
        "which is why models are judged on a later period."
    )

with right:
    st.subheader("A late order is rated lower")
    d = pd.DataFrame(imp["dose_response"])
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=d["timing"],
            y=d["mean_score"],
            mode="lines+markers",
            name="Mean review score",
            line={"color": theme.BLUE, "width": 2},
            marker={"size": 9},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": d["ci_high"] - d["mean_score"],
                "arrayminus": d["mean_score"] - d["ci_low"],
                "color": theme.BLUE,
            },
            customdata=d[["orders"]],
            hovertemplate="%{y:.2f} stars · %{customdata[0]:,} orders",
        )
    )
    fig2.update_yaxes(range=[1, 5], title_text="Mean review score (stars)")
    fig2.update_xaxes(tickangle=-30)
    st.plotly_chart(theme.style_fig(fig2, 400, legend=False), width="stretch")
    worst = d.loc[d["mean_score"].idxmin()]
    c.caption(
        f"Look for the first late bucket and the floor ({worst['mean_score']:.1f} stars at "
        f"'{worst['timing']}'): the damage starts at the first missed day. Bars show 95% intervals."
    )

# ---------------------------------------------------------------- honesty
monthly = pd.DataFrame(res["monthly"])
mp = monthly[monthly["model"] == prim].sort_values("month")
first, last = mp.iloc[0], mp.iloc[-1]
timing = {r["group"]: r for r in imp["review_timing"]}
before = timing["late, reviewed BEFORE arrival"]["share_of_group_orders"]
c.callout(
    f"<b>Two things to keep in mind.</b> "
    f"(1) <b>{c.pct(before, 0)} of late orders were reviewed before the parcel arrived</b>: much of the "
    f"damage is done while the customer is still waiting, so interventions must happen before the "
    f"promised date. "
    f"(2) Ranking quality is not constant: the primary model's ROC-AUC was {first['roc_auc']:.2f} in "
    f"{first['month']} and {last['roc_auc']:.2f} in {last['month']}. Models need monitoring and "
    f"periodic retraining.",
    kind="warn",
)
