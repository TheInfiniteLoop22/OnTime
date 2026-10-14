"""Seller scorecard: a seller's track record, as the model sees it, with its typical risk reasons."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ontime_ui import components as c
from ontime_ui import data, theme

theme.inject_css()
data.require_artifacts()

cfg = data.CFG
alpha = cfg["features"]["shrinkage_alpha"]
t = data.features_all()
prim = data.primary_model()
codes = data.reason_codes()

c.hero(
    "Seller scorecard",
    "How a seller has actually performed, what the model believed about them at the time, and the "
    "reasons that most often make their orders risky.",
)


# ------------------------------------------------------------------ seller picker
@st.cache_data(show_spinner=False)
def seller_table() -> pd.DataFrame:
    g = t.sort_values("approved_at").groupby("primary_seller_id")
    s = g.agg(
        orders=("is_late", "size"),
        late_rate=("is_late", "mean"),
        first=("approved_at", "min"),
        last=("approved_at", "max"),
        state=("seller_state", "last"),
        recent_estimate=("seller_late_rate_90d_shrunk", "last"),
        past_at_last=("seller_n_past", "last"),
    )
    return s.reset_index()


sellers = seller_table()
eligible = sellers[sellers["orders"] >= 30]
with st.sidebar:
    st.subheader("Choose a seller")
    sort_by = st.selectbox(
        "Sort sellers by",
        ["Most orders", "Highest recent late-rate estimate", "Newest (fewest past orders)"],
    )
    ordered = {
        "Most orders": eligible.sort_values("orders", ascending=False),
        "Highest recent late-rate estimate": eligible.sort_values(
            "recent_estimate", ascending=False
        ),
        "Newest (fewest past orders)": sellers.sort_values("past_at_last"),
    }[sort_by].head(200)
    labels = {
        r.primary_seller_id: f"{r.primary_seller_id[:8]}… · {r.state} · {r.orders:,} orders"
        for r in ordered.itertuples()
    }
    seller_id = st.selectbox("Seller", list(labels), format_func=labels.get)
    typed = st.text_input("…or paste a seller ID").strip()
    if typed in set(sellers["primary_seller_id"]):
        seller_id = typed

me = sellers[sellers["primary_seller_id"] == seller_id].iloc[0]
d = t[t["primary_seller_id"] == seller_id].sort_values("approved_at")
platform = t["is_late"].mean()
rank_pool = eligible["recent_estimate"].dropna()
pctile = (rank_pool < me["recent_estimate"]).mean()

# ------------------------------------------------------------------ header + KPIs
st.markdown(f"#### Seller `{seller_id}` · {me['state']}")
thin = me["past_at_last"] < alpha
if thin:
    c.callout(
        f"<b>Thin history:</b> only {int(me['past_at_last'])} completed orders before the most "
        f"recent one. The model leans on the platform average for sellers with fewer than {alpha} "
        "orders (Bayesian shrinkage), so a new seller is not judged on a handful of deliveries."
    )
    st.write("")
k1, k2, k3, k4 = st.columns(4)
with k1:
    c.kpi(
        "Orders in the data", f"{int(me['orders']):,}", f"{me['first']:%b %Y} to {me['last']:%b %Y}"
    )
with k2:
    c.kpi("Observed late rate", c.pct(me["late_rate"]), f"Platform average {c.pct(platform)}")
with k3:
    c.kpi(
        "Model's current estimate",
        c.pct(me["recent_estimate"]),
        "Shrunk late rate over the last 90 days, as of the latest order",
    )
with k4:
    c.kpi(
        "Position among sellers",
        f"{pctile:.0%}",
        "of sellers with 30+ orders have a lower estimate (higher = riskier)",
    )
st.write("")

# ------------------------------------------------------------------ monthly track record
st.subheader("Track record over time")
d = d.assign(month=d["approved_at"].dt.to_period("M").dt.start_time)
mo = (
    d.groupby("month")
    .agg(
        orders=("is_late", "size"),
        late=("is_late", "mean"),
        estimate=("seller_late_rate_90d_shrunk", "mean"),
    )
    .reset_index()
)
plat_g = t.assign(month=t["approved_at"].dt.to_period("M").dt.start_time).groupby("month")[
    "is_late"
]
plat = plat_g.mean()[plat_g.size() >= 500]  # skip months too thin to be a baseline
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.34, 0.66], vertical_spacing=0.07
)
fig.add_trace(
    go.Bar(x=mo["month"], y=mo["orders"], name="Orders", marker_color=theme.BASELINE,
           hovertemplate="%{y} orders"),
    row=1, col=1,
)  # fmt: skip
shown = mo[mo["orders"] >= 5]
fig.add_trace(
    go.Scatter(x=shown["month"], y=shown["late"], mode="lines+markers", name="Observed late rate",
               line={"color": theme.BLUE, "width": 2}, marker={"size": 8}, hovertemplate="%{y:.1%}"),
    row=2, col=1,
)  # fmt: skip
fig.add_trace(
    go.Scatter(x=mo["month"], y=mo["estimate"], mode="lines+markers",
               name="Estimate the model used at the time",
               line={"color": theme.ORANGE, "width": 2}, marker={"size": 8}, hovertemplate="%{y:.1%}"),
    row=2, col=1,
)  # fmt: skip
fig.add_trace(
    go.Scatter(x=plat.index, y=plat.values, mode="lines", name="Platform late rate",
               line={"color": theme.MUTED, "width": 2, "dash": "dot"}, hovertemplate="%{y:.1%}"),
    row=2, col=1,
)  # fmt: skip
fig.update_yaxes(tickformat=".0%", row=2, col=1, title_text="Late rate")
fig.update_yaxes(row=1, col=1, title_text="Orders")
st.plotly_chart(theme.style_fig(fig, 460), width="stretch")
c.caption(
    "Look at how closely the orange estimate follows the blue observed rate with a delay: the "
    "model only knows outcomes that were already delivered. Months with fewer than 5 orders are "
    "left off the observed line."
)

# ------------------------------------------------------------------ reasons + routes
left, right = st.columns([1.1, 1])
with left:
    st.subheader("What typically makes their orders risky")
    mine = codes[(codes["model"] == prim) & codes["order_id"].isin(d["order_id"])]
    if mine.empty:
        st.info("This seller has no orders in the test period, so no reason codes were generated.")
    else:
        counts = mine["risk_groups"].explode().value_counts() / len(mine)
        counts = counts.head(8).sort_values()
        fig2 = go.Figure(
            go.Bar(
                x=counts.values,
                y=[g.replace("_", " ").capitalize() for g in counts.index],
                orientation="h",
                marker_color=theme.CRITICAL,
                hovertemplate="%{x:.0%} of orders<extra></extra>",
            )
        )
        fig2.update_xaxes(tickformat=".0%", title_text="Share of their test-period orders")
        fig2.update_layout(hovermode="closest")
        st.plotly_chart(theme.style_fig(fig2, 340, legend=False), width="stretch")
        c.caption(
            f"Across {len(mine):,} test-period orders, how often each reason is among the top "
            "risk-increasing ones for the primary model. A reason that is always present is a "
            "property of the seller or its routes, not of one unlucky order."
        )
with right:
    st.subheader("Where they ship")
    r = (
        d.groupby("customer_state")
        .agg(orders=("is_late", "size"), late_rate=("is_late", "mean"))
        .sort_values("orders", ascending=False)
        .head(6)
        .reset_index()
    )
    r["late_rate"] = r["late_rate"].map(lambda v: f"{v:.1%}")
    r.columns = ["Customer state", "Orders", "Late rate"]
    st.dataframe(r, hide_index=True, width="stretch")
    c.caption(
        "Late rates on few orders are noisy; compare them with the platform average, not with each other."
    )
