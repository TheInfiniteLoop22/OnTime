"""Order risk explorer: pick an order, see its risk, the reasons, a what-if and a second opinion."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ontime.explain.cases import pick_cases
from ontime.serving.service import MODEL_LABELS, MODELS, recompute_dependent_features
from ontime_ui import components as c
from ontime_ui import data, theme

theme.inject_css()
data.require_artifacts()

svc = data.get_service()
scored = data.test_scored()
res = data.read_json("results.json")
base_rate = res["population"]["late_rate"]
prim = data.primary_model()
codes = data.reason_codes()

c.hero(
    "Order risk explorer",
    "Pick an order from the test period, see how risky the model thinks it is and why, then try "
    "what changing the promise or the freight would do.",
)

# ------------------------------------------------------------------ sidebar: choose an order
with st.sidebar:
    st.subheader("Find an order")
    model = st.selectbox(
        "Model",
        MODELS,
        index=MODELS.index(prim),
        format_func=lambda m: MODEL_LABELS[m],
        help="The primary model was chosen by validation performance before the test set was opened.",
    )
    mode = st.radio(
        "How to pick", ["Riskiest orders of a day", "Case studies", "Paste an order ID"]
    )
    reveal = st.toggle(
        "Reveal actual outcome (back-test)",
        value=False,
        help="Off by default so you can judge the reasons before seeing what happened.",
    )

order_id: str | None = None
if mode == "Riskiest orders of a day":
    days = sorted(scored["day"].dt.date.unique())
    busiest = scored.groupby(scored["day"].dt.date).size().idxmax()
    with st.sidebar:
        day = st.date_input("Day", value=busiest, min_value=days[0], max_value=days[-1])
        n_show = st.slider("Orders in the list", 10, 100, 25, step=5)
    dfd = scored[scored["day"].dt.date == day].nsmallest(n_show, f"rank_{model}")
    if dfd.empty:
        st.info("No orders on that day in the test period. Pick another date.")
        st.stop()
    labels = {
        r.order_id: f"#{int(getattr(r, f'rank_{model}'))} · {getattr(r, f'p_{model}'):.0%} · "
        f"{r.seller_state}→{r.customer_state} · {r.distance_km:,.0f} km"
        for r in dfd.itertuples()
    }
    with st.sidebar:
        order_id = st.selectbox("Order", list(labels), format_func=labels.get)
elif mode == "Case studies":
    cases = pick_cases(codes)
    names = {
        "true_positive": "True positive: late, flagged high",
        "false_positive": "False positive: on time, flagged high",
        "false_negative": "False negative: late, flagged low",
    }
    with st.sidebar:
        key = st.radio("Case", list(cases), format_func=names.get)
    order_id = cases[key]
else:
    with st.sidebar:
        typed = st.text_input("Order ID (32 characters)").strip()
    if not typed:
        st.info("Paste an order ID from the test period in the sidebar.")
        st.stop()
    if typed not in set(scored["order_id"]):
        st.warning("That order is not in the test period.")
        st.stop()
    order_id = typed

row = scored[scored["order_id"] == order_id].iloc[0]
frow = data.feature_row(order_id)
exp = svc.explain_one(frow, model)
prob = exp["probability"]
rank, n_day = row[f"rank_{model}"], row["n_day"]
pct_rank = row[f"pct_{model}"]
band = data.band_for(pct_rank)

# ------------------------------------------------------------------ header
st.markdown(f"#### Order `{order_id}`")
st.markdown(
    c.band_badge(band)
    + f' &nbsp; <span style="color:{theme.INK2}">{row["seller_state"]} → {row["customer_state"]} · '
    f"approved {row['approved_at']:%d %b %Y, %H:%M}</span>",
    unsafe_allow_html=True,
)
k1, k2, k3 = st.columns(3)
with k1:
    c.kpi(
        "Model risk estimate",
        f"{prob:.0%}",
        f"{prob / base_rate:.1f}x the platform's late rate ({c.pct(base_rate)})",
        "A relative score. Probabilities are overconfident for the very riskiest orders; "
        "rank and band are more reliable.",
    )
with k2:
    c.kpi(
        "Rank within the day",
        f"#{rank:.0f} of {n_day:.0f}",
        f"Top {max(pct_rank * 100, 0.1):.1f}% of {row['day']:%d %b %Y}",
    )
with k3:
    if reveal:
        late = bool(row["is_late"])
        c.kpi(
            "Actual outcome",
            "LATE" if late else "On time",
            f"Delivered in {row['delivery_days']:.1f} days vs {row['promised_days']:.1f} promised",
        )
    else:
        c.kpi("Actual outcome", "Hidden", "Turn on 'Reveal actual outcome' in the sidebar")
st.write("")

tab_why, tab_what, tab_cmp, tab_facts = st.tabs(
    ["Why this score", "What-if", "Compare models", "Order details"]
)

# ------------------------------------------------------------------ tab 1: why
with tab_why:
    a, b = st.columns(2)
    with a:
        st.markdown("**What adds risk**")
        c.reasons(exp["risk_factors"], "risk")
    with b:
        st.markdown("**What reduces risk**")
        c.reasons(exp["protective_factors"], "protect")
    st.write("")
    st.markdown("**How each reason moves the score**")
    s = exp["contributions"]
    s = s[s.abs() > 0.005].sort_values(ascending=False)
    fig = go.Figure(
        go.Bar(
            x=s.values,
            y=[g.replace("_", " ").capitalize() for g in s.index],
            orientation="h",
            marker_color=[theme.CRITICAL if v > 0 else theme.BLUE for v in s.values],
            hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(
        title_text="Effect on the order's risk (log-odds): right = more risk, left = less"
    )
    fig.update_layout(hovermode="closest")
    st.plotly_chart(theme.style_fig(fig, max(260, 34 * len(s) + 90), legend=False), width="stretch")
    c.caption(
        "Each bar sums the features of one idea (for example three distance measures). "
        "Red pushes the order toward late, blue toward on time; the bars add up to the model's score."
    )

# ------------------------------------------------------------------ tab 2: what-if
with tab_what:
    st.markdown(
        "Change a lever and the app recomputes every feature that depends on it, then re-scores "
        "the order with the saved model."
    )
    cur_promise = float(frow.loc[0, "promised_days"])
    cur_freight = float(frow.loc[0, "total_freight"])
    w1, w2 = st.columns(2)
    with w1:
        lo = -int(max(0, min(5, cur_promise - 1)))
        extra = st.slider(
            "Extend the promised delivery by (days)",
            lo,
            20,
            0,
            help=f"Currently {cur_promise:.1f} days promised.",
        )
    with w2:
        freight = st.slider(
            "Freight charged (R$)",
            0.0,
            max(cur_freight * 2, 10.0),
            cur_freight,
            step=1.0,
            help="Changes freight and the freight-to-price ratio.",
        )
    edited = recompute_dependent_features(
        frow, extra, freight if abs(freight - cur_freight) > 1e-9 else None
    )
    new = svc.explain_one(edited, model)
    m1, m2, m3 = st.columns(3)
    m1.metric("Original risk estimate", f"{prob:.1%}")
    m2.metric(
        "What-if risk estimate",
        f"{new['probability']:.1%}",
        f"{(new['probability'] - prob) * 100:+.1f} pts"
        if abs(new["probability"] - prob) >= 0.0005
        else None,
        delta_color="inverse",
    )
    m3.metric(
        "Promised delivery",
        f"{cur_promise + extra:.1f} days",
        f"{extra:+d} days" if extra else None,
        delta_color="off",
    )

    sweep = svc.promise_sweep(frow, model, 20)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sweep["extra_promise_days"],
            y=sweep["probability"],
            mode="lines",
            line={"color": theme.BLUE, "width": 2},
            name="Risk estimate",
            hovertemplate="+%{x} days: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[max(extra, 0)],
            y=[new["probability"] if extra >= 0 else prob],
            mode="markers+text",
            marker={"size": 12, "color": theme.ORANGE},
            text=["your what-if"],
            textposition="top center",
            name="What-if",
            hoverinfo="skip",
        )
    )
    fig.update_yaxes(tickformat=".0%", title_text="Model risk estimate")
    fig.update_xaxes(title_text="Extra days added to the promise")
    st.plotly_chart(theme.style_fig(fig, 320, legend=False), width="stretch")
    c.caption(
        "Look at how steeply risk falls as the promise is lengthened: a flat stretch means more days "
        "no longer help this order."
    )
    c.callout(
        "<b>This shows the model's sensitivity, not a causal effect.</b> A longer promise may cost "
        "conversion or goodwill, and the real-world effect on delivery time was not measured here.",
        kind="warn",
    )

# ------------------------------------------------------------------ tab 3: compare models
with tab_cmp:
    rows, contribs = [], {}
    for m in MODELS:
        e = svc.explain_one(frow, m)
        contribs[m] = e["contributions"]
        rows.append(
            {
                "Model": MODEL_LABELS[m],
                "Risk estimate": f"{e['probability']:.1%}",
                "Rank in day": f"#{row[f'rank_{m}']:.0f} of {n_day:.0f}",
                "Band": data.band_for(row[f"pct_{m}"]),
                "Leading reason": (
                    e["risk_factors"][0].rsplit(" (", 1)[0] if e["risk_factors"] else "-"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    c.caption(
        "Three different model families score the same order; compare rank and band more than the percentage."
    )
    top_groups = (
        pd.DataFrame(contribs).abs().max(axis=1).sort_values(ascending=False).head(8).index.tolist()
    )
    fig = go.Figure()
    for m in MODELS:
        fig.add_trace(
            go.Bar(
                y=[g.replace("_", " ").capitalize() for g in top_groups],
                x=[contribs[m][g] for g in top_groups],
                orientation="h",
                name=MODEL_LABELS[m],
                marker_color=theme.MODEL_COLORS[m],
                hovertemplate="%{y}: %{x:+.2f}<extra>" + MODEL_LABELS[m] + "</extra>",
            )
        )
    fig.update_layout(barmode="group", hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Effect on risk (log-odds): right = more risk")
    st.plotly_chart(theme.style_fig(fig, 460), width="stretch")
    c.caption(
        "Where the three bars point the same way, the explanation is robust to the choice of model."
    )

# ------------------------------------------------------------------ tab 4: facts
with tab_facts:
    f = frow.iloc[0]
    facts = [
        ("Route", f"{row['seller_state']} → {row['customer_state']} ({f['distance_km']:,.0f} km)"),
        ("Promised delivery", f"{f['promised_days']:.1f} days from approval"),
        ("Seller hand-over window", f"{f['handling_window_days']:.1f} days"),
        (
            "Seller history",
            f"{f['seller_n_past']:.0f} completed orders · estimated late rate "
            f"{f['seller_late_rate_shrunk']:.1%} (all time), {f['seller_late_rate_90d_shrunk']:.1%} (90 days)",
        ),
        (
            "Route history",
            f"{f['route_late_rate_shrunk']:.1%} of earlier orders on this route were late",
        ),
        (
            "Network context",
            f"{f['global_late_rate_30d']:.1%} of deliveries in the last 30 days were late",
        ),
        (
            "Items & weight",
            f"{f['n_items']:.0f} item(s), {f['total_weight_kg']:.1f} kg, {f['total_volume_l']:.0f} L",
        ),
        (
            "Value & freight",
            f"R$ {f['total_price']:,.0f} goods, R$ {f['total_freight']:,.0f} freight",
        ),
        ("Category", str(f["category_group"]).replace("_", " ")),
        (
            "Payment",
            f"{f['payment_type']}, up to {f['payment_installments_max']:.0f} instalment(s)",
        ),
    ]
    st.dataframe(pd.DataFrame(facts, columns=["Fact", "Value"]), hide_index=True, width="stretch")
    c.caption(
        "These are the inputs the model saw at approval time; nothing recorded afterwards is used."
    )
