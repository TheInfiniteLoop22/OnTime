"""Drift monitor: is the world still the one the model was trained on?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ontime.features.build import CATEGORICAL_FEATURES
from ontime.monitoring.drift import psi_categorical, psi_numeric
from ontime_ui import components as c
from ontime_ui import data, theme

theme.inject_css()
data.require_artifacts()

REP = data.REPORTS
needed = [REP / "drift_status.csv", REP / "drift_psi.csv", REP / "drift_summary.json"]
if not all(p.exists() for p in needed):
    st.error("Drift tables are missing. Run `make monitor` (or `make all`) and reload.")
    st.stop()

cfg = data.CFG
mcfg = cfg["monitoring"]
status = pd.read_csv(REP / "drift_status.csv", parse_dates=["month"])
psi = pd.read_csv(REP / "drift_psi.csv", parse_dates=["month"])
summ = data.read_json("drift_summary.json")
expected = set(summ["expected_drift_features"])
watch, alert = mcfg["psi_watch"], mcfg["psi_alert"]
STATUS_COLOR = {"stable": theme.GOOD, "watch": theme.SERIOUS, "alert": theme.CRITICAL}
STATUS_ICON = {"stable": "●", "watch": "▲", "alert": "▲▲"}

c.hero(
    "Drift monitor",
    "Models decay silently. This page watches the incoming orders (does today's mix look like the "
    "training data?) and the model's ranking quality month by month.",
)

latest = status.iloc[-1]
k1, k2, k3, k4 = st.columns(4)
with k1:
    c.kpi(
        "Latest month status",
        f"{STATUS_ICON[latest['status']]} {latest['status'].upper()}",
        f"{latest['month']:%B %Y}",
    )
with k2:
    c.kpi(
        "Order-mix features in alert",
        f"{int(latest['trail_alert'])}",
        f"PSI at or above {alert:.2f} vs the previous {mcfg['trailing_months']} months",
        "Population Stability Index: how much a feature's distribution moved. State features are excluded.",
    )
with k3:
    c.kpi(
        "Ranking quality (ROC-AUC)",
        f"{latest['roc_auc']:.2f}",
        f"{latest['roc_vs_first']:+.2f} vs the first monitored month",
    )
with k4:
    c.kpi("Lift at 10% capacity", f"{latest['lift_at_10']:.2f}x", "1.0x = no better than random")
st.write("")

# ------------------------------------------------------------------ timeline
st.subheader("Signals over time")
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.08
)
fig.add_trace(
    go.Scatter(
        x=status["month"], y=status["roc_auc"], mode="lines+markers", name="ROC-AUC (deployed model)",
        line={"color": theme.BLUE, "width": 2}, marker={"size": 9}, hovertemplate="%{y:.2f}",
    ),
    row=1, col=1,
)  # fmt: skip
first = status["roc_auc"].dropna().iloc[0]
fig.add_hline(
    y=first - 0.05, line_dash="dash", line_color=theme.MUTED, row=1, col=1,
    annotation_text="alert line: 0.05 below first month", annotation_position="bottom right",
)  # fmt: skip
fig.add_trace(
    go.Bar(
        x=status["month"], y=status["trail_alert"].fillna(0), name="Order-mix features in alert", showlegend=False,
        marker_color=[STATUS_COLOR[s] for s in status["status"]],
        customdata=status["status"], hovertemplate="%{y} features · status %{customdata}",
    ),
    row=2, col=1,
)  # fmt: skip
fig.update_yaxes(title_text="ROC-AUC", row=1, col=1)
fig.update_yaxes(title_text="Features in alert", row=2, col=1, dtick=1)
st.plotly_chart(theme.style_fig(fig, 460), width="stretch")
c.caption(
    "Bars: red = alert, orange = watch, empty = stable. Look for the bars rising before the line falls: "
    "drift in the incoming orders is the early warning, ranking quality is the confirmation "
    "(which needs outcomes and so arrives late)."
)

# ------------------------------------------------------------------ heatmap
st.subheader("Which order features moved?")
trail = psi[(psi["baseline"] == "trailing") & (~psi["feature"].isin(expected))]
piv = trail.pivot_table(index="feature", columns="month", values="psi")
piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index].head(12)
hm = go.Figure(
    go.Heatmap(
        z=piv.values, x=[f"{m:%b %Y}" for m in piv.columns], y=piv.index, zmin=0, zmax=max(alert * 2, 0.6),
        colorscale=[[0, "#eef4fc"], [0.5, "#5598e7"], [1, "#0d366b"]],
        text=np.round(piv.values, 2), texttemplate="%{text}", hovertemplate="%{y} · %{x}: PSI %{z:.2f}<extra></extra>",
        colorbar={"title": "PSI"},
    )
)  # fmt: skip
hm.update_yaxes(autorange="reversed")
hm.update_layout(hovermode="closest")
st.plotly_chart(theme.style_fig(hm, max(320, 34 * len(piv) + 100), legend=False), width="stretch")
c.caption(
    f"PSI below {watch:.2f} is stable, {watch:.2f}-{alert:.2f} is a moderate shift, above {alert:.2f} is a "
    "major shift. Each month is compared with the three months before it."
)

# ------------------------------------------------------------------ drill-down
st.subheader("Look inside a feature")
features = list(piv.index) + sorted(set(trail["feature"]) - set(piv.index))
col_a, col_b = st.columns([1, 1])
feature = col_a.selectbox("Feature", features)
months = list(piv.columns)
month = col_b.selectbox("Month", months, index=len(months) - 1, format_func=lambda m: f"{m:%B %Y}")


@st.cache_data(show_spinner=False)
def _drill(feature: str, month: pd.Timestamp, trailing: int) -> tuple[pd.DataFrame, float]:
    t = data.features_all()
    cur = t[(t["approved_at"] >= month) & (t["approved_at"] < month + pd.DateOffset(months=1))]
    ref = t[
        (t["approved_at"] >= month - pd.DateOffset(months=trailing)) & (t["approved_at"] < month)
    ]
    if feature in CATEGORICAL_FEATURES:
        lv = sorted(set(ref[feature].astype(str)) | set(cur[feature].astype(str)))
        r = ref[feature].astype(str).value_counts(normalize=True).reindex(lv, fill_value=0)
        a = cur[feature].astype(str).value_counts(normalize=True).reindex(lv, fill_value=0)
        return pd.DataFrame(
            {"bin": lv, "reference": r.to_numpy(), "month": a.to_numpy()}
        ), psi_categorical(ref[feature], cur[feature])
    edges = np.unique(np.nanquantile(ref[feature].astype(float), np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref[feature].dropna(), bins=edges)[0] / max(ref[feature].notna().sum(), 1)
    a = np.histogram(cur[feature].dropna(), bins=edges)[0] / max(cur[feature].notna().sum(), 1)
    mid = [f"{edges[i]:.3g} to {edges[i + 1]:.3g}" for i in range(len(edges) - 1)]
    return pd.DataFrame({"bin": mid, "reference": r, "month": a}), psi_numeric(
        ref[feature], cur[feature]
    )


d, value = _drill(feature, month, mcfg["trailing_months"])
bar = go.Figure()
bar.add_trace(
    go.Bar(
        x=d["bin"],
        y=d["reference"],
        name=f"Previous {mcfg['trailing_months']} months",
        marker_color=theme.BASELINE,
    )
)
bar.add_trace(go.Bar(x=d["bin"], y=d["month"], name=f"{month:%B %Y}", marker_color=theme.BLUE))
bar.update_layout(barmode="group", hovermode="x unified")
bar.update_yaxes(tickformat=".0%", title_text="Share of orders")
bar.update_xaxes(tickangle=-30)
st.plotly_chart(theme.style_fig(bar, 340), width="stretch")
level = "major shift" if value >= alert else "moderate shift" if value >= watch else "stable"
c.caption(
    f"PSI = {value:.2f} ({level}). Bins are deciles of the reference period; bars show the share of orders in each."
)

# ------------------------------------------------------------------ state features
st.subheader("System-state features are tracked by level, not alerted")
t = data.features_all()
t = t.assign(month=t["approved_at"].dt.to_period("M").dt.start_time)
state_cols = [
    f for f in summ["expected_drift_features"] if f in t.columns and f not in CATEGORICAL_FEATURES
]
lvl = t[t["month"].isin(status["month"])].groupby("month")[state_cols].mean()
lvl.index = [f"{m:%b %Y}" for m in lvl.index]
st.dataframe(lvl.T.round(3), width="stretch")
c.caption(
    "These features describe the system's recent behaviour (late rates, seller tenure, calendar). The model "
    "uses them to adapt, so they are expected to move every month and are shown as levels."
)
c.callout(
    "<b>How far to trust this monitor.</b> It covers only a handful of months, the state-vs-order split "
    "was decided after seeing that raw PSI fires every month, and the thresholds are conventional, not "
    "tuned. Treat it as a worked design with an encouraging pattern, not as a validated early-warning system.",
    kind="warn",
)
