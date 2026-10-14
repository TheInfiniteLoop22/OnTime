"""Design tokens, global CSS and the shared Plotly look for the OnTime app.

Colours follow a validated categorical palette (blue, orange, aqua, violet in fixed order) plus a
reserved status palette for risk bands. Status colours always travel with an icon and a label,
never colour alone. Light theme only (set in .streamlit/config.toml).
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ---- tokens ---------------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE, PAGE = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f9f9f7"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
CRITICAL, SERIOUS, GOOD = "#d03b3b", "#ec835a", "#0ca30c"
PALE_BLUE = "#eef4fc"

MODEL_COLORS = {
    "logreg": BLUE,
    "lightgbm": ORANGE,
    "ebm": AQUA,
    "random_forest": VIOLET,
    "rule": MUTED,
    "prior_rate": BASELINE,
}
BAND_STYLE = {
    "High": {"color": CRITICAL, "icon": "▲▲", "label": "High risk", "bg": "#fbeaea"},
    "Elevated": {"color": "#c2552b", "icon": "▲", "label": "Elevated risk", "bg": "#fdf0ea"},
    "Normal": {"color": INK2, "icon": "●", "label": "Normal", "bg": "#f0efec"},
}

CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1240px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h1 {{ font-weight: 680; }}
[data-testid="stSidebar"] {{ border-right: 1px solid rgba(11,11,11,0.08); }}
#MainMenu, footer {{ visibility: hidden; }}

.ot-hero {{ padding: 6px 0 14px 0; }}
.ot-hero h1 {{ margin: 0 0 4px 0; font-size: 2.1rem; }}
.ot-hero p {{ margin: 0; color: {INK2}; font-size: 1.05rem; max-width: 62rem; }}

.ot-card {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10); border-radius: 12px;
            padding: 16px 18px; min-height: 188px; }}
.ot-kpi-label {{ font-size: .74rem; color: {INK2}; text-transform: uppercase;
                 letter-spacing: .05em; font-weight: 600; }}
.ot-kpi-value {{ font-size: 2.05rem; font-weight: 680; line-height: 1.15; margin: 4px 0 2px 0; }}
.ot-kpi-sub {{ font-size: .85rem; color: {INK2}; }}

.ot-badge {{ display: inline-flex; align-items: center; gap: 8px; padding: 4px 12px;
             border-radius: 999px; font-weight: 650; font-size: .92rem; }}
.ot-reason {{ border-left: 4px solid; padding: 9px 12px; margin: 7px 0; background: {SURFACE};
              border-radius: 6px; font-size: .96rem; line-height: 1.35;
              border-top: 1px solid rgba(11,11,11,0.06); border-right: 1px solid rgba(11,11,11,0.06);
              border-bottom: 1px solid rgba(11,11,11,0.06); }}
.ot-reason .tag {{ font-weight: 650; font-size: .78rem; text-transform: uppercase;
                   letter-spacing: .04em; margin-right: 6px; }}
.ot-callout {{ border-left: 4px solid {BLUE}; background: {PALE_BLUE}; padding: 12px 14px;
               border-radius: 6px; font-size: .95rem; color: {INK}; }}
.ot-callout.warn {{ border-left-color: {SERIOUS}; background: #fdf0ea; }}
.ot-cap {{ color: {INK2}; font-size: .86rem; margin: -4px 0 14px 0; }}
.ot-step {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10); border-radius: 12px;
            padding: 14px 16px; height: 100%; }}
.ot-step .n {{ display:inline-block; width: 26px; height: 26px; line-height: 26px; text-align:center;
               border-radius: 50%; background: {BLUE}; color: #fff; font-weight: 700; margin-bottom: 8px; }}
.ot-step h4 {{ margin: 0 0 4px 0; font-size: 1.02rem; }}
.ot-step p {{ margin: 0; color: {INK2}; font-size: .9rem; }}
.ot-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .86rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 14px; border-radius: 8px 8px 0 0; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    """Shared Plotly look: recessive grid, thin lines, horizontal legend above, no chart junk."""
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 36 if legend else 12, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "system-ui, -apple-system, 'Segoe UI', sans-serif",
            "color": INK2,
            "size": 13,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        showlegend=legend,
        hoverlabel={"bgcolor": SURFACE, "font_color": INK, "bordercolor": BASELINE},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, zeroline=False, ticks="outside",
                     tickcolor=BASELINE)  # fmt: skip
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, zeroline=False)
    return fig
