"""OnTime - entrypoint and navigation. Run with:  streamlit run app/Home.py  (make app)

Five pages (PROJECT_SPEC 6.8): Overview, Order Risk Explorer, Ops Queue & Policy Simulator,
Seller Scorecard, Model Insights. All numbers come from generated artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="OnTime - late-delivery risk",
    page_icon=":material/local_shipping:",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("views/risk_explorer.py", title="Order risk explorer", icon=":material/search:"),
    st.Page("views/ops_queue.py", title="Ops queue & policy", icon=":material/checklist:"),
    st.Page("views/seller_scorecard.py", title="Seller scorecard", icon=":material/storefront:"),
    st.Page("views/model_insights.py", title="Model insights", icon=":material/insights:"),
    st.Page("views/drift_monitor.py", title="Drift monitor", icon=":material/monitoring:"),
]
nav = st.navigation(pages)

span = ""
try:  # derived from the data, not typed
    from ontime_ui import data as _data

    _f = _data.features_all()
    span = f", orders {_f['approved_at'].dt.year.min()}-{_f['approved_at'].dt.year.max()}"
except Exception:  # artifacts not built yet: the pages show a friendly message
    pass

with st.sidebar:
    st.caption(
        f"Data: Olist Brazilian E-Commerce (CC BY-NC-SA 4.0){span}. "
        "Models were trained on the past and judged on a later test period."
    )

nav.run()
