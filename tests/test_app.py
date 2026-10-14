"""Smoke tests: every Streamlit page renders without exceptions and shows key elements.

Uses Streamlit's headless AppTest. Skipped if the pipeline artifacts are not built.
"""

import sys
from pathlib import Path

import pytest

from ontime.config import PROJECT_ROOT, project_path

APP = PROJECT_ROOT / "app"
READY = all(
    p.exists()
    for p in [
        project_path("data/processed/features.parquet"),
        project_path("reports/results.json"),
        project_path("reports/impact.json"),
        project_path("reports/test_reason_codes.parquet"),
        project_path("reports/drift_status.csv"),
        project_path("models/logreg.joblib"),
        project_path("models/ebm.joblib"),
        project_path("models/lightgbm.joblib"),
    ]
)
pytestmark = pytest.mark.skipif(not READY, reason="needs pipeline artifacts (make all)")

PAGES = [
    "overview",
    "risk_explorer",
    "ops_queue",
    "seller_scorecard",
    "model_insights",
    "drift_monitor",
]


@pytest.fixture(scope="module", autouse=True)
def _app_on_path():
    sys.path.insert(0, str(APP))
    yield
    sys.path.remove(str(APP))


def _run(page: str):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP / "views" / f"{page}.py"), default_timeout=240).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = _run(page)
    assert not at.exception, [str(e.value)[:200] for e in at.exception]


def test_home_entrypoint_compiles():
    import py_compile

    py_compile.compile(str(APP / "Home.py"), doraise=True)


def test_overview_numbers_come_from_artifacts():
    import json

    at = _run("overview")
    text = " ".join(m.value for m in at.markdown)
    res = json.loads(Path(project_path("reports/results.json")).read_text(encoding="utf-8"))
    assert f"{res['population']['late_rate'] * 100:.1f}%" in text


def test_risk_explorer_whatif_changes_the_estimate():
    at = _run("risk_explorer")
    assert len(at.metric) >= 3
    before = at.metric[1].value  # what-if estimate with no edits
    sliders = [s for s in at.slider if "Extend the promised" in s.label]
    assert sliders, "what-if slider missing"
    sliders[0].set_value(15).run()
    assert not at.exception
    after = [m for m in at.metric if m.label == "What-if risk estimate"][0].value
    assert after != before


def test_ops_queue_capacity_changes_queue_size():
    at = _run("ops_queue")
    sizes = []
    for k in (5, 20):
        cap = [s for s in at.slider if "Ops capacity" in s.label][0]
        cap.set_value(k).run()
        assert not at.exception
        # first KPI cards are markdown; the queue size card carries the label
        card = [m.value for m in at.markdown if "Queue size" in m.value][0]
        sizes.append(card)
    assert sizes[0] != sizes[1]
