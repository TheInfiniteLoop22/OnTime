"""README / model-card generator: no silent blanks, no stale placeholders."""

import pytest

from ontime.config import PROJECT_ROOT, project_path
from ontime.reporting.render import PLACEHOLDER, render_template

READY = all(
    project_path(p).exists()
    for p in [
        "reports/results.json",
        "reports/validation_results.json",
        "reports/impact.json",
        "reports/explain_summary.json",
        "reports/test_reason_codes.parquet",
        "data/processed/features.parquet",
    ]
)


def test_render_fills_known_keys():
    assert render_template("a {{x}} b {{ y }}", {"x": "1", "y": "2"}) == "a 1 b 2"


def test_render_raises_on_missing_value():
    with pytest.raises(KeyError):
        render_template("value: {{ not_there }}", {"x": "1"})


def test_templates_only_use_known_placeholder_syntax():
    for name in ("README.template.md", "model_card.template.md"):
        text = (PROJECT_ROOT / "src" / "ontime" / "reporting" / "templates" / name).read_text(
            encoding="utf-8"
        )
        assert "{{" not in PLACEHOLDER.sub("", text), "malformed placeholder in " + name


@pytest.mark.skipif(not READY, reason="needs pipeline artifacts (make all)")
def test_generated_documents_have_no_unfilled_placeholders():
    from ontime.reporting.render import build_reports

    out = build_reports(write=False)
    assert set(out) == {"README.md", "reports/model_card.md"}
    for rel, text in out.items():
        assert "{{" not in text and "}}" not in text, rel
    assert "PR-AUC" in out["README.md"]
    assert "Intended use" in out["reports/model_card.md"]
