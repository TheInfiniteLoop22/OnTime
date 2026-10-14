"""Turn feature contributions into plain-English reason codes.

Pipeline: contributions (log-odds per feature) -> summed into reason GROUPS -> top positive and
top negative groups above a minimum magnitude -> sentence from a template and the order's own
feature values. All wording lives in configs/reason_templates.yaml.
"""

from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ontime.config import PROJECT_ROOT
from ontime.features.build import FEATURE_COLUMNS

TEMPLATE_PATH = PROJECT_ROOT / "configs" / "reason_templates.yaml"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]  # fmt: skip
REGION_NAMES = {
    "N": "North",
    "NE": "North-East",
    "CO": "Centre-West",
    "SE": "South-East",
    "S": "South",
}
DISPLAY_FIELDS = {
    "weekday_name", "month_name", "customer_region_name", "seller_region_name",
    "category_name", "route_phrase", "bf_phrase", "geo_status",
}  # fmt: skip


def load_templates(path: Path | None = None) -> dict:
    return yaml.safe_load((path or TEMPLATE_PATH).read_text(encoding="utf-8"))


def feature_to_group(templates: dict) -> dict[str, str]:
    """Invert the group definitions; raises if a feature is in two groups."""
    out: dict[str, str] = {}
    for g, spec in templates["groups"].items():
        for f in spec["features"]:
            if f in out:
                raise ValueError(f"feature {f!r} appears in groups {out[f]!r} and {g!r}")
            out[f] = g
    return out


def template_fields(template: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def group_contributions(contrib: pd.DataFrame, f2g: dict[str, str]) -> pd.DataFrame:
    """Sum feature contributions inside each reason group (columns = group names)."""
    return contrib.T.groupby(f2g).sum().T


def add_display_fields(X: pd.DataFrame) -> pd.DataFrame:
    """Raw feature values plus human-readable helper fields used by templates."""
    d = X.copy()
    for c in d.columns:
        if str(d[c].dtype) == "category":
            d[c] = d[c].astype(object)
    d["weekday_name"] = d["approve_dow"].map(dict(enumerate(WEEKDAYS)))
    d["month_name"] = d["approve_month"].map(dict(enumerate(MONTHS, start=1)))
    d["customer_region_name"] = d["customer_region"].map(REGION_NAMES).fillna("unknown")
    d["seller_region_name"] = d["seller_region"].map(REGION_NAMES).fillna("unknown")
    d["category_name"] = d["category_group"].astype(str).str.replace("_", " ")
    d["route_phrase"] = np.where(d["same_state"] == 1, "in the same state", "in different states")
    d["bf_phrase"] = np.where(d["is_black_friday_window"] == 1, " (Black Friday rush)", "")
    d["geo_status"] = np.where(d["geo_imputed"] == 1, "approximate (zip code not found)", "exact")
    return d


class _NA:
    """Formats as 'n/a' under ANY format spec (so '{x:,.0f}' never prints 'nan')."""

    def __format__(self, spec: str) -> str:
        return "n/a"

    def __str__(self) -> str:
        return "n/a"


_NA_VALUE = _NA()


class _SafeDict(dict):
    """Missing keys and NaN values render as 'n/a' rather than 'nan' or a crash."""

    def __missing__(self, key: str) -> _NA:
        return _NA_VALUE

    def __getitem__(self, key: str):  # noqa: ANN204
        v = super().__getitem__(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return _NA_VALUE
        return v


def _render(template: str, row: dict) -> str:
    try:
        return template.format_map(_SafeDict(row))
    except (ValueError, TypeError, KeyError):
        return template.split("{")[0].strip() or "n/a"


def build_reason_codes(
    grouped: pd.DataFrame, display: pd.DataFrame, templates: dict
) -> pd.DataFrame:
    """One row per order: lists of 'adds risk' and 'reduces risk' sentences + group ids."""
    min_abs = templates["min_abs_contribution"]
    n_pos, n_neg = templates["n_positive"], templates["n_negative"]
    groups = list(grouped.columns)
    G = grouped.to_numpy()
    rows = display.to_dict("records")
    order_desc = np.argsort(-G, axis=1)
    pos, neg, pos_g, neg_g = [], [], [], []
    for i, row in enumerate(rows):
        p_txt, p_ids, n_txt, n_ids = [], [], [], []
        for j in order_desc[i]:
            v = G[i, j]
            if v < min_abs or len(p_ids) >= n_pos:
                break
            p_ids.append(groups[j])
            p_txt.append(
                _render(group_text(templates, groups[j], "positive"), row) + " (adds risk)"
            )
        for j in order_desc[i][::-1]:
            v = G[i, j]
            if v > -min_abs or len(n_ids) >= n_neg:
                break
            n_ids.append(groups[j])
            n_txt.append(
                _render(group_text(templates, groups[j], "negative"), row) + " (reduces risk)"
            )
        pos.append(p_txt)
        neg.append(n_txt)
        pos_g.append(p_ids)
        neg_g.append(n_ids)
    return pd.DataFrame(
        {
            "risk_factors": pos,
            "protective_factors": neg,
            "risk_groups": pos_g,
            "protective_groups": neg_g,
        },
        index=grouped.index,
    )


def validate_templates(templates: dict) -> list[str]:
    """Problems that would break rendering: unmapped features, unknown placeholders."""
    problems = []
    f2g = feature_to_group(templates)
    for f in FEATURE_COLUMNS:
        if f not in f2g:
            problems.append(f"feature without a reason group: {f}")
    for f in f2g:
        if f not in FEATURE_COLUMNS:
            problems.append(f"group lists unknown feature: {f}")
    allowed = set(FEATURE_COLUMNS) | DISPLAY_FIELDS
    for g, spec in templates["groups"].items():
        for kind in ("positive", "negative"):
            if kind not in spec and "text" not in spec:
                problems.append(f"{g}: needs `text` or a {kind} template")
                continue
            for field in template_fields(group_text(templates, g, kind)) - allowed:
                problems.append(f"{g}.{kind}: unknown placeholder {{{field}}}")
    return problems


def group_text(templates: dict, group: str, kind: str) -> str:
    """Template for a group: `positive`/`negative` override, otherwise the shared `text`."""
    spec = templates["groups"][group]
    return spec.get(kind) or spec["text"]
