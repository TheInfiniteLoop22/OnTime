"""Reason codes for the stage-2 (carrier-handover) model.

Same method as stage 1 (exact log-odds decomposition -> grouped -> plain-English sentences) so the
two scores can be read side by side. The only difference is the feature list: stage-2 groups
(`stage2_groups` in configs/reason_templates.yaml) are added to the stage-1 groups.
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from ontime.config import load_config, project_path
from ontime.explain.contributions import EXTRACTORS
from ontime.explain.reason_codes import (
    DISPLAY_FIELDS,
    add_display_fields,
    build_reason_codes,
    feature_to_group,
    group_contributions,
    group_text,
    load_templates,
    template_fields,
)
from ontime.features.build import FEATURE_COLUMNS
from ontime.models.data import prepare_frame
from ontime.models.splits import select_split
from ontime.stage2.features import STAGE2_FEATURES

log = logging.getLogger(__name__)
ALL_COLUMNS = FEATURE_COLUMNS + STAGE2_FEATURES


def stage2_templates(templates: dict | None = None) -> dict:
    """Stage-1 templates plus the stage-2 groups, as one templates dict."""
    t = dict(templates or load_templates())
    extra = t.get("stage2_groups", {})
    clash = set(extra) & set(t["groups"])
    if clash:
        raise ValueError(f"stage-2 group names clash with stage-1 groups: {sorted(clash)}")
    t["groups"] = {**t["groups"], **extra}
    return t


def validate_stage2_templates(templates: dict) -> list[str]:
    """Every stage-2 model feature in exactly one group; placeholders are known fields."""
    problems = []
    f2g = feature_to_group(templates)  # raises on a feature in two groups
    problems += [f"feature without a reason group: {f}" for f in ALL_COLUMNS if f not in f2g]
    problems += [f"group lists unknown feature: {f}" for f in f2g if f not in ALL_COLUMNS]
    allowed = set(ALL_COLUMNS) | DISPLAY_FIELDS
    for g in templates["groups"]:
        for kind in ("positive", "negative"):
            for field in template_fields(group_text(templates, g, kind)) - allowed:
                problems.append(f"{g}.{kind}: unknown placeholder {{{field}}}")
    return problems


def explain_rows(artifact: dict, df: pd.DataFrame, templates: dict | None = None) -> dict:
    """Contributions, grouped contributions and reason codes for rows of the stage-2 table."""
    templates = stage2_templates(templates)
    problems = validate_stage2_templates(templates)
    if problems:
        raise ValueError("stage-2 reason template problems:\n" + "\n".join(problems))
    cols = artifact["columns"]
    X = prepare_frame(df, artifact["levels"])
    for f in STAGE2_FEATURES:
        X[f] = df[f].to_numpy()
    X = X[cols]
    contrib, base = EXTRACTORS[artifact["kind"]](artifact["model"], X, columns=cols)
    p = np.clip(artifact["model"].predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    err = float(np.abs(contrib.sum(axis=1).to_numpy() + base - np.log(p / (1 - p))).max())
    if err > 1e-4:
        raise RuntimeError(f"stage-2 contributions do not add up to the model output ({err})")
    grouped = group_contributions(contrib, feature_to_group(templates))
    display = add_display_fields(df[ALL_COLUMNS].reset_index(drop=True))
    reasons = build_reason_codes(grouped.reset_index(drop=True), display, templates)
    return {"contrib": contrib, "grouped": grouped, "reasons": reasons, "max_identity_error": err}


def run_stage2_explain(write: bool = True) -> dict:
    """Reason codes for every test order (the model was already scored once on these months)."""
    cfg = load_config()
    art = joblib.load(project_path(cfg["paths"]["models_dir"]) / "stage2.joblib")
    table = pd.read_parquet(project_path(cfg["paths"]["processed_dir"]) / "features_stage2.parquet")
    test = select_split(table, "test", final_evaluation=True).reset_index(drop=True)
    out = explain_rows(art, test)
    reasons = out["reasons"].assign(order_id=test["order_id"].to_numpy(), t1=test["t1"].to_numpy())
    share = out["grouped"].abs().mean().sort_values(ascending=False)
    summary = {
        "orders": int(len(test)),
        "model": art["kind"],
        "max_additive_identity_error": out["max_identity_error"],
        "mean_abs_group_contribution": {k: float(v) for k, v in share.items()},
        "orders_with_a_stage2_reason": float(
            reasons["risk_groups"].map(lambda gs: any(g in cfg_stage2_groups() for g in gs)).mean()
        ),
    }
    log.info(
        "stage2 explain | %d orders, identity error %.1e", len(test), out["max_identity_error"]
    )
    if write:
        rep = project_path(cfg["paths"]["reports_dir"])
        reasons.to_parquet(rep / "stage2_reason_codes.parquet", index=False)
        import json

        (rep / "stage2_explain_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8", newline="\n"
        )
    return summary


def cfg_stage2_groups() -> set[str]:
    return set(load_templates().get("stage2_groups", {}))
