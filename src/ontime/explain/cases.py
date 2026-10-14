"""Helpers for the explainability case-study notebook (keeps logic out of the notebook).

Case selection is mechanical, not hand-picked:
  true positive  : the late order the primary model was MOST sure about
  false positive : the on-time order the primary model scored HIGHEST
  false negative : the late order the primary model scored LOWEST
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ontime.config import load_config, project_path
from ontime.explain.contributions import EXTRACTORS
from ontime.models.data import load_feature_table, prepare_frame
from ontime.models.splits import select_split

MODELS = ["logreg", "lightgbm", "ebm"]


def load_context() -> dict:
    """Test frames, artifacts, per-model contributions and the exported reason codes."""
    cfg = load_config()
    rep = project_path(cfg["paths"]["reports_dir"])
    arts = {
        m: joblib.load(project_path(cfg["paths"]["models_dir"]) / f"{m}.joblib") for m in MODELS
    }
    levels = arts["logreg"]["levels"]
    test = select_split(load_feature_table(), "test", final_evaluation=True).reset_index(drop=True)
    X = prepare_frame(test, levels)
    contribs = {m: EXTRACTORS[m](arts[m]["model"], X) for m in MODELS}
    codes = pd.read_parquet(rep / "test_reason_codes.parquet")
    return {"test": test, "X": X, "arts": arts, "contribs": contribs, "codes": codes}


def pick_cases(codes: pd.DataFrame, primary: str = "logreg") -> dict[str, str]:
    p = codes[codes["model"] == primary]
    return {
        "true_positive": p[p["is_late"] == 1].nlargest(1, "probability")["order_id"].iloc[0],
        "false_positive": p[p["is_late"] == 0].nlargest(1, "probability")["order_id"].iloc[0],
        "false_negative": p[p["is_late"] == 1].nsmallest(1, "probability")["order_id"].iloc[0],
    }


def case_summary(ctx: dict, order_id: str) -> dict:
    """Everything the notebook needs about one order."""
    test, codes = ctx["test"], ctx["codes"]
    idx = int(np.flatnonzero(test["order_id"].to_numpy() == order_id)[0])
    c = codes[codes["order_id"] == order_id].set_index("model")
    return {
        "idx": idx,
        "row": test.iloc[idx],
        "probability": c["probability"].to_dict(),
        "risk": {m: list(c.loc[m, "risk_factors"]) for m in c.index},
        "protective": {m: list(c.loc[m, "protective_factors"]) for m in c.index},
        "is_late": int(test.iloc[idx]["is_late"]),
    }
