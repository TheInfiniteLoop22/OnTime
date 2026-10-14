"""Business-impact analysis: how much does late delivery lower customer review scores?

Reviews are used HERE ONLY (an outcome we study), never as a model feature.

Honest framing: all estimates are *adjusted associations*, not randomized causal effects.
Unobserved seller quality, product quality and customer expectations can influence both
lateness and the score. Seller fixed effects (below) address the biggest of these threats
partially; they do not remove it.
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.miscmodels.ordinal_model import OrderedModel  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from ontime.analysis.eda import (  # noqa: E402
    BLUE,
    GREY,
    RED,
    _save,
    add_eda_columns,
    load_orders,
    set_style,
)
from ontime.config import load_config, project_path  # noqa: E402
from ontime.data.load import load_table  # noqa: E402

log = logging.getLogger(__name__)

TOP_CATEGORIES = 12  # categories beyond the most common ones are pooled into "other"


# --------------------------------------------------------------------------- data preparation
def one_review_per_order(reviews: pd.DataFrame) -> pd.DataFrame:
    """Keep the LAST review per order (latest answer timestamp).

    Why: ~0.5% of orders have 2-3 reviews (customers revise). The final review is the customer's
    settled opinion. Ties are broken by review_creation_date then review_id so the choice is
    deterministic.
    """
    r = reviews.sort_values(
        ["order_id", "review_answer_timestamp", "review_creation_date", "review_id"]
    )
    return r.drop_duplicates("order_id", keep="last")[
        ["order_id", "review_score", "review_creation_date", "review_answer_timestamp"]
    ]


def build_analysis_frame(orders: pd.DataFrame | None = None) -> pd.DataFrame:
    """Orders (delivered) joined to one review each, plus control variables."""
    o = add_eda_columns(load_orders() if orders is None else orders)
    d = o.merge(one_review_per_order(load_table("reviews")), on="order_id", how="inner")
    d["is_late"] = d["is_late"].astype(int)
    d["log_price"] = np.log1p(d["total_price"])
    d["log_freight"] = np.log1p(d["total_freight"])
    d["log_distance"] = np.log1p(d["distance_km"])
    cat = d["top_item_category"].fillna("unknown")
    top = cat.value_counts().head(TOP_CATEGORIES).index
    d["category_group"] = cat.where(cat.isin(top), "other")
    d["month_str"] = d["approve_month"].dt.strftime("%Y-%m")
    # `review_creation_date` is when the SURVEY was sent; `review_answer_timestamp` is when the
    # customer answered. A review answered before the parcel arrived is a reaction to the broken
    # promise ("it has not come yet"), not to the delivery experience itself.
    d["answered_before_delivery"] = (
        d["review_answer_timestamp"] < d["order_delivered_customer_date"]
    )
    d["low_score"] = (d["review_score"] <= 2).astype(int)
    return d


# --------------------------------------------------------------------------- 1. descriptive
def descriptive(d: pd.DataFrame) -> pd.DataFrame:
    """Score distribution and mean for late vs on-time orders."""
    g = d.groupby("is_late")["review_score"]
    out = pd.DataFrame(
        {
            "orders": g.size(),
            "mean_score": g.mean(),
            "median_score": g.median(),
            "share_1_star": d.groupby("is_late")["review_score"].apply(lambda s: (s == 1).mean()),
            "share_1_or_2_star": d.groupby("is_late")["low_score"].mean(),
            "share_5_star": d.groupby("is_late")["review_score"].apply(lambda s: (s == 5).mean()),
        }
    )
    out.index = out.index.map({0: "on time", 1: "late"})
    return out


def score_distribution(d: pd.DataFrame) -> pd.DataFrame:
    """Share of each star rating within late and on-time orders."""
    t = pd.crosstab(d["review_score"], d["is_late"], normalize="columns")
    t.columns = ["on time", "late"]
    return t


# --------------------------------------------------------------------------- 2. tests + CIs
def mann_whitney(d: pd.DataFrame) -> dict:
    """Mann-Whitney U on review score, late vs on time.

    Why not a t-test: scores are ordinal 1-5 and heavily skewed (most are 5). Mann-Whitney only
    uses ranks. We also report the common-language effect size P(late score < on-time score) +
    0.5 * P(tie), computed as 1 - U/(n1*n2) for the late group.
    """
    late = d.loc[d["is_late"] == 1, "review_score"].to_numpy()
    ontime = d.loc[d["is_late"] == 0, "review_score"].to_numpy()
    res = stats.mannwhitneyu(late, ontime, alternative="two-sided")
    u_late = float(res.statistic)
    auc = u_late / (len(late) * len(ontime))  # P(late > on-time) + 0.5 P(tie)
    return {
        "u_statistic": u_late,
        "p_value": float(res.pvalue),
        "prob_late_scores_higher_than_ontime": float(auc),
        "prob_ontime_scores_higher_than_late": float(1 - auc),
        "n_late": int(len(late)),
        "n_ontime": int(len(ontime)),
    }


def bootstrap_differences(d: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Percentile bootstrap 95% CIs for (late - on-time) mean score and low-score share.

    Resampling is stratified by group so every replicate keeps the original late/on-time sizes.
    """
    rng = np.random.default_rng(seed)
    s_late = d.loc[d["is_late"] == 1, "review_score"].to_numpy()
    s_on = d.loc[d["is_late"] == 0, "review_score"].to_numpy()
    mean_diff, low_diff = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        a = rng.choice(s_late, len(s_late))
        c = rng.choice(s_on, len(s_on))
        mean_diff[b] = a.mean() - c.mean()
        low_diff[b] = (a <= 2).mean() - (c <= 2).mean()
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return {
        "mean_score_diff": float(s_late.mean() - s_on.mean()),
        "mean_score_diff_ci95": pct(mean_diff),
        "low_score_share_diff": float((s_late <= 2).mean() - (s_on <= 2).mean()),
        "low_score_share_diff_ci95": pct(low_diff),
        "n_bootstrap": n_boot,
    }


# --------------------------------------------------------------------------- 3. adjusted models
CONTROLS = (
    "log_price + log_freight + n_items + log_distance + same_state"
    " + C(category_group) + C(customer_state) + C(month_str)"
)


def _coef(res, name: str = "is_late") -> dict:
    ci = res.conf_int().loc[name]
    return {
        "coef": float(res.params[name]),
        "ci95": [float(ci.iloc[0]), float(ci.iloc[1])],
        "p_value": float(res.pvalues[name]),
    }


def ols_adjusted(d: pd.DataFrame) -> dict:
    """OLS of score on is_late + controls; SEs clustered by seller.

    Clustering matters: orders from the same seller are not independent, and unclustered SEs
    would be too narrow.
    """
    groups = pd.factorize(d["primary_seller_id"])[0]
    base = smf.ols("review_score ~ is_late", d).fit(cov_type="cluster", cov_kwds={"groups": groups})
    full = smf.ols(f"review_score ~ is_late + {CONTROLS}", d).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    return {
        "unadjusted": {**_coef(base), "n": int(base.nobs)},
        "adjusted": {**_coef(full), "n": int(full.nobs), "r_squared": float(full.rsquared)},
    }


def seller_fixed_effects(d: pd.DataFrame) -> dict:
    """Within-seller estimate: compare late vs on-time orders of the SAME seller.

    Demeaning score and is_late by seller (the within transformation) removes anything constant
    per seller (quality, product mix, brand). Sellers with a single order carry no information
    and drop out automatically (their demeaned values are 0).
    """
    x = d.copy()
    for c in ["review_score", "is_late", "log_price", "log_freight", "n_items", "log_distance"]:
        x[c + "_w"] = x[c] - x.groupby("primary_seller_id")[c].transform("mean")
    # keep sellers with both late and on-time orders: only they identify the effect
    var = x.groupby("primary_seller_id")["is_late"].transform("nunique") > 1
    x = x[var]
    groups = pd.factorize(x["primary_seller_id"])[0]
    res = smf.ols(
        "review_score_w ~ 0 + is_late_w + log_price_w + log_freight_w + n_items_w + log_distance_w",
        x,
    ).fit(cov_type="cluster", cov_kwds={"groups": groups})
    out = _coef(res, "is_late_w")
    out.update({"n_orders": int(len(x)), "n_sellers": int(x["primary_seller_id"].nunique())})
    return out


def timing_of_review(d: pd.DataFrame) -> pd.DataFrame:
    """Mean score by lateness x whether the customer answered before the parcel arrived."""
    g = d.groupby(["is_late", "answered_before_delivery"])["review_score"]
    t = g.agg(["size", "mean"]).reset_index()
    t["group"] = t["is_late"].map({0: "on time", 1: "late"}) + np.where(
        t["answered_before_delivery"], ", reviewed BEFORE arrival", ", reviewed after arrival"
    )
    t["share_of_group_orders"] = t["size"] / t.groupby("is_late")["size"].transform("sum")
    return t.rename(columns={"size": "orders", "mean": "mean_score"})[
        ["group", "orders", "share_of_group_orders", "mean_score"]
    ]


def sensitivity_answered_after_delivery(d: pd.DataFrame) -> dict:
    """Re-run the adjusted OLS only on reviews answered AFTER the parcel arrived.

    This isolates the effect of a late *delivery experience* from the reaction to a missing
    parcel. It is smaller, and is the conservative number to quote.
    """
    sub = d[~d["answered_before_delivery"]]
    groups = pd.factorize(sub["primary_seller_id"])[0]
    res = smf.ols(f"review_score ~ is_late + {CONTROLS}", sub).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    return {**_coef(res), "n": int(res.nobs), "share_of_reviews_kept": float(len(sub) / len(d))}


def ordered_logit(d: pd.DataFrame, max_rows: int = 40000, seed: int = 42) -> dict:
    """Ordered logit (proportional odds) of the 1-5 score; treats the score as ordinal.

    Robustness check for OLS (which treats 1->2 and 4->5 as equal steps). To keep runtime sane,
    a seeded random subsample of `max_rows` orders is used; sample size is reported.
    Controls are a compact subset (no month/state dummies) because many dummies make the
    optimizer slow without changing the story; OLS carries the full control set.
    """
    s = d.sample(min(max_rows, len(d)), random_state=seed)
    X = pd.get_dummies(
        s[["is_late", "log_price", "log_freight", "n_items", "log_distance", "same_state"]].assign(
            same_state=s["same_state"].astype(int)
        ),
    ).astype(float)
    mod = OrderedModel(s["review_score"].astype(int), X, distr="logit")
    res = mod.fit(method="bfgs", disp=False, maxiter=300)
    ci = res.conf_int().loc["is_late"]
    coef = float(res.params["is_late"])
    return {
        "log_odds_coef": coef,
        "odds_ratio": float(np.exp(coef)),
        "odds_ratio_ci95": [float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1]))],
        "p_value": float(res.pvalues["is_late"]),
        "n": int(len(s)),
        "converged": bool(res.mle_retvals.get("converged", True)),
    }


# --------------------------------------------------------------------------- 4. dose-response
DOSE_BINS = [-np.inf, -8, -1, 0, 3, 7, 14, np.inf]
DOSE_LABELS = [
    ">=8 days early",
    "1-7 days early",
    "on the promised day",
    "1-3 days late",
    "4-7 days late",
    "8-14 days late",
    ">14 days late",
]


def dose_response(d: pd.DataFrame) -> pd.DataFrame:
    """Mean score by how many days the delivery date was before/after the estimate."""
    x = d.assign(timing=pd.cut(d["days_vs_estimate"], DOSE_BINS, labels=DOSE_LABELS, right=True))
    g = x.groupby("timing", observed=True)["review_score"]
    t = g.agg(["size", "mean", "std"]).reset_index()
    t["sem"] = t["std"] / np.sqrt(t["size"])
    t["ci_low"], t["ci_high"] = t["mean"] - 1.96 * t["sem"], t["mean"] + 1.96 * t["sem"]
    t["share_1_or_2_star"] = x.groupby("timing", observed=True)["low_score"].mean().to_numpy()
    return t.rename(columns={"size": "orders", "mean": "mean_score"}).drop(columns=["std", "sem"])


def plot_impact(d: pd.DataFrame, dose: pd.DataFrame) -> None:
    """Two figures: score distribution late vs on time, and the dose-response curve."""
    set_style()
    dist = score_distribution(d)
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    x = np.arange(1, 6)
    ax.bar(x - w / 2, dist["on time"] * 100, w, label="on time", color=BLUE)
    ax.bar(x + w / 2, dist["late"] * 100, w, label="late", color=RED)
    ax.set_xlabel("Review score (stars)")
    ax.set_ylabel("Share of orders (%)")
    ax.set_title("Late deliveries are rated far lower")
    ax.legend()
    _save(fig, "impact_score_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(dose))
    ax.errorbar(
        xs,
        dose["mean_score"],
        yerr=[dose["mean_score"] - dose["ci_low"], dose["ci_high"] - dose["mean_score"]],
        marker="o",
        color=RED,
        capsize=3,
    )
    ax.set_xticks(xs, dose["timing"], rotation=25, ha="right")
    ax.axvline(2.5, color=GREY, ls="--")
    ax.set_ylabel("Mean review score")
    ax.set_title("Dose-response: the later the parcel, the lower the score (95% CI)")
    for xi, n in zip(xs, dose["orders"], strict=True):
        ax.annotate(f"n={n:,}", (xi, 1.05), ha="center", fontsize=8, color=GREY)
    ax.set_ylim(1, 5)
    _save(fig, "impact_dose_response.png")


# --------------------------------------------------------------------------- orchestration
def run_impact(write: bool = True) -> dict:
    """Run the full analysis and write reports/impact.json (+ figures)."""
    cfg = load_config()
    d = build_analysis_frame()
    log.info(
        "impact | analysis frame: %d orders with a review (%d late)", len(d), d["is_late"].sum()
    )

    desc = descriptive(d)
    dose = dose_response(d)
    plot_impact(d, dose)
    lo, hi = proportion_confint(int(d["is_late"].sum()), len(d), method="wilson")

    result = {
        "framing": (
            "Adjusted ASSOCIATION between lateness and review score; "
            "not a randomized causal effect."
        ),
        "population": {
            "orders_with_review": int(len(d)),
            "late_orders": int(d["is_late"].sum()),
            "late_rate": float(d["is_late"].mean()),
            "late_rate_ci95": [float(lo), float(hi)],
        },
        "descriptive": desc.round(4).to_dict(orient="index"),
        "mann_whitney": mann_whitney(d),
        "bootstrap": bootstrap_differences(d, cfg["tuning"]["bootstrap_resamples"], cfg["seed"]),
        "ols": ols_adjusted(d),
        "seller_fixed_effects": seller_fixed_effects(d),
        "review_timing": timing_of_review(d).round(4).to_dict(orient="records"),
        "sensitivity_answered_after_delivery": sensitivity_answered_after_delivery(d),
        "ordered_logit": ordered_logit(d, seed=cfg["seed"]),
        "dose_response": dose.round(4).to_dict(orient="records"),
    }
    if write:
        out = project_path(cfg["paths"]["reports_dir"]) / "impact.json"
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        log.info("impact | wrote %s", out)
    return result
