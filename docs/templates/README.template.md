# OnTime - Explainable Late-Delivery Risk Engine

[![CI](https://github.com/TheInfiniteLoop22/OnTime/actions/workflows/ci.yml/badge.svg)](https://github.com/TheInfiniteLoop22/OnTime/actions/workflows/ci.yml)

**Predict, at the moment an e-commerce order is approved, whether it will miss the date promised to
the customer; explain every prediction in plain English; and turn the scores into a
capacity-aware plan for which orders operations should act on today.**

Built on the public [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
with classical ML only (logistic regression, EBM, LightGBM, random forest). No deep learning.

> This file is **generated** by `make report` from the project's result files, so every number
> below was produced by code in this repository (see `docs/templates/README.template.md`).

---

## Why it matters

Customers punish broken delivery promises. In this data, late orders are rated
**{{star_conservative}} to {{star_full}} stars lower** than on-time orders after adjusting for price,
category, region, season and distance (an adjusted association, not a proven causal effect), and
**{{late_low_share}}** of late orders end in a 1-2 star review versus **{{ontime_low_share}}** of
on-time ones.

The twist: **{{late_before_share}} of late orders were reviewed before the parcel arrived.**
Most of the damage happens while the customer is still waiting, so operations must act *before* the
promised date, which is exactly what a prediction at approval time enables.

## Headline results (test period, never used for tuning)

| | |
|---|---|
| Test period | {{test_from}} to {{test_to}} - {{test_orders}} orders, {{test_late}} late ({{test_late_rate}}) |
| Primary model | **{{primary}}** (chosen by validation results *before* the test set was opened) |
| Ranking quality | PR-AUC **{{primary_pr_auc}}** {{primary_pr_ci}} vs {{base_rate}} for random ranking ({{pr_ratio}} better) |
| Capacity value | Acting on the riskiest 10% of each day's orders: precision **{{precision10}}**, lift **{{lift10}}x** {{lift10_ci}} |
| Beats a hand-written rule? | {{rule_verdict}} |
| Glass-box vs black-box | {{gap_sentence}} |
| Simulated benefit | {{policy_sentence}} |
| **Stage 2: re-score at carrier handover** | PR-AUC **{{s2_pr_auc}}** {{s2_pr_ci}} on the same orders ({{s1_same_pr_auc}} at approval), ROC-AUC {{s2_roc}}, lift **{{s2_lift10}}x** at 10% capacity; late orders have a median {{s2_window_median_days}} days left to act |

{{results_table}}

PR-AUC is judged against the base rate (random ranking scores the late rate), and intervals are 95%
bootstrap intervals. Full detail: [`reports/results.md`](reports/results.md).

## How it works

```mermaid
flowchart LR
    A[9 Olist CSVs] --> B[Validate with pandera]
    B --> C[DuckDB SQL<br/>one row per order]
    C --> D[Leak-safe features<br/>{{n_features}} features]
    D --> E[Time-based split<br/>train / validation / test]
    E --> F[Baselines + LR, EBM,<br/>LightGBM, RF]
    F --> G[Tune on validation,<br/>calibrate later in time]
    G --> H[Evaluate on test once]
    H --> I[Reason codes<br/>SHAP, EBM, coefficients]
    H --> S[Stage 2: re-score at<br/>carrier handover]
    I --> J[Streamlit app<br/>+ policy simulator]
    S --> J
    J --> M[Drift monitor]
    C --> K[Impact analysis<br/>lateness to review score]
```

### What makes it rigorous
- **No leakage.** Features use only what is known at approval time. Seller/route history counts a
  past order only if it was *delivered before* the current order's approval time (approval time
  alone is not enough). Five required leakage tests plus a property test against a brute-force loop.
- **Time-based splits only.** Train past -> validation -> test future; tuning and calibration use
  different, time-ordered slices of validation.
- **Test set touched once.** The code, the primary-model rule and the test run were committed in
  that order; the test split is locked behind an explicit `final_evaluation=True`.
- **No invented numbers.** Results, this README and the model card are generated from run outputs.
- **Honest reporting.** The negative results are in the write-up: see *Limitations*.

## Stage 2: re-score the order when the seller hands it to the carrier

At approval, almost nothing about a fresh local problem (a carrier strike, a storm) can be seen:
outcome history only updates after parcels are *delivered*. But a few days later the seller hands the
parcel to the carrier, which is observable long before delivery. Stage 2 re-scores each order at
that moment with the approval-time features **plus**:

- the order's own handover timing (days to handover, handover versus the seller's shipping limit,
  promise days left),
- the seller's recent handover punctuality,
- a **destination backlog**: of earlier orders to the same state handed over 8-28 days ago, the share
  *still undelivered today*, which rises as a regional problem builds, days before outcomes arrive.

Same leakage discipline: the customer delivery date and reviews are never used; the backlog counts a
delivery only if it happened on a day before the decision day; events come from all orders, not only
ones that were eventually delivered. Tests include a brute-force equality check and a "future
deliveries must not change the feature" test.

| Test period | Stage 1 (at approval) | Stage 2 (at handover) |
|---|---|---|
| PR-AUC [95% CI] | {{s1_same_pr_auc}} | **{{s2_pr_auc}}** {{s2_pr_ci}} |
| ROC-AUC | {{s1_same_roc}} | {{s2_roc}} |
| Lift at 10% daily capacity | {{s1_same_lift10}}x | **{{s2_lift10}}x** {{s2_lift10_ci}} |
| Paired PR-AUC difference | | {{s2_pr_diff}} {{s2_pr_diff_ci}} |
| Paired lift@10% difference | | {{s2_lift_diff}} {{s2_lift_diff_ci}} |

Both columns are scored on the same orders and grouped by the *decision day* (handover day), so stage 1's lift here ({{s1_same_lift10}}x) differs slightly from the headline table above, which groups by approval day. The population is the {{s2_population}} orders with a valid handover timestamp.

{{s2_monthly_table}}

The approval-time model decays by {{s2_last_month}} (ROC-AUC {{s1_roc_last}}); the handover-time model
holds ({{s2_roc_first}} to {{s2_roc_last}}). What each block adds (validation period, logistic
regression):

{{s2_ablation_table}}

- The backlog says *when and where* trouble is, not which order: its median AUC inside a single
  state-month is only {{s2_within_auc}}, so it is not encoding an order's own outcome.
- Handover usually happens {{s2_handover_median_days}} days after approval, and {{s2_window_share3}} of
  late orders still had 3 or more days before the promised date when it happened, so there is time to act.
- Simulated policy: {{s2_policy_sentence}}.
- **Honest status.** The test months had already been used once to evaluate stage 1. Stage 2 was
  declared separately, selected on validation ({{s2_model}} by validation PR-AUC), scored once, and
  is reported next to stage 1 on the same orders, but it is *not* a first-look hold-out.
- **Probabilities.** Ranking is the product. A {{s2_cal_method}} calibrator, fitted on validation only,
  brings the test Brier score from {{s2_cal_brier_raw}} (raw) to {{s2_cal_brier}} and ECE from
  {{s2_cal_ece_raw}} to {{s2_cal_ece}}; stage 1 scores {{s2_cal_brier}} vs {{s2_cal_brier_s1}} Brier and
  {{s2_cal_ece_s1}} ECE, and always predicting the true late rate scores {{s2_const_brier}}. Base rates
  move between periods, so treat the percentages as relative scores.
- **Reasons.** Handover timing, seller handover history and destination backlog are explained in the same
  plain-English format as stage 1 (`reports/stage2_reason_codes.parquet`).

## Promise dates: quote shorter promises where it is safe (S1)

A LightGBM *quantile* model predicts how many calendar days an order will take to arrive, at levels from
0.5 to 0.99, using only approval-time features. The promise is the predicted quantile rounded up to whole
days; "on time" is judged exactly as the dataset does (delivery date vs promised date), which the code
checks reproduces `is_late` (it refuses to run above 1% disagreement; measured: {{eta_mismatch}}). Levels are chosen on validation; the test months
({{eta_n_test}} orders) are a third look and are labelled as such.

{{eta_table}}

- Matching Olist's validation on-time share ({{eta_val_cov}}) selected level {{eta_alpha}}. On the test
  months that promise is {{eta_len_diff}} days {{eta_len_ci}} vs Olist **and** {{eta_cov_diff}} pp
  {{eta_cov_ci}} more orders on time.
- Read off the model's frontier, Olist's own on-time share would need an average promise of about
  {{eta_interp_len}} days instead of {{eta_olist_len}} (descriptive, not tuned on test).
- {{eta_shift_note}} Very high levels buy little: the extreme tail is hard to predict and the promise grows
  quickly.
- Not modelled: how customers react to shorter or longer promises. See
  [`notebooks/06_promise_dates_and_api.ipynb`](notebooks/06_promise_dates_and_api.ipynb).

## Scoring API and Docker (S3)

```bash
uv sync --extra api && make api          # http://127.0.0.1:8000/docs
make docker-build && make docker-run     # same service in a container (needs `make all` first)
```

| Endpoint | What it does |
|---|---|
| `GET /health`, `GET /model-info` | liveness; model, training window, feature lists, reported metrics |
| `POST /score` | stage 1 at approval: probability, relative risk, reasons that add and reduce risk |
| `POST /score/handover` | stage 2: adds the five handover features |
| `POST /promise` | whole-day promises at several levels, plus the validated operating points |
| `GET /demo/orders/{id}` | score a known order by id (demo only) |

Payloads are validated (missing or non-numeric features return 422 with the list). The container returned
the same probability for the same order as the local run. The image is large (about 2 GB) because the
explainability stack (SHAP, InterpretML) is needed to load the saved models.

## Things I tried that did not work

Retraining monthly and dropping the calendar features were the two obvious fixes for the decay of the
approval-time model. Both were tested in a rolling-origin backtest ({{roll_months}}; a development
backtest, not a clean hold-out):

{{roll_table}}

- Retraining logistic regression every month on all outcomes known by then did **not** help
  (pooled lift {{roll_retrain_lift}} vs {{roll_static_lift}} for the static model; difference
  {{roll_lr_diff}} {{roll_lr_ci}}).
- Dropping the calendar proxies did not help either (lift difference {{roll_proxy_lr}} for logistic
  regression, {{roll_proxy_lgbm}} for LightGBM). My earlier suggestion to remove them is **retracted**.
- Short-window history (7 and 14 days) was also checked on validation-period data and did not carry
  more signal than the existing features in the months where the model failed.

The decay was not staleness: it was a sudden local shock that outcome-based history cannot see in time.
That is what motivated Stage 2.

## Monitoring

`make monitor` computes, per month, input drift (PSI) for order-intrinsic features against the previous
three months, ranking quality of the deployed model, and a status. System-state features
({{drift_state_features}}) move every month by design and are tracked by level, not alerted: using PSI
on them fired every single month, which made the first version of this monitor useless.

{{drift_table}}

{{drift_lead_sentence}} With only {{drift_n_months}} monitored months and a rule shaped after seeing that
raw PSI always fires, this is a worked design with an encouraging pattern, not a validated early-warning
system.

## What the model sees, and why

Every prediction comes with plain-English reasons. Example (true positive, order `{{example_order}}`,
model estimate {{example_prob}}, actually late):

{{example_reasons}}

Reasons are exact decompositions of the model output (coefficients for logistic regression, shape
functions for the EBM, TreeSHAP for LightGBM), grouped so one idea is shown once, and phrased as
facts: the *(adds risk / reduces risk)* tag comes from the model, never from intuition.
{{agreement_sentence}} See [`notebooks/04_explainability.ipynb`](notebooks/04_explainability.ipynb).

## The app

```
make app        # streamlit run app/Home.py   (needs the pipeline artifacts)
```

| | |
|---|---|
| ![Overview](docs/assets/app_overview.png) | ![Order risk explorer](docs/assets/app_risk_explorer.png) |
| ![What-if](docs/assets/app_what_if.png) | ![Ops queue and policy simulator](docs/assets/app_ops_queue.png) |
| ![Seller scorecard](docs/assets/app_seller_scorecard.png) | ![Model insights](docs/assets/app_model_insights.png) |
| ![Stage 2](docs/assets/app_stage2.png) | ![Drift monitor](docs/assets/app_drift_monitor.png) |

## Run it yourself

```bash
git clone https://github.com/TheInfiniteLoop22/OnTime.git && cd OnTime
make setup                         # uv sync + pre-commit  (Python 3.11, uv required)

# 1) data: either use the Kaggle API (put kaggle.json in ~/.kaggle) ...
make data
# ... or download the zip manually and unzip the 9 CSVs into data/raw/

make all                           # data -> build -> features -> impact -> baselines
                                   #   -> train -> evaluate -> explain -> report
make app                           # open the Streamlit app
make test && make lint             # pytest (incl. a synthetic mini-Olist pipeline run) and ruff
```

Individual stages: `make stage2` (handover model), `make eta` (promise dates), `make monitor` (drift tables), `make rolling` (retraining study), `make scores` (precompute app scores for a fast start).

### Deploy the app

`main` deliberately contains no data or models. `make bundle` builds a ~25 MB self-contained `deploy` branch (app, source, the generated artifacts the app reads, a runtime-only `requirements.txt`) without touching `main`. On [share.streamlit.io](https://share.streamlit.io): New app -> this repository -> branch `deploy` -> main file `app/Home.py` -> Python 3.11. The bundle holds derived, anonymised order-level data from a CC BY-NC-SA dataset: keep it non-commercial and attributed.

The test suite includes an integration test that generates a tiny synthetic Olist-shaped dataset and
runs the real stages on it, so CI never needs the Kaggle download.

## Repository map

| Path | What is there |
|---|---|
| `src/ontime/data` | download, load, pandera schemas, DuckDB order table, data-quality report |
| `src/ontime/features` | geo, calendar, product and leak-safe history features, leakage demo |
| `src/ontime/models` | splits (test locked), baselines, training, tuning, calibration |
| `src/ontime/evaluation` | metrics with bootstrap CIs, slices, policy simulation, plots, results report |
| `src/ontime/explain` | EBM / SHAP / coefficient explanations, agreement, reason codes |
| `src/ontime/serving` | scoring + what-if service used by the app |
| `src/ontime/stage2` | handover-time features (backlog, handover timing) and the stage-2 model |
| `src/ontime/eta` | quantile promise-date model (S1) |
| `src/ontime/api` | FastAPI scoring service (S3), see `Dockerfile` |
| `src/ontime/monitoring` | PSI drift and performance-by-month monitor |
| `src/ontime/deploy` | deploy-branch bundle builder |
| `app/` | Streamlit app (`Home.py`, `views/`, `ontime_ui/`) |
| `notebooks/` | 01 EDA, 02 impact analysis, 03 modelling walkthrough, 04 explainability, 05 diagnosing the decay and the handover stage, 06 promise dates and the API |
| `reports/` | generated results, model card, figures (the data and models are git-ignored) |
| `docs/` | specs, `PHASE_LOG.md` (what was done), `DECISIONS_LOG.md` (why), `LEARNING_LOG.md` (interview notes) |

## Limitations

- **Probabilities are relative scores.** The late rate moved from {{train_late_rate}} (training) to
  {{test_late_rate}} (test) and month to month; calibrated percentages are overconfident for the
  riskiest orders. Use rank and risk bands. Monthly re-calibration did not fix this.
- **The approval-time ranking decays.** Primary-model ROC-AUC was {{roc_first}} in {{month_first}} and
  {{roc_last}} in {{month_last}}; monthly retraining did not fix it. Stage 2 (re-scoring at handover) holds
  up, but it is a second look at the same months and needs fresh data to be confirmed.
- **Modest, not magic.** Most late orders are not predictable from approval-time information (a
  carrier incident, a stock-out). The value is a better *queue*, not certainty.
- **Simulation, not measurement.** The policy benefit assumes an intervention success rate; the
  dataset has no experiment on interventions.
- **Associations, not causes.** Explanations describe the model; the review-score gap is adjusted
  but observational.
- **Calendar proxies.** `approve_month` and the holiday features mostly encode the part of the year
  the training data covered; removing them did not improve out-of-time ranking in the retraining study.
- **One marketplace, 2016-2018.** Nothing here is validated on other platforms or periods.

## Future work

1. Confirm Stage 2 and the promise model on genuinely new data (the same months were seen two and three
   times).
2. A randomized pilot to measure the intervention success rate the policy simulator assumes, and the
   demand response to shorter promises.
3. Streamlit Community Cloud deployment from the `deploy` branch (see below), and scheduled retraining
   driven by the drift monitor.
4. Slim the API image (export the models to ONNX or drop SHAP at inference) if size matters.

## Data and licence

Olist Brazilian E-Commerce Public Dataset, CC BY-NC-SA 4.0: non-commercial use, attribution
required, share alike. The raw data is **not** committed; download it yourself. Code in this
repository is for portfolio/educational use.

## Documentation

[`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) · [`docs/FEATURE_SPEC.md`](docs/FEATURE_SPEC.md) ·
[`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) · [`docs/PHASE_LOG.md`](docs/PHASE_LOG.md) ·
[`docs/DECISIONS_LOG.md`](docs/DECISIONS_LOG.md) · [`docs/LEARNING_LOG.md`](docs/LEARNING_LOG.md) ·
[`reports/model_card.md`](reports/model_card.md) · [`reports/results.md`](reports/results.md) ·
[`reports/impact.json`](reports/impact.json)
