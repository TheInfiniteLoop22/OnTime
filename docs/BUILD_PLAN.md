# BUILD_PLAN.md — phase-by-phase plan for Claude Code

Work through phases in order. Each phase ends with: tests pass, ruff clean, a short summary to
the owner, and an entry in `docs/LEARNING_LOG.md`. Do not start the next phase until the owner
says so. Suggested prompt to Claude Code for each phase:

> "Read CLAUDE.md and docs/BUILD_PLAN.md. Execute Phase N only. Restate its acceptance criteria
> first, then build, then verify each criterion and report."

---

## Phase 0 — Scaffold
**Build:** repo layout from CLAUDE.md, `pyproject.toml` (uv), Makefile targets (stubs ok),
`configs/config.yaml` (paths, seed, split dates, shrinkage α, capacity grid, intervention success
grid, Black Friday window, category grouping), `ontime/config.py` loader, logging setup,
`.gitignore` (data/, models/, .env, kaggle.json), pre-commit with ruff, empty test that passes.
**Accept when:** `make setup && make test && make lint` succeed on a clean clone.

## Phase 1 — Data acquisition & validation
**Build:** `data/download.py` (Kaggle API; clear error message if credentials missing, with
instructions to place `kaggle.json`), `data/load.py`, `data/schemas.py` (pandera schemas for all
9 raw tables), a data-quality report written to `reports/data_quality.md` (row counts, null %,
duplicate keys, out-of-range values, geolocation outliers).
**Accept when:** `make data` downloads 9 CSVs; validation runs; data-quality report exists and
its numbers come from code.

## Phase 2 — Order-level table (SQL)
**Build:** `sql/build_order_table.sql` + `data/build_orders.py` using DuckDB:
cleaned geolocation centroids (median per prefix, Brazil bounding box filter, state-centroid
fallback + flag), item aggregation, primary seller selection, payment aggregation, target
`is_late`, `delivery_days`, population filters with logged counts at each step, censoring cutoff
determined and justified (plot delivered-share by approval week). Output
`data/interim/orders.parquet` + pandera schema check (one row per order_id).
**Accept when:** row-count funnel is logged and written to `reports/data_quality.md`; tests cover
haversine, primary-seller selection, and target definition on tiny fixtures.

## Phase 3 — EDA + business impact
**Build:** `notebooks/01_eda.ipynb` (questions listed in PROJECT_SPEC §6.2) and
`analysis/impact.py` + `notebooks/02_impact_analysis.ipynb` (PROJECT_SPEC §6.3: Mann-Whitney,
bootstrap CIs, OLS + ordered logit with controls, dose-response by days late). Save figures to
`reports/figures/`. Write `reports/impact.json`.
Finalize split dates in config based on monthly volume/late-rate plots; justify in LEARNING_LOG.
**Accept when:** notebooks run top-to-bottom (`jupyter nbconvert --execute`), impact numbers are
in impact.json with CIs, wording says "adjusted association", not "causal effect".

## Phase 4 — Feature engineering
**Build:** every feature in `docs/FEATURE_SPEC.md`, modules per family, `features/build.py`
producing `data/processed/features.parquet` with `order_id`, `approved_at`, `split`, target,
features. Leakage tests from FEATURE_SPEC §5 (all 5). A test asserting no forbidden column
appears in the feature list. A feature summary table (dtype, null %, train mean, val mean,
test mean) to `reports/feature_summary.md` — eyeball for drift/leakage red flags (e.g., a
feature with near-perfect separation).
**Accept when:** `make features` works; all leakage tests pass; feature summary exists.

## Phase 5 — Baselines & splits
**Build:** `models/splits.py` (reads config, asserts no time overlap), `models/baselines.py`
(prior-rate and rule baseline; rule thresholds chosen on train only), `evaluation/metrics.py`
(PR-AUC, ROC-AUC, Brier, log loss, precision@k / recall@k where k is a fraction of each day's
orders, lift), bootstrap CI helper.
**Accept when:** baseline validation metrics written to `reports/baselines.json`; metric functions
unit-tested against sklearn on toy inputs.

## Phase 6 — Models, tuning, calibration
**Build:** `models/train.py` for LR pipeline, EBM, LightGBM (and optional RF);
`models/tune.py` with Optuna on validation PR-AUC (LightGBM: num_leaves, learning_rate,
min_child_samples, feature_fraction, bagging, lambda_l1/l2, n_estimators with early stopping on
val; EBM: max_bins, interactions, learning_rate, outer_bags; LR: C). Experiment with/without class
weighting. `models/calibrate.py`: isotonic vs sigmoid, chosen by validation Brier.
Note the subtlety: calibrating on the same validation set used for early stopping/tuning is
slightly optimistic — split validation into two halves by time (tune on first, calibrate on
second) and document it.
Save models + metadata (params, features, train date range, git hash) to `models/`.
**Accept when:** `reports/validation_results.json` contains all models; model comparison table
printed; saved models reload and reproduce validation scores.

## Phase 7 — Final evaluation (test set, once)
**Build:** `evaluation/` → test metrics with bootstrap CIs for all models; PR curves, calibration
plots, lift/gains curves; precision@k/recall@k at capacity grid; monthly test performance;
slice table (`evaluation/slices.py`); policy simulation (`evaluation/policy.py`) over capacity ×
intervention-success grid vs random and rule baseline; top-20 false positives and false
negatives exported for error analysis. Write `reports/results.json` and generate
`reports/results.md` from it via a template (no hand-typed numbers).
**Accept when:** `make evaluate` regenerates everything deterministically; results.md references
every figure; the glass-box vs black-box gap is stated with CI.

## Phase 8 — Explainability
**Build:** `explain/ebm_explain.py` (global term importances, shape-function plots for top
features, top interactions); `explain/shap_explain.py` (TreeExplainer on LightGBM: beeswarm,
bar, dependence plots, waterfall for sample orders); `explain/agreement.py` (Spearman of global
rankings, direction agreement table); `explain/reason_codes.py` + `configs/reason_templates.yaml`
(grouping + templates per FEATURE_SPEC §7); export reason codes for every test order to
`reports/test_reason_codes.parquet`. `notebooks/04_explainability.ipynb` walks through 3
case studies (a true positive, a false positive, a false negative) in plain English.
**Accept when:** template coverage test passes; reason codes for a sample of orders read
naturally (owner reviews 10 of them); agreement report exists.

## Phase 9 — Streamlit app
**Build:** `app/Home.py` + pages per PROJECT_SPEC §6.8: Overview, Order Risk Explorer, Ops Queue
& Policy Simulator, Seller Scorecard, Model Insights. Loads precomputed artifacts (fast);
scores single orders live with the saved calibrated model. What-if panel: edit promised_days and
freight and see risk update (recompute only derived features that depend on them). Cache with
`st.cache_data`/`st.cache_resource`. Clean, consistent styling; every chart has a one-line
caption explaining what to look at.
**Accept when:** `make app` runs with no errors from a clean `make all`; all pages load in < 3s
after cache warm-up; no number shown in the app is hardcoded.

## Phase 10 — Documentation & polish
**Build:** README (problem → impact number → approach diagram (mermaid) → results table →
explainability examples → app screenshots → how to run → limitations → future work),
`reports/model_card.md`, `notebooks/03_modeling_walkthrough.ipynb`, finalize LEARNING_LOG,
GitHub Actions CI running ruff + pytest on a small fixture dataset (not the full Kaggle data).
Optionally deploy the app on Streamlit Community Cloud using a small precomputed artifact bundle
(check the dataset licence: CC BY-NC-SA — non-commercial, attribute Olist, don't commit raw data).
**Accept when:** a stranger could clone, add Kaggle creds, run `make all && make app`, and
understand the project from the README alone.

## Stretch phases (optional)
- **S1 Quantile ETA:** LightGBM quantile objective (α = 0.5, 0.9) for delivery_days; compare
  promise coverage and average promise length vs Olist's estimate ("could we promise shorter
  dates on safe routes and longer on risky ones?").
- **S2 Drift monitoring:** PSI per feature by month + rolling PR-AUC chart, as an app page.
- **S3 Serving:** FastAPI `/score` returning probability + reason codes; Dockerfile.

---

## Owner study checklist (do this alongside each phase)
- Phase 2: be able to write the order-level SQL join on a whiteboard.
- Phase 3: explain Mann-Whitney vs t-test, bootstrap CIs, and why controls change the estimate.
- Phase 4: explain the delivered-before-t₀ rule and Bayesian shrinkage with a numeric example.
- Phase 6: explain gradient boosting, EBM vs GAM, early stopping, calibration.
- Phase 7: explain PR-AUC, precision@k, lift, and why the threshold comes from capacity.
- Phase 8: explain Shapley values intuitively and one limitation; explain an EBM shape plot.
