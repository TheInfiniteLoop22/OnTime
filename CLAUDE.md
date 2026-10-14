# CLAUDE.md — OnTime: Explainable Late-Delivery Risk Engine

This file is read by Claude Code at the start of every session. Follow it strictly.

## What this project is
An end-to-end, **classical ML + core data science** project that predicts, at the moment an
e-commerce order is approved, whether it will be delivered **after the date promised to the
customer**, explains *why* in plain language, and turns predictions into an
operations policy ("which orders should the ops team act on today, given limited capacity?").

Dataset: Olist Brazilian E-Commerce Public Dataset (Kaggle: `olistbr/brazilian-ecommerce`).

Full specification: `docs/PROJECT_SPEC.md`
Feature definitions: `docs/FEATURE_SPEC.md`
Phase-by-phase build plan with acceptance criteria: `docs/BUILD_PLAN.md`

Read the relevant doc before starting any phase. Work **one phase at a time**, and stop at the
end of each phase so the owner (a student who must be able to explain every line) can review.

## Hard rules (never break these)
1. **No deep learning.** No PyTorch, TensorFlow, Keras, transformers, or neural nets of any kind
   (including sklearn's MLPClassifier). Allowed models: logistic regression, EBM (InterpretML),
   gradient-boosted trees (LightGBM), random forest for comparison, simple rule baselines.
2. **No leakage.** A feature may only use information available at the order's
   `order_approved_at` timestamp. Never use `order_delivered_carrier_date`,
   `order_delivered_customer_date`, review data, or anything recorded after approval as a feature.
   Historical aggregates (e.g., seller past late rate) may only use orders whose **outcome was
   already known** (delivered) **before** the current order's approval time. See
   `docs/FEATURE_SPEC.md` §"Leak-safe history features". Every such feature needs a unit test.
3. **Time-based splits only.** Never use random train/test splits. Splits are defined in
   `configs/config.yaml`. Cross-validation, if used, must be expanding-window time-series CV.
4. **Test set is touched once.** All tuning, threshold selection, and calibration happen on the
   validation set. The test set is only evaluated in the final evaluation phase.
5. **No invented numbers.** Never write a metric, count, or claim into README, reports, or the
   app that was not produced by running code in this repo. Results files are generated, not typed.
6. **Logic lives in `src/`, not notebooks.** Notebooks only import from `src/ontime` and narrate.
7. **Determinism.** Every random process takes `seed` from config (default 42).

## Stack
- Python 3.11, dependency management with `uv` (fallback: pip + requirements.txt)
- Data: DuckDB (SQL joins over the raw CSVs → Parquet), pandas, numpy, pandera (data validation)
- ML: scikit-learn, LightGBM, interpret (EBM), optuna (tuning), statsmodels (impact analysis)
- Explainability: shap, InterpretML EBM global/local explanations, sklearn PDP/ICE
- Viz: matplotlib/seaborn for report figures, plotly for the app
- App: Streamlit (multi-page)
- Quality: pytest, ruff (lint + format), pre-commit
- Utilities: `holidays` (Brazil calendar), `kaggle` CLI for download, PyYAML

Do not add new heavy dependencies without asking.

## Repository layout
```
ontime/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── Makefile
├── configs/config.yaml
├── data/                 # gitignored: raw/, interim/, processed/
├── docs/                 # PROJECT_SPEC.md, FEATURE_SPEC.md, BUILD_PLAN.md
├── notebooks/            # 01_eda, 02_impact_analysis, 03_modeling_walkthrough, 04_explainability
├── src/ontime/
│   ├── config.py
│   ├── data/             # download.py, load.py, schemas.py (pandera), build_orders.py
│   ├── sql/              # build_order_table.sql and other .sql files
│   ├── features/         # geo.py, calendar.py, product.py, history.py, build.py
│   ├── models/           # splits.py, baselines.py, train.py, tune.py, calibrate.py
│   ├── evaluation/       # metrics.py, policy.py, slices.py, plots.py
│   ├── explain/          # shap_explain.py, ebm_explain.py, reason_codes.py, agreement.py
│   ├── analysis/         # impact.py (late delivery → review score)
│   └── pipeline.py       # CLI entry: python -m ontime.pipeline <stage>
├── app/                  # Streamlit: Home.py + views/ (not pages/: see DECISIONS_LOG D-069)
├── models/               # gitignored saved models
├── reports/              # figures/, results.json, results.md, model_card.md
└── tests/
```

## Commands (keep the Makefile in sync)
```
make setup        # uv sync, pre-commit install
make data         # download Olist via Kaggle API into data/raw
make build        # DuckDB SQL → data/interim/orders.parquet, validated
make features     # data/processed/features.parquet
make train        # baselines + LR + EBM + LightGBM, tuned on validation, calibrated
make evaluate     # final test-set evaluation → reports/results.json + figures
make explain      # SHAP + EBM artifacts, reason-code tables
make impact       # review-score impact analysis
make app          # streamlit run app/Home.py
make test         # pytest
make lint         # ruff check + ruff format --check
make all          # data → build → features → train → evaluate → explain → impact
```

## Coding conventions
- Type hints on all public functions; short docstrings that say *why*, not just *what*.
- Pure functions for feature logic so they are unit-testable on tiny hand-made DataFrames.
- Paths and parameters come from `configs/config.yaml` via `ontime.config`, never hardcoded.
- Log with `logging`, not print. Each pipeline stage logs row counts in and out.
- Figures: saved to `reports/figures/` as PNG at dpi=150, consistent style, titles and axis labels
  in plain English, every figure referenced in `reports/results.md`.
- Keep functions small. If a file passes ~300 lines, split it.

## How to work with the owner
- At the start of each phase: restate the phase goal and acceptance criteria from BUILD_PLAN.md.
- At the end of each phase: run tests + lint, summarize what was built, list any decisions made,
  and add a short "Explain it like an interview" note to `docs/LEARNING_LOG.md` covering the
  concepts used in that phase (e.g., why PR-AUC instead of accuracy, what calibration means).
- If a design decision is ambiguous, prefer the simpler, more explainable option and note it.
