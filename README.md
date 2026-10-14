# OnTime — Explainable Late-Delivery Risk Engine

Predicts at order approval whether an Olist e-commerce order will miss its promised delivery date,
explains why in plain English, and turns scores into a capacity-aware operations policy.

> Work in progress — built phase by phase. See `docs/BUILD_PLAN.md`, `docs/PHASE_LOG.md`,
> `docs/DECISIONS_LOG.md`. The full README is written in Phase 10.

## Quick start
```
make setup     # uv sync + pre-commit
make test
make lint
```
Data: Olist Brazilian E-Commerce (Kaggle `olistbr/brazilian-ecommerce`, CC BY-NC-SA 4.0) — not
committed; download with `make data` (needs Kaggle credentials) or place the 9 CSVs in `data/raw/`.

## The app
```
make app        # streamlit run app/Home.py  (needs `make all` artifacts)
```
Five pages: Overview, Order risk explorer (reasons, what-if, model comparison), Ops queue & policy
simulator, Seller scorecard, Model insights.

| | |
|---|---|
| ![Overview](docs/assets/app_overview.png) | ![Risk explorer](docs/assets/app_risk_explorer.png) |
| ![What-if](docs/assets/app_what_if.png) | ![Ops queue](docs/assets/app_ops_queue.png) |
