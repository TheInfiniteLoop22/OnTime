# OnTime — Explainable Late-Delivery Risk Engine for E-commerce

## 1. One-line pitch
Predict at order approval which e-commerce orders will miss their promised delivery date,
explain each prediction in plain language, and convert the predictions into a capacity-aware
operations policy — backed by a statistical analysis showing *why* late delivery matters
(it measurably drags down customer review scores).

## 2. The real problem
Online marketplaces promise customers a delivery date at checkout. When that promise breaks:
- customers leave poor reviews and churn,
- support tickets and refund requests rise,
- sellers and the marketplace lose ranking and trust.

Operations teams can intervene on a *limited number* of orders per day (expedite with the
carrier, nudge a slow seller, proactively message the customer with a revised date). The question
is not just "will this order be late?" but **"which N orders should we act on today, and what
should we tell the seller or customer about why?"**

That framing is what makes this a data science project rather than a Kaggle classifier:
1. **Quantify the business problem** with data (does lateness actually hurt reviews, and by how much?).
2. **Predict** risk with a leak-free, time-respecting model.
3. **Explain** each prediction (reason codes), and the model as a whole (global drivers).
4. **Decide**: turn scores into a policy under a capacity constraint and measure its value.

## 3. Why this dataset
**Olist Brazilian E-Commerce Public Dataset** (Kaggle `olistbr/brazilian-ecommerce`, CC BY-NC-SA 4.0).
~100k real, anonymized orders from 2016–2018 across 9 relational tables:

| Table | Key columns used |
|---|---|
| orders | order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date*, order_delivered_customer_date*, order_estimated_delivery_date |
| order_items | order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value |
| products | product_id, product_category_name, product_weight_g, product_length_cm, product_height_cm, product_width_cm, product_photos_qty (note: source has typo `lenght` columns) |
| sellers | seller_id, seller_zip_code_prefix, seller_state |
| customers | customer_id, customer_unique_id, customer_zip_code_prefix, customer_state |
| geolocation | zip_code_prefix → lat/lng (multiple rows per prefix, some outliers) |
| payments | order_id, payment_type, payment_installments, payment_value |
| reviews | order_id, review_score, review_creation_date (**analysis only, never a feature**) |
| category_translation | Portuguese → English category names |

\* used only to build the target / for analysis, never as features.

Why it is a good portfolio choice:
- **Messy, relational, real** — requires genuine SQL joins, deduplication, geocoding cleanup,
  handling multi-seller orders. This demonstrates core DS, not just `model.fit`.
- **A natural target** with no manual labeling: late = delivered after promised date.
- **Imbalanced** (minority class roughly high single digits % — verify in EDA, don't assume),
  which forces correct metric choice.
- **Temporal** — forces leak-safe feature engineering and time-based validation, the #1 thing
  interviewers probe.
- **Second outcome (reviews)** enables a separate statistical impact analysis.
- Widely known to interviewers, but almost always used for EDA or sentiment — rarely for a
  rigorous, leak-safe, explainable risk + policy system. That's the differentiation.

## 4. Problem definition
- **Unit of prediction:** one order.
- **Decision time (t₀):** `order_approved_at`. Only information known at t₀ may be used.
- **Population:** orders with `order_status == 'delivered'` and non-null delivered/estimated dates.
  Cancelled/unavailable orders are excluded from modeling (documented, counted).
- **Target:** `is_late = order_delivered_customer_date.date() > order_estimated_delivery_date.date()`.
  Compare dates (not timestamps) since the estimate is a date. Document this choice.
- **Censoring guard:** the dataset ends around late 2018; the last weeks contain mostly undelivered
  orders. Exclude orders approved after a cutoff set in config (determine from EDA where delivered
  share drops) so the test set is not biased toward fast deliveries.
- **Secondary (stretch) target:** `delivery_days` (approval → delivery) for quantile ETA models.

## 5. Splits (time-based)
Defined in `configs/config.yaml`, finalized after EDA looks at monthly volume. Starting proposal:
- Train: approvals up to 2018-01-31
- Validation: 2018-02-01 → 2018-04-30 (tuning, early stopping, calibration, threshold/capacity)
- Test: 2018-05-01 → censoring cutoff (touched once)

Also report **month-by-month performance on test** to show stability/drift — the Black Friday
2017 spike and early-2018 lateness surge make this interesting to discuss.

## 6. Methodology

### 6.1 Data engineering
- DuckDB SQL builds one row per order (`src/ontime/sql/build_order_table.sql`): aggregates items
  (count, total price, total freight, total weight/volume, n_distinct_sellers, max shipping_limit),
  picks the **primary seller** (highest item value) for seller-level features, joins customer and
  seller zip → cleaned geolocation.
- Geolocation cleanup: median lat/lng per zip prefix, drop points outside Brazil's bounding box,
  fall back to state centroid for missing prefixes (flag imputed rows).
- Pandera schemas validate raw and built tables (types, ranges, uniqueness, null budgets).

### 6.2 EDA (notebook 01 + figures)
Late rate over time, by customer state, by seller state, by route (seller_state→customer_state),
by category, by distance bucket, by weekday/hour; distribution of promised days vs actual days;
how "generous" Olist's estimate is (most orders arrive early — key insight); Black Friday effect.

### 6.3 Business impact analysis (notebook 02, `analysis/impact.py`)
Question: **How much does lateness reduce review scores?**
- Descriptive: review score distribution late vs on-time; share of 1-star reviews.
- Statistical test: Mann-Whitney U on review scores; bootstrap CI for difference in mean score
  and in % of 1–2 star reviews.
- Adjusted estimate: OLS (and ordered logit as robustness) of review_score on is_late +
  controls (price, freight, category, state, n_items, month). Report coefficient + CI.
- Dose-response: review score vs days late (binned), to show the effect grows with delay.
- Be honest in wording: this is an **adjusted association**, not a randomized causal effect.
  State the main threats (unobserved seller quality) in the write-up.
This gives the project a "why should the business care" number in the README.

### 6.4 Features
See `docs/FEATURE_SPEC.md`. Families: logistics geometry (distance, same-state, route),
promise/time windows (promised days, seller handling window), order composition (items,
weight, volume, freight ratio), product/category, calendar (weekday, hour, month, days to
Brazilian holiday, Black Friday window), payment, and **leak-safe historical** features
(seller past late rate with Bayesian shrinkage, route past late rate, seller recent load).

### 6.5 Models (all classical ML)
| Model | Role |
|---|---|
| Prior-rate baseline | predicts the training late rate (floor) |
| Rule baseline | e.g., "inter-state AND promised_days < X" — what an ops analyst might do by hand |
| Logistic regression (scaled, one-hot, L2) | transparent linear benchmark, coefficients as odds ratios |
| **EBM (Explainable Boosting Machine)** | glass-box model: accurate *and* intrinsically interpretable shape functions |
| **LightGBM** | strong black-box-ish benchmark, explained post-hoc with SHAP |
| Random forest (optional) | sanity comparison |

Class imbalance: compare `class_weight`/`scale_pos_weight` vs none — but since we calibrate and
pick thresholds by capacity, unweighted + calibrated often wins. Let the validation set decide.

Tuning: Optuna on validation PR-AUC (LightGBM, EBM key params, LR C). Modest budgets (50–100 trials).

Calibration: isotonic or Platt on validation; compare Brier score before/after. Calibrated
probabilities matter because the app shows "34% chance of late delivery" to humans.

**Headline question for the write-up: "How much accuracy do we give up by choosing the glass-box
EBM over LightGBM?"** If the gap is small, recommend EBM for production — a mature, defensible
conclusion interviewers love.

### 6.6 Evaluation
- Primary: **PR-AUC** (imbalanced). Also ROC-AUC, Brier score, log loss, calibration curve.
- Operational: **precision@k / recall@k** where k = daily ops capacity (e.g., top 5%, 10%, 20% of
  daily orders), lift curve, cumulative gains.
- Policy value: "if ops can act on top-k% risky orders per day and an intervention prevents
  lateness with probability p (assumption, config, sensitivity-tested over a range), how many late
  orders are prevented vs random selection vs the rule baseline?" Present as a curve over k and p.
  Clearly labeled as a simulation with stated assumptions.
- Slices: performance by customer region, distance bucket, category group, new vs established
  seller, month. Flag slices where the model is weak (e.g., new sellers with no history).
- Error analysis: inspect top false positives and false negatives with their reason codes.
- Bootstrap CIs on test PR-AUC and precision@k.

### 6.7 Explainability (the core differentiator)
1. **Global — EBM**: term importances and shape functions (e.g., risk vs distance curve,
   risk vs promised days) and pairwise interactions.
2. **Global — SHAP on LightGBM**: beeswarm summary, dependence plots for top features.
3. **Agreement check**: rank-correlation of global importances between EBM and SHAP, and
   whether shape functions agree in direction. Agreement = more trustworthy story; disagreement
   = discuss why.
4. **Local — reason codes**: for any order, top 3 risk-increasing and top 2 risk-decreasing
   factors translated into English templates, e.g.
   *"Seller ships far: 2,140 km from customer (adds risk)"*,
   *"Seller's recent on-time record is weak: 21% late in past orders (adds risk)"*,
   *"Generous promise: 29 days promised (reduces risk)"*.
   Templates live in a single mapping file so wording is reviewable.
5. **What-if**: in the app, adjust promised days or shipping mode proxy (e.g., freight) to see
   how risk changes — demonstrates the model's sensitivity and an actionable lever
   ("promise 4 more days on this route to cut risk from 40% to 15%").
6. **Model card** (`reports/model_card.md`): intended use, data, metrics, slices, limitations,
   ethical notes (sellers shouldn't be penalized on thin history — shrinkage addresses this).

### 6.8 Product: Streamlit app
Pages:
1. **Overview** — the business case: late rate trend, review-score impact headline with CI.
2. **Order Risk Explorer** — pick an order from the test set → calibrated risk, reason codes,
   SHAP waterfall, EBM local explanation side by side, actual outcome revealed.
3. **Ops Queue / Policy Simulator** — choose a date and capacity k → ranked queue of orders to
   act on with reasons; slider for intervention success rate → expected late orders prevented.
4. **Seller Scorecard** — seller's shrunk late rate over time, volume, typical reasons.
5. **Model Insights** — model comparison table, calibration plot, PR curves, EBM shape plots,
   SHAP global, slice table, glass-box vs black-box verdict.

### 6.9 Stretch goals (only after core is done)
- Quantile LightGBM / quantile regression for delivery days → honest P50/P90 ETA vs Olist's
  promise; measure "promise coverage" (how often the actual date falls within P90).
- Monthly drift monitoring: PSI on key features + rolling PR-AUC.
- FastAPI `/score` endpoint + Dockerfile.

## 7. What NOT to do
- No deep learning, no NLP on review text (keeps it distinct and inside scope).
- No random splits, no SMOTE-before-split, no tuning on test.
- Don't chase accuracy; it's meaningless here (predicting "never late" is ~90%+ accurate).

## 8. Deliverables checklist
- [ ] Reproducible pipeline (`make all`) from raw download to reports
- [ ] `reports/results.md` + `results.json` generated by code
- [ ] 4 narrative notebooks importing from `src/`
- [ ] Streamlit app with 5 pages
- [ ] Model card
- [ ] Tests (incl. leakage tests) passing, ruff clean
- [ ] README with problem, impact number, approach diagram, results table, screenshots, how to run

## 9. Resume bullets (templates — fill ONLY with numbers your pipeline produced)
**OnTime — Explainable Late-Delivery Risk Engine | Python, SQL (DuckDB), scikit-learn, LightGBM, InterpretML (EBM), SHAP, Optuna, Streamlit** — link
- Built a leak-safe ML pipeline over Olist's ~100k-order, 9-table e-commerce dataset to predict
  late deliveries at order approval, engineering geospatial, promise-window, and time-aware
  seller-history features with Bayesian shrinkage; achieved PR-AUC of [X] vs [Y] for a rule baseline
  on a temporally held-out test set.
- Quantified business impact: late orders received [Δ] lower review scores ([CI]) after
  controlling for price, category, and region (OLS + bootstrap), motivating proactive intervention.
- Compared a glass-box Explainable Boosting Machine with SHAP-explained LightGBM ([gap] PR-AUC
  difference) and generated plain-English reason codes per order; a capacity-aware policy
  flagging the top [k]% of daily orders captured [R]% of late deliveries ([L]× lift over random).

## 10. Interview prep — questions you must be able to answer
1. Why PR-AUC and not accuracy/ROC-AUC as the primary metric?
2. How exactly did you prevent leakage in the seller late-rate feature? What happens if you don't?
3. Why a time-based split? What would a random split have done to your metrics?
4. What is calibration and why does it matter for this product?
5. What is an EBM, how is it different from a GAM and from gradient boosting?
6. What do SHAP values mean? What are their limitations (correlated features)?
7. How did you choose k / the threshold? Why not 0.5?
8. Is the review-score effect causal? What could confound it?
9. How does Bayesian shrinkage help sellers with few orders?
10. Where does your model fail (slices) and what would you do next?
