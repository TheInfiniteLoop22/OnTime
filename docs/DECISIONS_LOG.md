# DECISIONS_LOG — why each choice was made

Purpose: interview prep. Each entry = decision, alternatives considered, reasoning, trade-off.
New entries are appended at the end of every phase. Numbers cited here come from code output only.

| ID | Phase | Decision |
|----|-------|----------|
| D-001 | 0 | uv + Python 3.11 pinned |
| D-002 | 0 | `src/` layout, logic out of notebooks |
| D-003 | 0 | Single YAML config as source of truth |
| D-004 | 0 | Private GitHub repo, commits straight to `main` |
| D-005 | 0 | Raw data never committed |
| D-006 | 0 | Split dates / cutoff marked PROVISIONAL until EDA |

---

### D-001 — uv with Python pinned to 3.11 (Phase 0)
- **Decision:** `requires-python = ">=3.11,<3.12"`, dependencies managed with `uv`.
- **Alternatives:** pip + requirements.txt; the machine's default Python 3.14.
- **Why:** CLAUDE.md specifies 3.11. `interpret`, `shap`, `lightgbm` and `numba` often lag behind
  the newest Python, so 3.14 risks install failures. uv gives a lockfile (`uv.lock`) for
  reproducibility.
- **Trade-off:** requires uv installed; a lockfile adds a file to maintain.

### D-002 — `src/ontime` package; notebooks only narrate (Phase 0)
- **Why:** logic in importable modules is unit-testable and reusable by the pipeline, notebooks
  and Streamlit app. Notebooks with logic are untestable and drift.
- **Trade-off:** slightly more setup than a flat script folder.

### D-003 — One `configs/config.yaml` (Phase 0)
- **Why:** seed, split dates, shrinkage alpha, capacity grid are experiment parameters. Central
  config means changes are visible in git diffs and nothing is hardcoded (CLAUDE.md rule 7).
- **Trade-off:** one more indirection when reading code.

### D-004 — Private repo, direct commits to main (Phase 0)
- **Why:** owner's instruction; single contributor, regular small commits per phase give a
  readable history. CI runs lint + tests on every push.
- **Trade-off:** no PR review gate; mitigated by CI and phase-end owner review.

### D-005 — Raw data not committed (Phase 0)
- **Why:** Olist data is CC BY-NC-SA 4.0 (non-commercial, attribution) and large; reproducibility
  comes from the download script instead. `data/*` is gitignored with `.gitkeep` placeholders.

### D-006 — Split dates and censoring cutoff are provisional (Phase 0)
- **Why:** PROJECT_SPEC §5 gives a starting proposal only. Fixing dates before looking at monthly
  volume and delivered-share would be guessing. They are finalized in Phase 2/3 from plots and
  recorded here.

---

### D-007 — Zip prefixes loaded as strings (Phase 1)
- **Alternatives:** leave as int64 (pandas default).
- **Why:** they are identifiers; leading zeros (e.g. `01037`) would be lost as ints and joins
  between customers/sellers/geolocation could silently mismatch.

### D-008 — Pandera schemas are `strict=False` and nullable only where data truly is (Phase 1)
- **Why:** we validate the columns we depend on without breaking if Kaggle adds extra columns;
  `lazy=True` reports every violation in one pass. Nulls allowed only for timestamps that are
  legitimately missing for undelivered orders and for product attributes.
- **Trade-off:** extra columns are not policed.

### D-009 — Validation failure stops the pipeline but still writes the report (Phase 1)
- **Why:** the report is the evidence for what went wrong; failing hard prevents building on bad
  data.

### D-010 — Data-quality report is generated, with a marked section (Phase 1)
- **Why:** CLAUDE.md rule 5 (no invented numbers). The section is delimited by HTML comment
  markers so Phase 2 can append the row-count funnel to the same file without overwriting it.

### D-011 — `tabulate` added as a dependency (Phase 1)
- **Why:** needed for markdown tables from pandas. CLAUDE.md says to ask before heavy
  dependencies; this is a tiny pure-Python package. Flagged here for transparency.

---

### D-012 - Build the table for ALL orders in SQL, filter in Python (Phase 2)
- **Alternatives:** put the WHERE clauses inside the SQL.
- **Why:** each exclusion is counted and logged as a funnel (BUILD_PLAN requirement), and the
  analysis phase can reuse the unfiltered table (e.g. cancellations). SQL does joins and
  aggregations; Python expresses the business rules readably.

### D-013 - Target is date-vs-date, not timestamp-vs-timestamp (Phase 2)
- **Why:** the promise is a calendar date (estimated timestamps are all midnight). An order
  delivered at 23:00 on the promised day kept the promise; comparing timestamps would call it
  late. Unit-tested on exactly that case.

### D-014 - Primary seller = highest item value, ties by seller_id (Phase 2)
- **Alternatives:** first item's seller; seller with most items; model each seller separately.
- **Why:** value is the best simple proxy for "who matters for this shipment"; deterministic
  tie-break gives reproducible results. 1,241 multi-seller orders (~1.3%) are affected;
  `n_distinct_sellers` stays as a feature to capture the complexity.

### D-015 - Geolocation: bounding-box filter + median centroid + state fallback (Phase 2)
- **Alternatives:** mean centroid; geocoding API; drop orders with missing geography.
- **Why:** median resists leftover GPS noise; dropping rows would bias against rural/new zips;
  state centroid keeps the order and a flag lets the model/analysis see imputed rows. Counts:
  209 seller + 260 customer rows imputed.

### D-016 - Censoring cutoff 2018-08-20 from mean delivery time, not delivered share (Phase 2)
- **Why:** delivered share barely falls in this (apparently pre-filtered) dataset, so the
  spec's suggested diagnostic is not informative. Mean delivery time among delivered orders and
  weekly volume collapse in the last two weeks, a truncation signature: only fast orders
  appear. Keeping them would bias the test set toward on-time. Cost: 1,458 orders dropped.
- **Honest caveat:** the cutoff is judged by eye on a weekly plot; a week either way would
  not change conclusions.

### D-017 - Carrier date and review columns never enter the order table (Phase 2)
- **Why:** leakage by construction is safer than leakage by discipline. Reviews are joined
  separately in the Phase 3 analysis only.

### D-018 - Drop the 61 'delivered before approved' orders (Phase 2)
- **Alternatives:** keep and clip the duration to zero; swap timestamps.
- **Why:** implausible rows cannot be repaired without guessing. 61 orders is 0.06% of the data,
  so dropping costs nothing; the count is shown in the funnel.

### D-019 - Haversine, not road distance (Phase 2)
- **Why:** only zip-prefix centroids are known; straight-line distance is a standard and cheap
  proxy and correlates with road distance at national scale.

---

### D-020 - Keep the latest review per order (Phase 3)
- **Alternatives:** first review; average of reviews; drop orders with several reviews.
- **Why:** the last answer is the customer's settled opinion; dropping would discard data
  (~0.5% of orders). Deterministic tie-break by creation date then review_id.

### D-021 - Mann-Whitney + stratified bootstrap instead of a t-test (Phase 3)
- **Why:** scores are ordinal and skewed (mass at 5); rank test avoids normality assumptions and
  gives an interpretable "probability on-time is rated higher". Bootstrap CIs need no
  distributional assumption. Resampling is stratified by group so group sizes stay fixed.
- **Trade-off:** bootstrap treats orders as independent; seller clustering is handled in the OLS.

### D-022 - Final split dates: keep the proposal (Phase 3)
- **Decision:** train <= 2018-01-31, validation 2018-02-01..04-30, test 2018-05-01..cutoff.
- **Alternatives considered (with real counts):** validation through May (test late orders drop
  from 1,096 to 642); train through February (test again 642 late orders).
- **Why:** the test set keeps > 1,000 late orders so PR-AUC bootstrap intervals are usable.
- **Known cost:** late rate 5.6% / 12.6% / 4.5% across splits = distribution shift; calibration
  on validation will be biased upward for test. We report it, not hide it.

### D-023 - Controls and seller-clustered standard errors in OLS (Phase 3)
- **Controls:** log price, log freight, items, log distance, same-state, category group (top 12 +
  other), customer state, approval month - all known at order time and plausible confounders.
- **Why cluster by seller:** orders of one seller are correlated; unclustered SEs are too narrow.
- **Not controlled:** seller quality (handled by D-024) and anything unobserved.

### D-024 - Seller fixed effects as the main robustness check (Phase 3)
- **Why:** the strongest threat to the estimate is "slow sellers are also worse sellers".
  Within-seller demeaning compares a seller with itself. Result was almost unchanged
  (-1.98 vs -1.97), which supports lateness itself carrying the penalty.
- **Limit:** removes only constant seller traits, not time-varying problems.

### D-025 - Ordered logit on a 40,000-order seeded subsample, compact controls (Phase 3)
- **Why:** it is a robustness check on the ordinal nature of the score, not the headline. Many
  dummies make the optimizer slow with no change to the conclusion; the sample size is reported.

### D-026 - Report the review-timing split and quote the conservative number (Phase 3)
- **Finding:** 70.3% of late orders were answered before delivery (mean 1.65 stars).
- **Decision:** the README headline will quote both the full adjusted gap (~2.0 stars) and the
  after-delivery gap (~0.55), and interpret the difference as reaction to a broken promise.
- **Why:** quoting only the large number would overstate what a late *delivery experience*
  costs. The split also changes the product story: act before the promised date.

### D-027 - Notebooks: computed inferences, claims verified against output (Phase 3)
- **Why:** the owner explained that notebooks will be read by other people. Hand-typed numbers in
  prose go stale and violate rule 5. Inference cells are f-strings over computed values, and every
  qualitative claim was checked against executed output; wrong ones were rewritten.

### D-028 - Ruff per-file ignores for notebooks and analysis imports (Phase 3)
- **Why:** prose lines in notebooks exceed 100 chars by design, and modules that set the
  matplotlib backend must import pyplot afterwards (E402). Scoped ignores keep real code strict.

---

### D-029 - Outcome is "known" at delivery, not when the promise date passes (Phase 4)
- **Alternatives:** treat an order as known-late once its estimated date has passed undelivered
  (available earlier in real life).
- **Why:** waiting for delivery is simpler, strictly conservative and cannot leak. Cost: history
  reacts a little slower to a seller that has just started slipping.

### D-030 - Strict `<` on timestamps; self-exclusion falls out of the rule (Phase 4)
- **Why:** an order's own delivery time is >= its own approval time (we dropped the 61 cases
  where it was not), so `delivered(p) < t0(o)` can never include itself, even at identical
  seconds. No special-case code means no special-case bug.

### D-031 - One prior for every shrunk rate: global late rate known before t0 (Phase 4)
- **Alternatives:** per-feature priors (e.g. window-specific global rate); a fixed train-set rate.
- **Why:** one leak-safe prior is easy to explain: "a seller with no history is assumed average".
  A fixed train-set rate would leak future information into early train rows. For the first 256
  orders (before any delivery was observed) the prior is undefined and the feature is NaN; the
  model pipeline imputes with a missing-indicator (Phase 6).
- `alpha = 20` (config): a seller's own record counts as much as the prior after 20 orders.

### D-032 - `customer_is_capital_metro` not implemented (Phase 4)
- **Why:** optional in the spec and needs an external capital-prefix table (new data source);
  customer_region, state-level rates and distance cover most of the same signal. Revisit only
  if error analysis points at metro vs interior.

### D-033 - Seller load uses the all-orders universe, not the modelling population (Phase 4)
- **Why:** the modelling population contains only orders that were *later delivered*. Counting
  only those would let a future outcome decide whether an order appears in a seller's load - a
  subtle leak. Load needs no outcome, so it counts every approved order of that seller.

### D-034 - Vectorized `searchsorted` over a packed (key, time) code, no per-row loops (Phase 4)
- **Alternatives:** `merge_asof` per group; DuckDB window functions; a loop over orders.
- **Why:** one sort + one binary search per feature scales to millions of rows with ~30 lines
  and is directly comparable to a naive O(n^2) reference in a property test (5 seeds).

### D-035 - Categorical/state handling: regions as features, states only as meta (Phase 4)
- **Why:** FEATURE_SPEC lists regions, not states; 27 state levels are partly redundant with
  route/customer-state late rates and invite memorisation. States stay as meta columns for
  slicing and reason codes.

### D-036 - Keep `approve_month` but flag it (Phase 4)
- **Why:** in the spec. Risk: with one seasonal cycle it acts as a time proxy and may not
  transfer. Will be examined with EBM shape plots in Phase 8; candidate for removal if unstable.

### D-037 - Split labels use whole-day boundaries with a contiguity assertion (Phase 4)
- **Why:** `train_end`/`val_end` are dates meant inclusively; an order approved at 23:59 on the
  last day must be in that split. The code refuses configs with gaps or overlaps.

---

### D-038 - Test split is locked in code, not by convention (Phase 5)
- **Alternatives:** rely on discipline; keep test in a separate file.
- **Why:** CLAUDE.md rule 4. `select_split("test")` raises unless `final_evaluation=True`, so
  touching the test set is an explicit, reviewable argument. Tested.

### D-039 - Top-k is evaluated per DAY, ties broken by seeded jitter (Phase 5)
- **Why:** ops capacity is daily ("act on k% of today's orders"), not a global quota; a global
  top-k would concentrate all actions on bad days and overstate precision. Random tie-breaking
  makes constant-score baselines equivalent to random selection instead of favouring
  row order, and is reproducible via the seed.

### D-040 - Bootstrap by days for top-k metrics, by orders for the rest (Phase 5)
- **Why:** orders on the same day share carrier conditions and are not independent; resampling
  orders would give intervals that are too narrow for top-k metrics.

### D-041 - Rule baseline: two named families, threshold on train by F1 (Phase 5)
- **Alternatives:** only the spec's inter-state & promised_days rule; optimize for precision.
- **Why:** "what an analyst would write by hand" includes the obvious distance rule; picking
  between two fixed families by train F1 is a fair, simple contest. The rule gets calibrated
  two-level probabilities from train so Brier/log loss are defined.
- **Trade-off:** F1 on a 5.6% class gives a broad rule (flags >50% of validation orders); a
  precision-oriented rule would flag fewer but is just a different point on the same curve.

### D-042 - PR-AUC baseline is the base rate, reported next to every model (Phase 5)
- **Why:** PR-AUC has no fixed floor of 0.5; the floor is the positive rate (12.6% on
  validation, 4.5% on test). Quoting a PR-AUC without it is meaningless, and the two splits
  have different floors, so comparisons across splits must use lift or the difference to base.

---

### D-043 - Validation split by time into tune / calib_fit / calib_eval (Phase 6)
- **Alternatives:** tune and calibrate on all validation (plan warns it is optimistic);
  K-fold CV (forbidden by rule 3 unless expanding-window).
- **Why:** tuning and calibration must not share rows; evaluation of calibrators must come
  *after* their fit rows in time, mimicking deployment. The split dates are in config.
- **Cost:** small slices (calib_eval has only 185 late orders), so calibrator comparison is
  noisy; I report the table rather than hide it.

### D-044 - val_calib is the honest column; val_tune numbers are labelled optimistic (Phase 6)
- **Why:** hyperparameters and early stopping both use val_tune, so its scores are biased
  upward. Models are compared on val_calib, which tuning never sees.

### D-045 - Calibrator candidates include "none"; choice is forward-in-time by Brier (Phase 6)
- **Why:** if scores are already reasonable, calibration only adds noise (LR and RF kept
  "none"). Isotonic is flexible and can win by memorizing its fit rows; evaluating on later
  rows removes that advantage. Deployed calibrator is refit on all of val_calib.

### D-046 - Class weighting is a searched hyperparameter, not an assumption (Phase 6)
- **Why:** spec: "let the validation set decide". For LightGBM, `scale_pos_weight` options are
  none / sqrt / balanced. Because we calibrate, weighting only matters for ranking quality.
- **Result:** helped LightGBM on val_tune, hurt LR; evidence is weak (unequal trial counts).

### D-047 - EBM early stopping uses a TIME-ordered validation slice via `bags` (Phase 6)
- **Alternatives:** EBM's default internal random validation split; no early stopping.
- **Why:** rule 3. The latest 15% of train (by approval time) is the validation set in every
  outer bag; remaining rows are bootstrapped per bag, keeping bagging's variance reduction.

### D-048 - Final models are trained on train only, not refit on train + validation (Phase 6)
- **Alternatives:** refit on train + val_tune (more recent data).
- **Why:** clean, reproducible provenance and a hold-out that stays valid; also keeps Phase 7
  simple. **Cost:** the model never learns from the Feb-Apr 2018 regime. Revisit as future work.

### D-049 - Random forest uses fixed settings, no tuning (Phase 6)
- **Why:** it is a sanity comparison per the spec, not a contender; fixed
  `min_samples_leaf=50` avoids memorizing individual orders.

### D-050 - LR preprocessing choices (Phase 6)
- **Choices:** median impute + missing indicators, log1p on 9 skewed quantities, one-hot for
  region/category/payment and also for weekday/month (cyclical, not ordered), duplicates
  (`log_distance_km`) dropped, rare levels pooled below 1%.
- **Note:** the search range for C ends at 100 and the optimum (50) sits near it: a wider range
  might help slightly; weak regularization suggests the model is not overfit-prone here.

### D-051 - Did not re-tune after seeing val_calib (Phase 6)
- **Why:** each look-and-adjust cycle leaks the hold-out into the model. The right fix for
  regime-specific tuning (CV over several regimes) is presented to the owner as a decision
  rather than silently applied.

### D-052 - Determinism is checked by running training twice (Phase 6)
- **Why:** CLAUDE.md rule 7. Result: identical for LR/LightGBM/EBM; RF within ~1e-17 due to
  parallel float summation, which is acceptable and documented.

---

### D-053 - Primary model fixed by a pre-registered rule (Phase 7)
- **Rule:** best PR-AUC on val_calib (validation data never used for tuning); committed in
  code before the test run (commit 8c95fb6). Outcome: logistic regression.
- **Why:** choosing the headline model after seeing test is p-hacking. The rule is cheap,
  mechanical and auditable.

### D-054 - Evaluation code is debugged on validation via `--dry-run` (Phase 7)
- **Why:** the test set must be read once. A dry run exercises the identical code path on
  validation (written to a git-ignored folder), so nothing in the real run is a first try.

### D-055 - Report both deployed and rolling recalibration; report the surprise (Phase 7)
- **Why:** Phase 6 predicted over-prediction under a falling base rate and I proposed rolling
  recalibration as the production remedy. On test it did not help (Brier 0.0454 vs 0.0435).
  A trailing window lags month-to-month swings of 1.2%-7.5%. Conclusion: do not promise
  absolute probabilities; present risk bands and rank-based decisions.

### D-056 - Policy simulation: per-day quota, assumed p, analytic random, oracle bound (Phase 7)
- **Why:** capacity is daily; p is unknowable from this data so it is swept; the analytic
  random baseline avoids Monte-Carlo noise; the oracle shows how much headroom exists given
  capacity, so gains are read against a ceiling, not just against zero.
- **Label:** simulation, never a measured outcome; the CI covers sampling noise over days only,
  not uncertainty in p.

### D-057 - Slice rule: flag if ROC-AUC is > 0.05 below overall and >= 30 late orders (Phase 7)
- **Why:** per-slice AUCs on few positives are noise; a minimum count plus a margin keeps the
  table to differences worth discussing. Thresholds are conventions, stated in the report.

### D-058 - Review-star translation given as a range (Phase 7)
- **Why:** Phase 3 showed the late-vs-on-time gap depends on when the review was written
  (0.55 vs 1.97 stars). Quoting a single number would overstate certainty; both are labelled
  as associations.

### D-059 - Primary-model result is reported, not explained away (Phase 7)
- **Why:** the tuned nonlinear models did not beat plain logistic regression and the rule
  baseline collapsed on test. These are the findings; any story of "the better model needs
  more tuning" would require touching the test set again, which is not allowed.

---

### D-060 - Explain three models with exact additive decompositions in log-odds (Phase 8)
- **Alternatives:** SHAP for everything (model-agnostic KernelSHAP); LIME.
- **Why:** each family has an exact native decomposition (coefficients, EBM terms, TreeSHAP), so
  explanations are not approximations and the sum-to-logit identity can be unit-tested. A
  shared unit (log-odds) lets reason codes and agreement checks treat models uniformly.

### D-061 - EBM interactions split equally between their two features (Phase 8)
- **Why:** reason groups work at feature level. An equal split is simple, symmetric and keeps the
  additive identity. Cost: attribution inside an interaction is a convention, not a measurement.

### D-062 - Reason sentences state facts; the sign supplies the verdict (Phase 8)
- **Alternatives:** direction-aware adjectives ("tight promise", "weak record").
- **Why:** learned directions can contradict intuition (`global_late_rate_30d`), and an
  adjective that disagrees with its own number destroys trust. Cost: sentences are less
  vivid. The exception policy: reintroduce an adjective only for features whose direction is
  verified and stable in all three models.

### D-063 - Group related features into 16 reasons before ranking (Phase 8)
- **Why:** FEATURE_SPEC section 7. Summing e.g. distance_km, log_distance_km and
  promise_per_1000km prevents one idea appearing as three reasons and hides sign flips between
  collinear siblings. Test: every feature maps to exactly one group and one template.

### D-064 - Global explanations on validation; reason codes on test (Phase 8)
- **Why:** global importances should not be fitted to test behaviour; per-order codes are
  produced on test because that is the data the case studies and error analysis use. Nothing
  here changes a model or threshold.

### D-065 - Marginal direction reported next to conditional direction (Phase 8)
- **Why:** a feature whose model direction differs from its one-at-a-time direction is
  usually correcting for correlated features and must not be read as a stand-alone effect.
  This is the guard against over-interpreting coefficients and SHAP signs.

### D-066 - Calendar proxies are not removed now (Phase 8)
- **Why:** removing `approve_month`/holiday features changes the model after the test set was
  opened; a clean v2 needs a new validation cycle. Recorded as the first follow-up and shown
  openly in the case studies.
