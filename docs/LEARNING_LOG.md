# LEARNING_LOG — explain it like an interview

Concept notes per phase, written to be said out loud.

---

## Phase 0 — Scaffold

**Why a config file and a seed?** Experiments must be reproducible: same data + same config +
same seed = same numbers. Parameters in YAML are visible in git history.

**Why `src/` package + thin notebooks?** Code in modules can be unit-tested and reused by the
pipeline, app and notebooks. A notebook is a narrative, not the source of truth.

**Why lint + tests in CI from day one?** Cheap to start, and leakage tests later become a
safety net that runs automatically on every push.

---

## Phase 1 — Data acquisition & validation

**What is data validation and why do it?** Models silently accept garbage. A schema is a
contract: types, unique keys, allowed values, ranges. If the source changes, the pipeline fails
at the door instead of producing wrong metrics.

**Why store zip codes as strings?** They are labels, not quantities; arithmetic on them is
meaningless and leading zeros carry information.

**Why check inconsistent timestamps (delivered before approved)?** Real data has entry errors.
Because our target and features are built from timestamps, such rows can create impossible
durations or leak; they are counted now and filtered deliberately in Phase 2.

**Why does the geolocation table have 1M rows for 19k prefixes?** It has many GPS points per
zip prefix (and duplicates). We reduce to one centroid per prefix (median is robust to outliers).

---

## Phase 2 - Order-level table

**Why do the joins in SQL?** One order has many items, payments and (maybe) sellers; the model
needs one row per order. GROUP BY aggregates (sum price, count items) and window functions
(ROW_NUMBER to pick the primary seller) are the natural tool. Whiteboard test: orders LEFT JOIN
item aggregates LEFT JOIN primary seller LEFT JOIN geo, always on order_id, always one row
per order (we assert uniqueness).

**What is right-censoring?** The data stops at a date; orders approved right before it only
appear if they were delivered quickly, so the last weeks look unrealistically fast. Using them
would make the test set look easier than reality.

**Why compare dates, not timestamps, for lateness?** The promise is a date. Kept-promise is
"delivered by end of that day".

**Why a funnel?** Anyone can ask "who is in your model?"; the funnel answers precisely and shows
that exclusions are tiny and deliberate (cancelled orders have no outcome to predict).

**Why median centroid per zip?** A zip prefix has many GPS points; a few are wrong. The median
ignores outliers, the mean does not.

---

## Phase 3 - EDA and impact analysis

**Why PR-AUC, not accuracy?** Only ~7% of orders are late, so "always on time" scores ~93%
accuracy and finds nothing. PR-AUC (area under precision vs recall) only rewards finding the
rare class and is not inflated by the huge on-time majority.

**Mann-Whitney vs t-test?** A t-test compares means and assumes roughly normal data; scores are
1-5 and piled at 5. Mann-Whitney compares ranks: "how often does a random on-time order outrank a
random late one?" Use it for ordinal or skewed outcomes.

**What is a bootstrap CI?** Resample the data with replacement many times, recompute the statistic
each time, and take the 2.5th/97.5th percentiles. It shows how much the answer would wobble with a
different sample, with no formula or normality assumption.

**Why do controls change (or not change) an estimate?** If a third factor drives both lateness and
low scores (confounder), the raw gap mixes both. Controlling for it removes that part. Here the
estimate barely moved, so the measured confounders explain little.

**What does a seller fixed effect do?** It compares each seller only with itself, so anything
constant about the seller (quality, product type) cancels. If the effect survives, it is not just
"bad sellers are late".

**Why not say "causal"?** No randomization: unobserved factors can still link lateness and
reviews, and only reviewers are observed. We say "adjusted association".

**The surprise finding:** 7 in 10 late orders were reviewed before the parcel arrived. The
penalty is mostly the reaction to a broken promise, so interventions must happen before the
promised date.

**Why a time-based split even before modelling?** The late rate moves from ~1% to ~19% across
months; a random split would put neighbours from the same bad week in train and test and inflate
every metric.

---

## Phase 4 - Feature engineering and leakage

**What is data leakage?** Using information at training or prediction time that would not be
available when the real decision is made. It makes validation look great and production fail.

**What is the exact rule for the seller late-rate?** A past order contributes only if its
delivery time is strictly before the current order's approval time. "Approved earlier" is not
enough - that order may still be in transit, and its eventual lateness is the future.

**What happens if you ignore it?** Measured in `reports/leakage_demo.md`: the leaky version
looks clearly better than the safe one on every split, yet cannot be reproduced in production.

**What is Bayesian shrinkage, numerically?** Seller A: 3 past orders, 2 late; the platform
average is 7%. Raw rate 67% is noise. With alpha = 20: (2 + 20*0.07) / (3 + 20) = 0.148 -
pulled most of the way back to average. Seller B: 400 orders, 100 late: (100 + 1.4)/420 = 0.241,
almost the raw 25%. Few orders -> trust the prior; many orders -> trust the data.

**How do you compute history without a loop over orders?** Sort the past outcomes by (seller,
time); for each query a binary search gives how many same-seller outcomes are before t0 and a
cumulative sum gives how many were late. Tested against an obviously-correct O(n^2) loop.

**Why a property test?** Hand-written cases cover only the cases you thought of. Random data
compared against a trivially correct reference finds the ones you did not.

**Why can a feature that is leak-free still drift?** Platform growth makes `seller_n_past`
larger in later periods; the model trained on small numbers sees large ones at test time. We
check per-split means for exactly this.

---

## Phase 5 - Metrics and baselines

**Why PR-AUC and what is its baseline?** PR-AUC summarizes precision vs recall across all
thresholds. A random scorer's PR-AUC equals the positive rate (12.6% on validation here), not
0.5 like ROC-AUC. So "PR-AUC = 0.17" means "about 1.3x better than random".

**Why not ROC-AUC as the headline?** With few positives, ROC-AUC is dominated by the huge
negative class (false-positive *rate* stays small even when precision is poor), so it looks
flattering. PR-AUC reacts directly to precision.

**What is precision@k / recall@k / lift?** If ops can touch k% of orders: precision@k = share of
those that are truly late; recall@k = share of all late orders that we reached; lift = precision@k
divided by the base rate ("how many times better than picking at random").

**Why a daily quota?** Capacity is per day. A single global top-k hides that bad days would
swallow the budget.

**What is a bootstrap CI for a ranking metric?** Resample (orders or days) with replacement,
recompute the metric, take percentiles. It answers "how much would this number move with a
different sample?".

**Why baselines at all?** They anchor the claim. "Our model has PR-AUC 0.30" means little;
"0.30 vs 0.17 for a one-line rule vs 0.13 for guessing the base rate" is a result.

**Why choose the rule threshold on train only?** Otherwise the baseline gets a free peek at
validation, and the model's improvement is understated.

**Why calibrated probabilities for the rule?** So Brier and log loss are defined for it; a raw
0/1 flag has no probability quality to measure.

---

## Phase 6 - Models, tuning, calibration

**Logistic regression vs gradient boosting?** LR fits one weight per (scaled) feature; its
coefficients exponentiate to odds ratios, so it is transparent but cannot bend. Gradient boosting
adds many shallow trees, each correcting the previous ones' errors, so it captures non-linear
effects and interactions at the cost of transparency.

**What is an EBM and how is it different from a GAM and from boosting?** A GAM is a sum of one
smooth function per feature. An EBM *is* a GAM, but its shape functions are learned by cyclic
gradient boosting with tiny trees, one feature at a time, and it can add a few pairwise
interaction terms. Result: boosting-level accuracy with GAM-level readability - you can plot
exactly how each feature moves the score.

**What does early stopping do?** Boosting keeps adding trees; after some point it fits noise.
Early stopping watches a held-out set and stops when its metric has not improved for N rounds,
so the number of trees is chosen by data rather than guessed. The held-out set here is the
latest slice of time, not a random sample.

**Why Optuna, and why not tune on the test set?** Optuna's TPE sampler proposes the next
hyperparameters from the results so far, finding good settings in far fewer trials than a grid.
Anything tuned on a set gets optimistic scores on it, so we tune on one slice, calibrate on a
later one and keep the test untouched.

**What is calibration, with an example?** Among orders the model scores 0.30, about 30 in 100
should be late. A class-weighted model may say 0.80 for those same orders - ranking is intact
but the number is a lie. A calibrator learns the map from score to real frequency. Sigmoid
(Platt) has 2 parameters and is stable; isotonic is a monotone step function and is flexible
but needs more data.

**Why did class weighting help LightGBM's ranking search yet destroy its probabilities?**
Weights make positives "count more", which changes the scores' scale (the model is trained for a
50/50 world). Ranking can be fine; probabilities are not, hence calibration after.

**Why do calibrated probabilities get worse when the world changes?** A calibrator memorizes
"scores of 0.3 mean 30% late" for the period it was fit on. If the base rate then falls, the same
scores now over-predict. Hence rolling recalibration in production, and trusting rank over
absolute risk when conditions shift.

**Why can a simple model beat a fancy one out of time?** Flexible models can learn patterns
specific to the period they were tuned on (like a Feb-Mar surge). Constrained models carry less
of that baggage. The only way to know is a later hold-out - which is what val_calib is.

---

## Phase 7 - Final evaluation

**Why was the test set opened only once?** Every look at a hold-out set that influences a
decision turns it into a second validation set. After many looks its scores are optimistic.
One look, with the code and the primary model fixed in advance, keeps the number honest.

**How do you read PR-AUC = 0.093 here?** The base rate is 4.5%, so random ranking gives 0.045;
0.093 is about twice as good, and the 95% CI [0.085, 0.105] excludes the base rate by a wide
margin. Absolute PR-AUC looks small only because late orders are rare.

**What is lift@10% = 2.2?** If ops acts on the 10% highest-risk orders of each day, the share
of those that are truly late is 2.2 times the overall late rate.

**How do you compare two models fairly?** Paired bootstrap: resample orders once and score both
models on the same resample; the interval for the DIFFERENCE is much tighter than comparing two
separate intervals. If it contains 0, there is no reliable difference.

**Why can a calibrator fail out of time, and why did monthly recalibration not rescue it?**
The calibrator maps scores to the late rate seen in its fitting window. If the monthly late
rate jumps unpredictably (1.2% to 7.5% here), any trailing window lags it. Rank is robust to
that; absolute probability is not.

**What is concept/regime drift in this project?** The relationship between features and
lateness changed over the summer: ROC-AUC fell from 0.78 to 0.57. The fix is monitoring
(PSI/rolling PR-AUC) and retraining, not a better static model.

**Why is a "perfect ranking" line in the policy plot?** Capacity caps how many late orders can
be reached. Comparing against that ceiling shows how much of the achievable benefit the model
captures (at 10% capacity and p = 30%: about 74 of a ceiling of 306).

**Why a simulation and not a result?** We do not know how often an intervention actually
works; p is an assumption. The CI shows sampling uncertainty over days, not uncertainty in p.

---

## Phase 8 - Explainability

**What do Shapley values mean, intuitively?** Imagine the features joining the prediction one at
a time in every possible order. A feature's Shapley value is its average extra push on the
prediction across all those orders. The values add up exactly from the average prediction to
this order's prediction.

**What is their main limitation?** With correlated features the credit is split somewhat
arbitrarily (route late rate vs customer-state late rate), so one feature's value can look small
because its sibling took the credit. Also: SHAP explains the model, not the world.

**How do you read an EBM shape plot?** Each panel is one feature's learned effect on the
log-odds as the value changes. Zero is the average order. Rising = more of it, more risk; the
band is bag-to-bag uncertainty. Because the model is a sum of such panels (plus a few pairs), the
plots are the whole model, not a summary of it.

**Why group features into reasons?** Three distance features would otherwise produce three
near-identical bullets, and correlated siblings can carry opposite signs that confuse a reader.
Summing inside a group gives one honest message.

**Why avoid words like "tight" in reason codes?** A model's learned direction for a feature can
differ from intuition. If the sentence says "elevated" next to a low number, users stop trusting
all of it. Facts plus a sign are always true.

**Linear model vs tree model: why do explanations disagree?** Correlated features share credit
differently in a linear model; trees can use a feature only where it matters. Disagreement is a
reminder not to read single coefficients, not proof one model is wrong.

**What can explanations NOT tell you?** Causes. They describe how the model uses information
available at approval time; unexplained delays (the false-negative case) are information the
model never had.


---

## Phase 9 - The app

**How does the app avoid recomputing everything on every click?** Streamlit reruns the script
top to bottom on each interaction. `st.cache_resource` keeps heavy objects (loaded models) alive
across reruns; `st.cache_data` memoizes functions that return data (scoring the test period,
reading JSON). The first visit pays the scoring cost, later interactions only run the cheap parts.

**Why show bands instead of probabilities?** Probabilities drift when the base rate moves and are
overconfident at the top end (Phase 7). Operations act on a fixed daily capacity, so rank within the
day is both more honest and more useful.

**What is the what-if panel really doing?** It edits one lever, recomputes the features that are
mathematically derived from it, and re-scores. That measures how the *model* responds, which is
useful for understanding; it does not prove that lengthening a real promise would change delivery.

**How did you test a UI?** Streamlit's headless AppTest runs each page and lets a test move
sliders and read metrics (functional check). Visual quality needs eyes: screenshots from a headless
browser on a fresh server, including deep links. The deep-link bug was only visible that way.

**What would you do for production?** Precompute scores in a batch job, serve them from a store,
add monitoring for drift and calibration, retrain on a schedule, and put the scoring service behind
an API with the same decomposition checks as unit tests.


---

# Interview prep - the ten questions, with answer outlines
Numbers are intentionally not repeated here: quote them from `reports/results.md` and
`reports/model_card.md`, which are generated from the run.

**1. Why PR-AUC and not accuracy or ROC-AUC?** Late orders are rare, so "never late" is highly
accurate and useless. ROC-AUC is flattered by the huge on-time majority. PR-AUC tracks precision and
recall on the rare class; its floor is the base rate, so always quote it next to that floor. (Phase 3, 5)

**2. How exactly did you prevent leakage in the seller late-rate?** A past order counts only if it
was *delivered* before the current order's approval time. Approval time alone is not enough: that order
may still be in transit and its lateness is future information. Five required tests plus a property test
against a brute-force loop, and a measured demonstration that the leaky versions look better. (Phase 4)

**3. Why a time-based split? What would random have done?** Lateness comes in episodes, so random
splits put neighbours from the same bad week in train and test and inflate every metric. Time splits mimic
deployment and also exposed the regime shift. (Phase 3, 5)

**4. What is calibration and why does it matter here?** Among orders scored 0.3, about 30% should be
late. The app shows a probability to humans, and class weighting breaks the score scale. We compared none,
sigmoid and isotonic on data *later* than the fit data, and found calibration is hostage to the base rate.
(Phase 6, 7)

**5. What is an EBM; how is it different from a GAM and from boosting?** A GAM is a sum of one smooth
function per feature. An EBM is a GAM whose functions are learned by cyclic boosting with tiny trees, with a
few pairwise terms: boosting-level accuracy, GAM-level readability. (Phase 6, 8)

**6. What do SHAP values mean and what are their limits?** Average marginal contribution of a feature
over all feature orderings; they add up from the average prediction to this one. With correlated
features credit is split arbitrarily, and SHAP explains the model, not the world. (Phase 8)

**7. How did you choose k / the threshold; why not 0.5?** Operations has a daily capacity, so the
decision is "top k% of each day", evaluated per day. A 0.5 threshold would select almost nothing at a
single-digit base rate. k comes from capacity, not from the data. (Phase 5, 7)

**8. Is the review-score effect causal?** No: adjusted association. Threats: unobserved seller or order
difficulty, selection into reviewing, time-varying carrier problems. Seller fixed effects and the
after-delivery sensitivity analysis reduce, not remove, them. The surprise: most late orders are reviewed
*before* arrival. (Phase 3)

**9. How does Bayesian shrinkage help sellers with few orders?** (late + alpha*prior)/(n + alpha):
a new seller is judged mostly on the platform average, an established one on their own record. It stops
a handful of deliveries from deciding a seller's fate. Worked example in Phase 4 notes.

**10. Where does it fail and what next?** Slices (some regions, later months), orders delayed for
reasons invisible at approval, and decay over time after training. Next: expanding-window CV, drop the
calendar proxies, drift monitoring, a randomized pilot for the intervention success rate. (Phase 7, 10)

## Phase 10 - Documentation and polish

**Why generate the README?** CLAUDE.md rule 5: no number may be typed by hand. A README whose
numbers come from the same files as the results cannot drift from them, and a missing value raises
instead of leaving a blank.

**Why a synthetic dataset in CI?** The real pipeline runs end to end on a tiny Olist-shaped dataset in
seconds, so every push exercises the SQL, schemas, features and leakage logic without the Kaggle download
or the licence concern of redistributing data.

**What is a model card?** A one-page, standard disclosure: what the model is for and not for, the data,
metrics by period and slice, calibration, limits and ethical considerations. It is the artefact a
reviewer or a new teammate reads first.


---

## Phase 11 - Monitoring, negative results, and a second decision point

**Why does PSI fail on some features, and what did you do?** PSI compares a feature's distribution to a
reference. Features that are time-indexed or system state (rolling late rates, tenure, month) move
every month by design and take a narrow range inside one month, so PSI explodes and the monitor cries
wolf. Alert on order-intrinsic features, track state features by level, confirm with performance.

**What is label delay and why does it matter for monitoring?** Outcomes arrive ~10+ days after the
decision. Performance monitoring is therefore late; input-drift signals (no labels needed) are the
early warning. Same reason outcome-based history cannot see a fresh local shock.

**What did you try that did not work and why report it?** Monthly retraining and removing calendar
proxies. Negative results rule out cheap fixes, show the cause is not model staleness, and stop others
repeating them. Reporting them is also what makes the positive result credible.

**How do you know a big improvement is not leakage?** Ask what information the feature uses and when it
becomes known; test that changing the future leaves the feature unchanged; compare to a naive
implementation; ablate; check that the new signal does what it was designed for (here: the backlog
discriminates *when/where* but not *which order*, within-group AUC 0.53); and be suspicious
of improvements bigger than your intuition.

**Why a second model at a later decision time?** The information needed was not available at approval.
A decision point should be chosen where the information is good enough and there is still time to act.
Here handover is ~1-2 days after approval with ~10 days left for late orders.

**What is a development backtest vs a hold-out?** A hold-out is data you have not looked at for any
design decision. Once you have used it, it is a development set. Say so, disclose it, and ask for new
data to confirm.

**What does `make bundle` buy you?** A reproducible, small, data-bearing branch for hosting without
polluting `main` with large or licensed files, and a deployment you can verify in a clean environment.
