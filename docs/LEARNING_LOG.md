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
