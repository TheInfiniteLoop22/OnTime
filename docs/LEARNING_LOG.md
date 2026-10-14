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
