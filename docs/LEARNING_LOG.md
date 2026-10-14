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
