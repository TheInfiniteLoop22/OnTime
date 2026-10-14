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
