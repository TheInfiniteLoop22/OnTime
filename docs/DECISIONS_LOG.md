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
