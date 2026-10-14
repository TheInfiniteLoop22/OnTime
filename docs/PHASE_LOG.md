# PHASE_LOG — what was done at each phase

One entry per phase: goal, what was built, files, how to run/verify, results, open items.

---

## Phase 0 — Scaffold (2026-10-14)

**Goal:** repo layout, tooling, config, Makefile, CI so every later phase has a clean base.

**Acceptance (BUILD_PLAN):** `make setup && make test && make lint` succeed on a clean clone.

**Built**
- Folder layout from CLAUDE.md; spec docs moved into `docs/`.
- `pyproject.toml` (uv, Python 3.11, all stack deps, ruff/pytest config).
- `configs/config.yaml` (paths, seed, splits, shrinkage alpha, capacity and intervention grids,
  Black Friday dates, geo bounding box).
- `src/ontime/config.py` (config loader, logging setup); `src/ontime/pipeline.py` (stage CLI stub).
- `Makefile` with all targets; `.gitignore`; `.pre-commit-config.yaml` (ruff);
  `.github/workflows/ci.yml` (ruff + pytest).
- `tests/test_config.py`.
- Docs: `PHASE_LOG.md`, `DECISIONS_LOG.md`, `LEARNING_LOG.md`.

**Verification:** see results appended below after running the commands.

**Verification results (run 2026-10-14):** `pytest`: 3 passed; `ruff check`: all checks passed; `ruff format --check`: all files formatted; `uv sync` on Python 3.11 OK; `make` installed via winget (ezwinports.make).

