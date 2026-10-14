"""CLI entry point: `python -m ontime.pipeline <stage>`.

Stages are registered here as they are implemented in later phases.
"""

from __future__ import annotations

import argparse
import logging

from ontime.config import setup_logging

log = logging.getLogger("ontime.pipeline")

PENDING: dict[str, str] = {
    "build": "Phase 2",
    "features": "Phase 4",
    "train": "Phase 6",
    "evaluate": "Phase 7",
    "explain": "Phase 8",
    "impact": "Phase 3",
}


def stage_data() -> None:
    """Download (if needed), load, validate all raw tables and write the quality report."""
    import pandera.errors

    from ontime.data.download import download
    from ontime.data.load import load_all
    from ontime.data.quality import write_report
    from ontime.data.schemas import validate_table

    download()
    tables = load_all()
    validation: dict[str, str] = {}
    for name, df in tables.items():
        try:
            tables[name] = validate_table(name, df)
            validation[name] = "pass"
        except pandera.errors.SchemaErrors as exc:
            validation[name] = f"FAIL ({len(exc.failure_cases)} violations)"
            log.error("Schema validation failed for %s:\n%s", name, exc.failure_cases.head(20))
    write_report(tables, validation)
    if any(v != "pass" for v in validation.values()):
        raise SystemExit("Raw data validation failed; see log and reports/data_quality.md")


STAGES = {"data": stage_data}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ontime.pipeline")
    parser.add_argument("stage", choices=sorted([*STAGES, *PENDING]))
    args = parser.parse_args(argv)
    setup_logging()
    if args.stage in STAGES:
        STAGES[args.stage]()
    else:
        log.warning("Stage '%s' is not implemented yet (%s).", args.stage, PENDING[args.stage])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
