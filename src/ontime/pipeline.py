"""CLI entry point: `python -m ontime.pipeline <stage>`.

Stages are registered here as they are implemented in later phases.
"""

from __future__ import annotations

import argparse
import logging

from ontime.config import setup_logging

log = logging.getLogger("ontime.pipeline")

PENDING: dict[str, str] = {
    "train": "Phase 6",
    "evaluate": "Phase 7",
    "explain": "Phase 8",
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


def stage_build() -> None:
    """Build data/interim/orders.parquet and the population funnel report."""
    from ontime.data.build_orders import build

    build()


def stage_impact() -> None:
    """Run the review-score impact analysis; writes reports/impact.json and figures."""
    from ontime.analysis.impact import run_impact

    run_impact()


def stage_features() -> None:
    """Build data/processed/features.parquet and reports/feature_summary.md."""
    from ontime.features.build import build

    build()


def stage_baselines() -> None:
    """Fit prior-rate and rule baselines on train, score validation, write baselines.json."""
    from ontime.models.baselines import run_baselines

    run_baselines()


STAGES = {
    "baselines": stage_baselines,
    "data": stage_data,
    "build": stage_build,
    "features": stage_features,
    "impact": stage_impact,
}


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
