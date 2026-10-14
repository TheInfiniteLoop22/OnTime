"""CLI entry point: `python -m ontime.pipeline <stage>`.

Stages are registered here as they are implemented in later phases.
"""

from __future__ import annotations

import argparse
import logging

from ontime.config import setup_logging

log = logging.getLogger("ontime.pipeline")

PENDING: dict[str, str] = {}


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


def stage_evaluate(dry_run: bool = False) -> None:
    """Final evaluation. Opens the TEST set unless --dry-run (validation, written to dryrun/)."""
    from ontime.evaluation.run import run_evaluation

    run_evaluation(dry_run=dry_run)


def stage_explain() -> None:
    """Global explanations, agreement report and reason codes for every test order."""
    from ontime.explain.run import run_explain

    run_explain()


def stage_monitor() -> None:
    """Drift tables: PSI by feature and month, label drift, performance by month."""
    from ontime.monitoring.run import run_monitor

    run_monitor()


def stage_rolling() -> None:
    """Rolling-origin retraining backtest (v2 study): monthly retrain, calendar-proxy ablation."""
    from ontime.models.rolling import run_rolling

    run_rolling()


def stage_eta() -> None:
    from ontime.eta.quantile import run_eta

    run_eta()


def stage_stage2() -> None:
    """Stage-2 model: re-score each order at carrier handover (validation-selected, test once)."""
    from ontime.stage2.run import run_stage2

    run_stage2()
    from ontime.stage2.explain import run_stage2_explain

    run_stage2_explain()


def stage_scores() -> None:
    """Precompute test-period scores so the app starts without scoring (fast cold start)."""
    from ontime.serving.batch import write_test_scores

    write_test_scores()


def stage_bundle() -> None:
    """Build the deploy bundle and push it to the `deploy` branch (main stays data-free)."""
    from ontime.deploy.bundle import publish_deploy_branch

    log.info("bundle | %s", publish_deploy_branch(push=True))


def stage_report() -> None:
    """Generate README.md and reports/model_card.md from the result files."""
    from ontime.reporting.render import build_reports

    build_reports()


def stage_train() -> None:
    """Tune, fit, calibrate and evaluate all models on validation; save to models/."""
    from ontime.models.run import run_training

    run_training()


STAGES = {
    "scores": stage_scores,
    "bundle": stage_bundle,
    "stage2": stage_stage2,
    "eta": stage_eta,
    "rolling": stage_rolling,
    "monitor": stage_monitor,
    "report": stage_report,
    "explain": stage_explain,
    "evaluate": stage_evaluate,
    "train": stage_train,
    "baselines": stage_baselines,
    "data": stage_data,
    "build": stage_build,
    "features": stage_features,
    "impact": stage_impact,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ontime.pipeline")
    parser.add_argument("stage", choices=sorted([*STAGES, *PENDING]))
    parser.add_argument(
        "--dry-run", action="store_true", help="evaluate: use validation instead of test"
    )
    args = parser.parse_args(argv)
    setup_logging()
    if args.stage == "evaluate":
        stage_evaluate(dry_run=args.dry_run)
    elif args.stage in STAGES:
        STAGES[args.stage]()
    else:
        log.warning("Stage '%s' is not implemented yet (%s).", args.stage, PENDING[args.stage])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
