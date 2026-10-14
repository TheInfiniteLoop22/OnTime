"""CLI entry point: `python -m ontime.pipeline <stage>`.

Stages are registered here as they are implemented in later phases.
"""

from __future__ import annotations

import argparse
import logging

from ontime.config import setup_logging

log = logging.getLogger("ontime.pipeline")

STAGES: dict[str, str] = {
    "data": "Phase 1",
    "build": "Phase 2",
    "features": "Phase 4",
    "train": "Phase 6",
    "evaluate": "Phase 7",
    "explain": "Phase 8",
    "impact": "Phase 3",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ontime.pipeline")
    parser.add_argument("stage", choices=sorted(STAGES))
    args = parser.parse_args(argv)
    setup_logging()
    log.warning("Stage '%s' is not implemented yet (%s).", args.stage, STAGES[args.stage])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
