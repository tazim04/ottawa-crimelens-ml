from __future__ import annotations

import argparse
import logging

from app.model.pipelines.triage.scoring import run_scoring_pipeline

logger = logging.getLogger(__name__)
DEFAULT_PREVIEW_ROWS = 5


def configure_logging() -> None:
    """Configure a simple console logger for the scoring CLI."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the daily scoring workflow."""
    parser = argparse.ArgumentParser(
        description="Run the daily crime anomaly scoring workflow."
    )
    parser.add_argument(
        "--target-date",
        help="Target scoring date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Rolling lookback window used when building daily features.",
    )
    parser.add_argument(
        "--min-history-days",
        type=int,
        help="Minimum historical days required for a row to be scored.",
    )
    parser.add_argument(
        "--persist-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist scored rows to Postgres. Use --no-persist-results to disable.",
    )
    parser.add_argument(
        "--results-table",
        default="crime_anomaly_scores",
        help="Destination table used when persisting scores.",
    )
    parser.add_argument(
        "--if-exists",
        default="delete_rows",
        choices=["fail", "replace", "append", "delete_rows"],
        help="Behavior to use if the destination table already exists.",
    )
    return parser


def main() -> int:
    """Parse CLI args, run the scoring pipeline, and print a short summary."""
    configure_logging()
    args = build_parser().parse_args()

    logger.info("Starting scoring with the following parameters")
    for arg, value in vars(args).items():
        logger.info("  %s: %s", arg, value)

    pipeline_kwargs = {
        "target_date": args.target_date,
        "lookback_days": args.lookback_days,
        "min_history_days": args.min_history_days,
        "persist_results": args.persist_results,
        "results_table": args.results_table,
        "if_exists": args.if_exists,
    }

    # Run the scoring pipeline and handle any exceptions that occur during execution
    try:
        scored_results = run_scoring_pipeline(**pipeline_kwargs)
    except Exception as e:
        logger.exception("Error occurred while running scoring pipeline: %s", e)
        return 1

    logger.info("Scored %s rows", len(scored_results))
    if scored_results.empty:
        logger.info("No scoring rows were produced for the requested date")
        return 0

    logger.info("Scored %s rows", len(scored_results))
    logger.debug(
        "Scored results preview:\n%s",
        scored_results.head(DEFAULT_PREVIEW_ROWS).to_string(index=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
