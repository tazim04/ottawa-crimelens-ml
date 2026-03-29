from __future__ import annotations

import argparse
import logging

from app.model.model import DEFAULT_MODEL_VERSION
from app.model.pipelines.training import run_training_pipeline

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure a simple console logger for the training CLI."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for offline model training."""
    parser = argparse.ArgumentParser(
        description="Train the crime anomaly model and persist its artifact."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Training range start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        help="Training range end date in YYYY-MM-DD format. Defaults to today when omitted.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Rolling lookback window used when building training features.",
    )
    parser.add_argument(
        "--min-history-days",
        type=int,
        help="Minimum historical days required for a training row.",
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help="Model version label stored in the saved artifact.",
    )
    parser.add_argument(
        "--contamination",
        default="auto",
        help="IsolationForest contamination setting, for example 'auto' or 0.05.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees to train in the Isolation Forest.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for model training.",
    )
    return parser


def main() -> int:
    """Parse CLI args, train a model artifact, and print a short summary."""
    configure_logging()
    args = build_parser().parse_args()

    logger.info("Starting training with the following parameters")
    for arg, value in vars(args).items():
        logger.info("  %s: %s", arg, value)

    artifact, saved_path = run_training_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        min_history_days=args.min_history_days,
        model_version=args.model_version,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    logger.info("Trained model version: %s", artifact.model_version)
    logger.info("Training rows: %s", artifact.training_row_count)
    logger.info("Feature columns: %s", len(artifact.feature_columns))
    logger.info("Saved artifact: %s", saved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
