from __future__ import annotations

import argparse

from app.scoring import run_scoring_pipeline


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
        "--model-artifact-path",
        help="Path to the trained model artifact. Falls back to MODEL_ARTIFACT_PATH when omitted.",
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
        default="append",
        choices=["fail", "replace", "append", "delete_rows"],
        help="Behavior to use if the destination table already exists.",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        help="Percentile threshold for assigning a high triage label.",
    )
    parser.add_argument(
        "--medium-percentile",
        type=float,
        help="Percentile threshold for assigning a medium triage label.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of scored rows to print as a preview.",
    )
    return parser


def main() -> int:
    """Parse CLI args, run the scoring pipeline, and print a short summary."""
    args = build_parser().parse_args()

    scored_results = run_scoring_pipeline(
        target_date=args.target_date,
        model_artifact_path=args.model_artifact_path,
        lookback_days=args.lookback_days,
        min_history_days=args.min_history_days,
        persist_results=args.persist_results,
        results_table=args.results_table,
        if_exists=args.if_exists,
        high_percentile=args.high_percentile,
        medium_percentile=args.medium_percentile,
    )

    print(f"Scored {len(scored_results)} rows.")
    if scored_results.empty:
        print("No scoring rows were produced for the requested date.")
        return 0

    preview_rows = max(args.preview_rows, 0)
    if preview_rows:
        print(scored_results.head(preview_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
