from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.features.constants import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_HISTORY_DAYS,
)
from app.model.pipelines.triage.labelling import (
    DEFAULT_TRIAGE_HIGH_PERCENTILE,
    DEFAULT_TRIAGE_MEDIUM_PERCENTILE,
)
from experiments.common import build_evaluation_frame, ensure_output_dir, save_json


DEFAULT_OUTPUT_DIR = Path("experiments/output/alignment")

###### Measure the alignment between anomaly scores and baseline feature deviations. This helps verify that the model is learning meaningful patterns. ######
# example: high z-score -> high anomaly score -> high triage label, while low z-score -> low anomaly score -> low triage label

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the alignment experiment."""
    parser = argparse.ArgumentParser(
        description="Measure and plot alignment between anomaly scores and baseline deviations."
    )
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--model-artifact-path",
        default="artifacts/crime_model.joblib",
        help="Path to the trained model artifact.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Rolling window used to build evaluation features.",
    )
    parser.add_argument(
        "--min-history-days",
        type=int,
        default=DEFAULT_MIN_HISTORY_DAYS,
        help="Minimum history days required per row.",
    )
    parser.add_argument(
        "--medium-percentile",
        type=float,
        default=DEFAULT_TRIAGE_MEDIUM_PERCENTILE,
        help="Percentile threshold used for medium triage.",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=DEFAULT_TRIAGE_HIGH_PERCENTILE,
        help="Percentile threshold used for high triage.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write plots and summary tables into.",
    )
    return parser


def correlation_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Compute simple alignment statistics for the evaluation frame."""
    return {
        "pearson_score_vs_abs_count_delta": float(
            frame["anomaly_score"].corr(frame["abs_count_delta_from_mean"], method="pearson")
        ),
        "spearman_score_vs_abs_count_delta": float(
            frame["anomaly_score"].corr(frame["abs_count_delta_from_mean"], method="spearman")
        ),
        "pearson_score_vs_abs_count_zscore": float(
            frame["anomaly_score"].corr(frame["abs_count_zscore"], method="pearson")
        ),
        "spearman_score_vs_abs_count_zscore": float(
            frame["anomaly_score"].corr(frame["abs_count_zscore"], method="spearman")
        ),
    }


def triage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate supporting statistics by triage label."""
    summary = (
        frame.groupby("triage_label", observed=False)
        .agg(
            row_count=("triage_label", "size"),
            mean_anomaly_score=("anomaly_score", "mean"),
            median_anomaly_score=("anomaly_score", "median"),
            mean_abs_count_delta=("abs_count_delta_from_mean", "mean"),
            median_abs_count_delta=("abs_count_delta_from_mean", "median"),
            mean_abs_count_zscore=("abs_count_zscore", "mean"),
            median_abs_count_zscore=("abs_count_zscore", "median"),
        )
        .reindex(["low", "medium", "high"])
        .reset_index()
    )
    return summary


def plot_scatter(
    frame: pd.DataFrame,
    *,
    x_column: str,
    x_label: str,
    output_path: Path,
) -> None:
    """Save a scatter plot showing alignment between a feature and anomaly score."""
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=frame,
        x=x_column,
        y="anomaly_score",
        hue="triage_label",
        hue_order=["low", "medium", "high"],
        alpha=0.6,
        s=30,
    )
    plt.xlabel(x_label)
    plt.ylabel("Anomaly score")
    plt.title(f"Anomaly Score vs {x_label}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_boxplot(frame: pd.DataFrame, *, output_path: Path) -> None:
    """Save a boxplot that compares supporting z-scores across triage labels."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=frame,
        x="triage_label",
        y="abs_count_zscore",
        order=["low", "medium", "high"],
    )
    plt.xlabel("Triage label")
    plt.ylabel("Absolute count z-score")
    plt.title("Triage Separation by Absolute Count Z-Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> int:
    """Run the alignment experiment and save tables and plots."""
    args = build_parser().parse_args()
    sns.set_theme(style="whitegrid")

    output_dir = ensure_output_dir(args.output_dir)
    evaluation_frame = build_evaluation_frame(
        start_date=args.start_date,
        end_date=args.end_date,
        model_artifact_path=args.model_artifact_path,
        lookback_days=args.lookback_days,
        min_history_days=args.min_history_days,
        medium_percentile=args.medium_percentile,
        high_percentile=args.high_percentile,
    )

    if evaluation_frame.empty:
        raise ValueError("No evaluation rows were produced for the requested date range")

    metrics = correlation_summary(evaluation_frame)
    metrics["row_count"] = int(len(evaluation_frame))
    metrics["lookback_days"] = int(args.lookback_days)

    triage_stats = triage_summary(evaluation_frame)

    evaluation_frame.to_csv(output_dir / "evaluation_frame.csv", index=False)
    triage_stats.to_csv(output_dir / "triage_summary.csv", index=False)
    save_json(metrics, output_dir / "metrics.json")

    plot_scatter(
        evaluation_frame,
        x_column="abs_count_delta_from_mean",
        x_label="Absolute deviation from rolling mean",
        output_path=output_dir / "score_vs_abs_count_delta.png",
    )
    plot_scatter(
        evaluation_frame,
        x_column="abs_count_zscore",
        x_label="Absolute count z-score",
        output_path=output_dir / "score_vs_abs_count_zscore.png",
    )
    plot_boxplot(
        evaluation_frame,
        output_path=output_dir / "triage_vs_abs_count_zscore.png",
    )

    print(f"Saved alignment outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
