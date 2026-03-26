from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.features.constants import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_HISTORY_DAYS,
)
from app.features.feature_builder import build_training_features
from app.model.model import score_feature_frame, train_isolation_forest
from experiments.common import ensure_output_dir, save_json, top_quantile_members


DEFAULT_OUTPUT_DIR = Path("experiments/output/stability")

##### Stability analysis across random seeds and lookback windows. Verifies the consistency of anomaly detection results. #####


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the stability experiment."""
    parser = argparse.ArgumentParser(
        description="Measure score-ranking stability across random seeds and lookback windows."
    )
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--seeds",
        default="7,42,99",
        help="Comma-separated random seeds to compare.",
    )
    parser.add_argument(
        "--lookbacks",
        default=str(DEFAULT_LOOKBACK_DAYS),
        help="Comma-separated lookback windows to compare.",
    )
    parser.add_argument(
        "--min-history-days",
        type=int,
        default=DEFAULT_MIN_HISTORY_DAYS,
        help="Minimum history days required per row.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees used by each Isolation Forest run.",
    )
    parser.add_argument(
        "--top-quantile",
        type=float,
        default=0.9,
        help="Quantile threshold used to compare overlap among top anomalies.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write stability tables and plots into.",
    )
    return parser


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def pairwise_seed_metrics(
    scored_runs: dict[int, pd.DataFrame],
    *,
    top_quantile: float,
) -> pd.DataFrame:
    """Compute pairwise ranking agreement across random seeds."""
    rows: list[dict[str, float | int | str]] = []
    for left_seed, right_seed in combinations(sorted(scored_runs), 2):
        merged = scored_runs[left_seed].merge(
            scored_runs[right_seed],
            on=["grid_id", "date"],
            suffixes=("_left", "_right"),
            how="inner",
        )
        left_top = top_quantile_members(
            scored_runs[left_seed],
            score_column="anomaly_score",
            quantile=top_quantile,
        )
        right_top = top_quantile_members(
            scored_runs[right_seed],
            score_column="anomaly_score",
            quantile=top_quantile,
        )
        union_size = len(left_top | right_top)
        overlap = len(left_top & right_top) / union_size if union_size else 1.0
        rows.append(
            {
                "seed_left": left_seed,
                "seed_right": right_seed,
                "spearman_score_correlation": float(
                    merged["anomaly_score_left"].corr(
                        merged["anomaly_score_right"],
                        method="spearman",
                    )
                ),
                "top_quantile_jaccard": float(overlap),
            }
        )
    return pd.DataFrame(rows)


def lookback_metrics(
    lookback_runs: dict[int, pd.DataFrame],
    *,
    top_quantile: float,
) -> pd.DataFrame:
    """Compute pairwise agreement across lookback windows."""
    rows: list[dict[str, float | int]] = []
    for left_lookback, right_lookback in combinations(sorted(lookback_runs), 2):
        merged = lookback_runs[left_lookback].merge(
            lookback_runs[right_lookback],
            on=["grid_id", "date"],
            suffixes=("_left", "_right"),
            how="inner",
        )
        left_top = top_quantile_members(
            lookback_runs[left_lookback],
            score_column="anomaly_score",
            quantile=top_quantile,
        )
        right_top = top_quantile_members(
            lookback_runs[right_lookback],
            score_column="anomaly_score",
            quantile=top_quantile,
        )
        union_size = len(left_top | right_top)
        overlap = len(left_top & right_top) / union_size if union_size else 1.0
        rows.append(
            {
                "lookback_left": left_lookback,
                "lookback_right": right_lookback,
                "spearman_score_correlation": float(
                    merged["anomaly_score_left"].corr(
                        merged["anomaly_score_right"],
                        method="spearman",
                    )
                ),
                "top_quantile_jaccard": float(overlap),
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(
    frame: pd.DataFrame,
    *,
    index_column: str,
    columns_column: str,
    value_column: str,
    title: str,
    output_path: Path,
) -> None:
    """Render a simple heatmap for pairwise stability metrics."""
    if frame.empty:
        return
    pivot = frame.pivot(index=index_column, columns=columns_column, values=value_column)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap="Blues", vmin=0.0, vmax=1.0, fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> int:
    """Run stability experiments across seeds and lookback windows."""
    args = build_parser().parse_args()
    sns.set_theme(style="whitegrid")
    output_dir = ensure_output_dir(args.output_dir)

    seeds = parse_int_list(args.seeds)
    lookbacks = parse_int_list(args.lookbacks)

    seed_features = build_training_features(
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=DEFAULT_LOOKBACK_DAYS,
        min_history_days=args.min_history_days,
    )
    if seed_features.empty:
        raise ValueError("No rows were produced for the requested seed stability range")

    seed_runs: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        artifact = train_isolation_forest(
            seed_features,
            n_estimators=args.n_estimators,
            random_state=seed,
        )
        seed_runs[seed] = score_feature_frame(seed_features, artifact)

    seed_summary = pairwise_seed_metrics(
        seed_runs,
        top_quantile=args.top_quantile,
    )
    seed_summary.to_csv(output_dir / "seed_stability.csv", index=False)

    lookback_runs: dict[int, pd.DataFrame] = {}
    for lookback in lookbacks:
        feature_frame = build_training_features(
            start_date=args.start_date,
            end_date=args.end_date,
            lookback_days=lookback,
            min_history_days=min(args.min_history_days, lookback),
        )
        if feature_frame.empty:
            continue
        artifact = train_isolation_forest(
            feature_frame,
            n_estimators=args.n_estimators,
            random_state=seeds[0],
        )
        lookback_runs[lookback] = score_feature_frame(feature_frame, artifact)

    lookback_summary = lookback_metrics(
        lookback_runs,
        top_quantile=args.top_quantile,
    )
    lookback_summary.to_csv(output_dir / "lookback_stability.csv", index=False)

    plot_heatmap(
        seed_summary,
        index_column="seed_left",
        columns_column="seed_right",
        value_column="spearman_score_correlation",
        title="Seed Stability: Spearman Correlation",
        output_path=output_dir / "seed_spearman_heatmap.png",
    )
    plot_heatmap(
        lookback_summary,
        index_column="lookback_left",
        columns_column="lookback_right",
        value_column="spearman_score_correlation",
        title="Lookback Stability: Spearman Correlation",
        output_path=output_dir / "lookback_spearman_heatmap.png",
    )

    summary = {
        "seed_runs": seeds,
        "lookback_runs": lookbacks,
        "top_quantile": args.top_quantile,
        "seed_pair_count": int(len(seed_summary)),
        "lookback_pair_count": int(len(lookback_summary)),
    }
    save_json(summary, output_dir / "summary.json")

    print(f"Saved stability outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
