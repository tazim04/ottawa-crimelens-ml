from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.features.feature_builder import build_training_features
from app.model.model import load_model_artifact, score_feature_frame
from app.model.pipelines.triage.labelling import assign_triage_labels

##### Common utilities for experiments, such as building evaluation frames and selecting top quantiles. #####


def window_suffix(lookback_days: int) -> str:
    """Return the feature-name suffix used for rolling-window columns."""
    return f"{lookback_days}d"


def zscore_column_name(lookback_days: int) -> str:
    """Return the total-count z-score column for a lookback window."""
    return f"count_zscore_{window_suffix(lookback_days)}"


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create and return the experiment output directory."""
    resolved = Path(output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(payload: dict[str, object], output_path: str | Path) -> Path:
    """Persist a JSON payload with readable formatting."""
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved


def build_evaluation_frame(
    *,
    start_date: str,
    end_date: str,
    model_artifact_path: str | Path,
    lookback_days: int,
    min_history_days: int,
    medium_percentile: float,
    high_percentile: float,
) -> pd.DataFrame:
    """
    Build a scored historical frame suitable for evaluation experiments.

    The resulting frame combines features, anomaly scores, and triage metadata so
    that experiments can operate on one aligned dataset.
    """
    feature_frame = build_training_features(
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        min_history_days=min_history_days,
    )
    artifact = load_model_artifact(model_artifact_path)
    scored_frame = score_feature_frame(feature_frame, artifact)
    triaged_frame = assign_triage_labels(
        scored_frame,
        medium_percentile=medium_percentile,
        high_percentile=high_percentile,
    )
    evaluation_frame = feature_frame.merge(
        triaged_frame,
        on=["grid_id", "date"],
        how="inner",
    )

    total_zscore_column = zscore_column_name(lookback_days)
    evaluation_frame["abs_count_delta_from_mean"] = evaluation_frame[
        "count_delta_from_mean"
    ].abs()
    evaluation_frame["abs_count_zscore"] = evaluation_frame[total_zscore_column].abs()
    return evaluation_frame


def top_quantile_members(
    frame: pd.DataFrame,
    *,
    score_column: str,
    quantile: float,
) -> set[tuple[str, str]]:
    """Return the keys for rows in the top quantile of a score column."""
    if frame.empty:
        return set()
    threshold = frame[score_column].quantile(quantile)
    selected = frame.loc[frame[score_column] >= threshold, ["grid_id", "date"]].copy()
    selected["date"] = selected["date"].astype(str)
    return set(selected.itertuples(index=False, name=None))
