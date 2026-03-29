from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.features.constants import TIME_BUCKET_COLUMNS
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.model.storage import (
    ArtifactLocation,
    resolve_model_artifact_storage,
)


##### Core ML Layer ######


DEFAULT_MODEL_VERSION = "isolation_forest_v1"

# Metadata fields to exclude from model features by default, they are not predictive
DEFAULT_EXCLUDE_COLUMNS = (
    "grid_id",
    "date",
)

# Patterns of feature columns to exclude from the model, they are redundant with other features
DEFAULT_DROPPED_FEATURE_PATTERNS = (
    "rolling_min_",
    "rolling_max_",
    "rolling_sum_",
    "_delta_from_mean",
    "_rolling_std_",
)


@dataclass(slots=True)
class ModelArtifact:
    """
    Persisted model bundle used by both training and scoring.

    The artifact stores the fitted estimator plus the exact feature-column
    schema needed to rebuild the model matrix during scoring.
    """

    model: IsolationForest
    feature_columns: list[str]
    model_version: str
    created_at: str
    training_row_count: int
    model_params: dict[str, Any]


def infer_feature_columns(
    feature_frame: pd.DataFrame,
    exclude_columns: tuple[str, ...] = DEFAULT_EXCLUDE_COLUMNS,
) -> list[str]:
    """
    Infer model feature columns from a feature frame.

    The model uses a curated subset of numeric features to reduce redundant
    dimensions while preserving baseline context, relative-change signals, and
    composition features.
    """
    feature_columns = [
        column
        for column in feature_frame.columns
        if column not in exclude_columns
        and pd.api.types.is_numeric_dtype(feature_frame[column])
        and _is_selected_model_feature(column)
    ]
    if not feature_columns:
        raise ValueError("feature_frame does not contain any numeric feature columns")
    return feature_columns


def _is_selected_model_feature(column: str) -> bool:
    """Return whether a numeric feature column should be used by the model."""
    if column in {
        "reported_date_fallback_rate",
        "reported_hour_fallback_rate",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "history_days",
        "total_crimes",
    }:
        return True

    if column.startswith("rolling_mean_") or column.startswith("rolling_std_"):
        return True
    if column.startswith("count_zscore_"):
        return True

    # Drop columns like *_rolling_min, *_delta_from_mean, etc that are redundant with the retained features and add dimensionality without clear predictive value
    # These include statistical summaries for time bucket and category features (meant to be used for calculating shifts for triage explanations)
    if any(pattern in column for pattern in DEFAULT_DROPPED_FEATURE_PATTERNS):
        return False

    if column in TIME_BUCKET_COLUMNS:
        return True
    if any(
        column.startswith(f"{bucket}_rolling_mean_") for bucket in TIME_BUCKET_COLUMNS
    ):
        return True
    if any(column.startswith(f"{bucket}_zscore_") for bucket in TIME_BUCKET_COLUMNS):
        return True

    if column.startswith("category_"):
        if "_share" in column:
            return True
        if "_rolling_mean_" in column:
            return True
        if "_zscore_" in column:
            return True
        return True

    return False


def prepare_model_matrix(
    feature_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a clean numeric matrix for model training or scoring.

    If ``feature_columns`` is omitted, the function infers the schema from the
    provided frame. Missing required columns raise immediately.
    """
    resolved_feature_columns = feature_columns or infer_feature_columns(feature_frame)
    missing_columns = [
        column
        for column in resolved_feature_columns
        if column not in feature_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"feature_frame is missing required feature columns: {missing_columns}"
        )

    # Select and coerce the feature columns to a clean numeric matrix, filling non-convertible values with 0.0
    matrix = feature_frame[resolved_feature_columns].copy()
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return matrix.astype(float)


def train_isolation_forest(
    feature_frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    contamination: str | float = "auto",
    n_estimators: int = 200,
    random_state: int = 42,
) -> ModelArtifact:
    """
    Train an Isolation Forest on the provided feature frame.

    The returned artifact includes both the fitted model and the exact feature
    schema required to score future data consistently.
    """

    # Prepare the model matrix and infer feature columns if not provided
    resolved_feature_columns = feature_columns or infer_feature_columns(feature_frame)
    matrix = prepare_model_matrix(feature_frame, resolved_feature_columns)

    # Fit on the aligned numeric matrix so saved feature order matches scoring
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(matrix)

    return ModelArtifact(
        model=model,
        feature_columns=resolved_feature_columns,
        model_version=model_version,
        created_at=datetime.now(UTC).isoformat(),
        training_row_count=len(matrix),
        model_params={
            "contamination": contamination,
            "n_estimators": n_estimators,
            "random_state": random_state,
        },
    )


def save_model_artifact(
    artifact: ModelArtifact,
    output_path: ArtifactLocation,
) -> ArtifactLocation:
    """
    Persist a trained model artifact to the configured storage backend.
    """
    storage = resolve_model_artifact_storage(output_path)
    return storage.save_artifact(artifact, output_path)


def load_model_artifact(input_path: ArtifactLocation) -> ModelArtifact:
    """
    Load a persisted model artifact from the configured storage backend.
    """
    storage = resolve_model_artifact_storage(input_path)
    return storage.load_artifact(input_path)


def score_feature_frame(
    feature_frame: pd.DataFrame,
    artifact: ModelArtifact,
) -> pd.DataFrame:
    """
    Score a feature frame with a trained artifact.

    The returned frame preserves ``grid_id`` and ``date`` when present and adds
    an ``anomaly_score`` where larger values indicate more anomalous rows.
    """
    matrix = prepare_model_matrix(feature_frame, artifact.feature_columns)

    # Compute scores, IsolationForest returns higher values for normal points so invert it
    anomaly_scores = -artifact.model.score_samples(matrix)

    # Preserve metadata columns for output
    output_columns = [
        column for column in ("grid_id", "date") if column in feature_frame.columns
    ]

    # Build and return the scored frame with metadata and scores
    scored_frame = (
        feature_frame[output_columns].copy()
        if output_columns
        else pd.DataFrame(index=feature_frame.index)
    )
    scored_frame["anomaly_score"] = anomaly_scores
    scored_frame["model_version"] = artifact.model_version
    return scored_frame
