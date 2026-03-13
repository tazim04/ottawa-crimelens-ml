from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

DEFAULT_MODEL_VERSION = "isolation_forest_v1"
DEFAULT_EXCLUDE_COLUMNS = ("grid_id", "date")


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

    All numeric columns except metadata fields like ``grid_id`` and ``date``
    are treated as model inputs.
    """
    feature_columns = [
        column
        for column in feature_frame.columns
        if column not in exclude_columns and pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    if not feature_columns:
        raise ValueError("feature_frame does not contain any numeric feature columns")
    return feature_columns


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
        column for column in resolved_feature_columns if column not in feature_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"feature_frame is missing required feature columns: {missing_columns}"
        )

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
    resolved_feature_columns = feature_columns or infer_feature_columns(feature_frame)
    matrix = prepare_model_matrix(feature_frame, resolved_feature_columns)

    # Fit on the aligned numeric matrix so saved feature order matches scoring.
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
    output_path: str | Path,
) -> Path:
    """
    Persist a trained model artifact to disk with ``joblib``.
    """
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(asdict(artifact), resolved_path)
    return resolved_path


def load_model_artifact(input_path: str | Path) -> ModelArtifact:
    """
    Load a persisted model artifact from disk.
    """
    payload = joblib.load(Path(input_path))
    return ModelArtifact(**payload)


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

    # IsolationForest returns higher values for normal points, so invert it.
    anomaly_scores = -artifact.model.score_samples(matrix)

    output_columns = [
        column for column in ("grid_id", "date") if column in feature_frame.columns
    ]
    scored_frame = feature_frame[output_columns].copy() if output_columns else pd.DataFrame(index=feature_frame.index)
    scored_frame["anomaly_score"] = anomaly_scores
    scored_frame["model_version"] = artifact.model_version
    return scored_frame
