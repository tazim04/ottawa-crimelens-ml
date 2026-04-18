from __future__ import annotations

from datetime import date, datetime
import logging
from pathlib import Path

import pandas as pd

from app.features.feature_builder import build_training_features
from app.model.model import (
    DEFAULT_MODEL_VERSION,
    ModelArtifact,
    save_model_artifact,
    train_isolation_forest,
)

from app.features.constants import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_HISTORY_DAYS,
)
from app.model.storage import ArtifactLocation, resolve_artifact_location
from app.time_utils import local_today

logger = logging.getLogger(__name__)


##### Pipeline-level training workflow ######


DEFAULT_MODEL_ARTIFACT_PATH = Path("artifacts/crime_model.joblib")


# Main pipeline function, orchestrating the full workflow from feature building to model training and artifact persistence.
def run_training_pipeline(
    *,
    start_date: str,
    end_date: str | date | datetime | None = None,
    lookback_days: int | None = None,
    min_history_days: int | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    contamination: str | float = "auto",
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[ModelArtifact, ArtifactLocation]:
    """
    Build training features, train the model artifact, and persist it to disk.
    """
    logger.info("Building training dataset")
    resolved_end_date = resolve_training_end_date(end_date)
    training_features = build_training_dataset(
        start_date=start_date,
        end_date=resolved_end_date,
        lookback_days=lookback_days,
        min_history_days=min_history_days,
    )
    if training_features.empty:
        raise ValueError(
            "No training features were produced for the requested date range"
        )

    logging.info(
        "Training dataset built with %d rows and %d columns", *training_features.shape
    )

    logger.info("Training Isolation Forest model artifact")
    artifact = train_isolation_forest(
        training_features,
        model_version=model_version,
        contamination=parse_contamination(contamination),
        n_estimators=n_estimators,
        random_state=random_state,
    )
    saved_path = save_model_artifact(artifact, resolve_model_artifact_path())

    logger.info("Model artifact trained and saved to %s", saved_path)

    return artifact, saved_path


##### Helper functions for the training workflow #####


def resolve_model_artifact_path() -> ArtifactLocation:
    """
    Resolve the training artifact output path from environment or default.
    """
    return resolve_artifact_location(
        None,
        default_location=DEFAULT_MODEL_ARTIFACT_PATH,
    )


def resolve_training_end_date(
    end_date: str | date | datetime | None = None,
) -> str:
    """
    Resolve the training end date, defaulting to today's local date when omitted.
    """
    if end_date is None:
        return local_today().isoformat()
    if isinstance(end_date, datetime):
        return end_date.date().isoformat()
    if isinstance(end_date, date):
        return end_date.isoformat()
    return end_date


def parse_contamination(value: str | float = "auto") -> str | float:
    """
    Normalize contamination input to the type expected by the model layer.
    """
    if isinstance(value, float):
        return value
    if isinstance(value, str) and value == "auto":
        return value
    return float(value)


def build_training_dataset(
    start_date: str,
    end_date: str | date | datetime,
    *,
    lookback_days: int | None = None,
    min_history_days: int | None = None,
) -> pd.DataFrame:
    """
    Build the training feature frame for the requested historical date range.
    """
    return build_training_features(
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days
        if lookback_days is not None
        else DEFAULT_LOOKBACK_DAYS,
        min_history_days=min_history_days
        if min_history_days is not None
        else DEFAULT_MIN_HISTORY_DAYS,
        include_explanation_features=False,
    )
