from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from app.db import engine
from app.feature_builder import build_daily_features
from app.model import load_model_artifact, score_feature_frame

from app.triage import assign_triage_labels

from app.triage import (
    DEFAULT_TRIAGE_HIGH_PERCENTILE,
    DEFAULT_TRIAGE_MEDIUM_PERCENTILE,
)


###### Pipeline-level scoring workflow #######


DEFAULT_MODEL_ARTIFACT_PATH = Path("artifacts/crime_model.joblib")
DEFAULT_SCORING_RESULTS_TABLE = "crime_anomaly_scores"
ToSqlIfExists = Literal[
    "fail", "replace", "append", "delete_rows"
]  # as per pandas.DataFrame.to_sql() if_exists parameter options


# Main pipeline function, orchestrating the full workflow with optional persistence of results.
def run_scoring_pipeline(
    target_date: str | date | datetime | None = None,
    *,
    model_artifact_path: str | Path | None = None,
    lookback_days: int | None = None,
    min_history_days: int | None = None,
    persist_results: bool = True,
    results_table: str = DEFAULT_SCORING_RESULTS_TABLE,
    if_exists: ToSqlIfExists = "append",
    high_percentile: float = DEFAULT_TRIAGE_HIGH_PERCENTILE,
    medium_percentile: float = DEFAULT_TRIAGE_MEDIUM_PERCENTILE,
) -> pd.DataFrame:
    """
    Execute the full daily scoring workflow.
    """
    resolved_target_date = resolve_scoring_date(target_date)
    scored_frame = score_daily_features(
        target_date=resolved_target_date,
        model_artifact_path=model_artifact_path,
        lookback_days=lookback_days,
        min_history_days=min_history_days,
        high_percentile=high_percentile,
        medium_percentile=medium_percentile,
    )

    if persist_results:
        persist_scored_results(
            scored_frame,
            table_name=results_table,
            if_exists=if_exists,
        )

    return scored_frame


##### Helper functions for the scoring workflow #####


def resolve_scoring_date(target_date: str | date | datetime | None = None) -> date:
    """
    Resolve the scoring date, defaulting to today's local date when omitted.
    """
    if target_date is None:
        return datetime.now().date()
    if isinstance(target_date, datetime):
        return target_date.date()
    if isinstance(target_date, date):
        return target_date
    return date.fromisoformat(target_date)


def resolve_model_artifact_path(model_artifact_path: str | Path | None = None) -> Path:
    """
    Resolve the trained model artifact location from args or environment.
    """
    candidate = model_artifact_path or os.getenv(
        "MODEL_ARTIFACT_PATH", str(DEFAULT_MODEL_ARTIFACT_PATH)
    )
    return Path(candidate)


def build_scoring_features(
    target_date: str | date | datetime,
    *,
    lookback_days: int | None = None,
    min_history_days: int | None = None,
) -> pd.DataFrame:
    """
    Build the feature frame used by the daily scoring workflow.
    Bases the feature construction on the target date and optional lookback/history parameters.
    """
    builder_kwargs: dict[str, int] = {}
    if lookback_days is not None:
        builder_kwargs["lookback_days"] = lookback_days
    if min_history_days is not None:
        builder_kwargs["min_history_days"] = min_history_days
    return build_daily_features(target_date=target_date, **builder_kwargs)


def persist_scored_results(
    scored_frame: pd.DataFrame,
    *,
    table_name: str = DEFAULT_SCORING_RESULTS_TABLE,
    if_exists: ToSqlIfExists = "append",
) -> int:
    """
    Persist daily scoring results to Postgres and return the inserted row count.
    """
    if scored_frame.empty:
        return 0

    scored_frame.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=False,
        method="multi",
    )
    return len(scored_frame)


def score_daily_features(
    target_date: str | date | datetime,
    *,
    model_artifact_path: str | Path | None = None,
    lookback_days: int | None = None,
    min_history_days: int | None = None,
    high_percentile: float = DEFAULT_TRIAGE_HIGH_PERCENTILE,
    medium_percentile: float = DEFAULT_TRIAGE_MEDIUM_PERCENTILE,
) -> pd.DataFrame:
    """
    Build features for a target day, score them with the trained model, and triage the output.
    """
    feature_frame = build_scoring_features(
        target_date=target_date,
        lookback_days=lookback_days,
        min_history_days=min_history_days,
    )
    if feature_frame.empty:
        return pd.DataFrame(
            columns=[
                "grid_id",
                "date",
                "anomaly_score",
                "model_version",
                "triage_percentile",
                "triage_label",
            ]
        )

    artifact = load_model_artifact(resolve_model_artifact_path(model_artifact_path))
    scored_frame = score_feature_frame(feature_frame, artifact)
    return assign_triage_labels(
        scored_frame,
        high_percentile=high_percentile,
        medium_percentile=medium_percentile,
    )
