from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from sqlalchemy import inspect, text

from app.db import engine
from app.features.feature_builder import build_daily_features
from app.features.constants import DEFAULT_LOOKBACK_DAYS
from app.model.model import load_model_artifact, score_feature_frame

from app.model.pipelines.triage.labelling import (
    add_triage_explanations,
    assign_triage_labels,
)

from app.model.pipelines.triage.labelling import (
    DEFAULT_TRIAGE_HIGH_PERCENTILE,
    DEFAULT_TRIAGE_MEDIUM_PERCENTILE,
)

logger = logging.getLogger(__name__)


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
) -> pd.DataFrame:
    """
    Execute the full daily scoring workflow.
    """
    logger.info("Starting daily scoring pipeline")
    resolved_target_date = resolve_scoring_date(target_date)
    scored_frame = score_daily_features(
        target_date=resolved_target_date,
        model_artifact_path=model_artifact_path,
        lookback_days=lookback_days,
        min_history_days=min_history_days,
        high_percentile=DEFAULT_TRIAGE_HIGH_PERCENTILE,
        medium_percentile=DEFAULT_TRIAGE_MEDIUM_PERCENTILE,
    )

    logger.info("Scoring pipeline completed for target date %s", resolved_target_date)

    if persist_results:
        logger.info(
            "Persisting scored results to Postgres table '%s' with if_exists='%s'",
            results_table,
            if_exists,
        )
        persist_scored_results(
            scored_frame,
            table_name=results_table,
            if_exists=if_exists,
        )
        logger.info("Scored results persisted to table '%s'", results_table)
    else:
        logger.info(
            "Persistence of scored results is disabled. Skipping database write."
        )
        logging.info("Writing to csv file")
        scored_frame.to_csv("scored_results.csv", index=False)

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

    target_url = engine.url.render_as_string(hide_password=True)
    logger.info(
        "Writing %s scored rows to '%s' on %s (if_exists='%s')",
        len(scored_frame),
        table_name,
        target_url,
        if_exists,
    )

    if if_exists == "append":
        ensure_result_table_columns(scored_frame, table_name=table_name)

    scored_frame.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=False,
        method="multi",
    )
    return len(scored_frame)


def ensure_result_table_columns(
    scored_frame: pd.DataFrame,
    *,
    table_name: str,
) -> None:
    """
    Add missing nullable columns to an existing results table before appending.
    """
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns(table_name=table_name)
    }
    missing_columns = [
        column for column in scored_frame.columns if column not in existing_columns
    ]
    if not missing_columns:
        return

    logger.info(
        "Adding missing columns to '%s' before append: %s",
        table_name,
        ", ".join(missing_columns),
    )
    with engine.begin() as connection:
        for column in missing_columns:
            connection.execute(
                text(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{column}" '
                    f"{sql_type_for_series(scored_frame[column])}"
                )
            )


def sql_type_for_series(series: pd.Series) -> str:
    """Map a pandas series to a simple Postgres column type for result persistence."""
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    if pd.api.types.is_object_dtype(series):
        non_null = series.dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], date):
            return "DATE"
    return "TEXT"


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
                "triage_explanation",
            ]
        )

    artifact = load_model_artifact(resolve_model_artifact_path(model_artifact_path))
    scored_frame = score_feature_frame(feature_frame, artifact)
    triaged_frame = assign_triage_labels(
        scored_frame,
        high_percentile=high_percentile,
        medium_percentile=medium_percentile,
    )
    return add_triage_explanations(
        triaged_frame,
        feature_frame,
        lookback_days=lookback_days
        if lookback_days is not None
        else DEFAULT_LOOKBACK_DAYS,
    )
