from datetime import date, datetime, timedelta

import pandas as pd
from sqlalchemy import text

from app.db import engine
from app.features.aggregation import (
    compute_features as _compute_features,
    prepare_daily_frame as _prepare_daily_frame,
)
from app.features.constants import (
    CRIME_RECORDS_TABLE,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_HISTORY_DAYS,
)
from app.features.utils import (
    build_date_range as _date_range,
    coerce_date as _coerce_date,
)


def build_daily_features(
    target_date: str | date | datetime,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> pd.DataFrame:
    """
    Build model-ready features for a single scoring date.

    The returned frame contains one row per grid cell for ``target_date``.
    Rolling features are computed from the prior ``lookback_days`` and do not
    include the target day itself.
    """
    # Normalize caller input before handing off to the shared builder path.
    resolved_target_date = _coerce_date(target_date)
    features = _build_feature_frame(
        start_date=resolved_target_date,
        end_date=resolved_target_date,
        lookback_days=lookback_days,
        min_history_days=min_history_days,
    )
    return features.reset_index(drop=True)


def build_training_features(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> pd.DataFrame:
    """
    Build model-ready features for a historical training range.

    The returned frame contains one row per grid cell per day between
    ``start_date`` and ``end_date`` inclusive, filtered to rows with enough
    observed history to support the requested rolling window.
    """
    # Normalize and validate the requested training range up front.
    resolved_start_date = _coerce_date(start_date)
    resolved_end_date = _coerce_date(end_date)
    if resolved_start_date > resolved_end_date:
        raise ValueError("start_date must be on or before end_date")

    features = _build_feature_frame(
        start_date=resolved_start_date,
        end_date=resolved_end_date,
        lookback_days=lookback_days,
        min_history_days=min_history_days,
    )
    return features.reset_index(drop=True)


def _build_feature_frame(
    start_date: date,
    end_date: date,
    lookback_days: int,
    min_history_days: int,
) -> pd.DataFrame:
    """
    Fetch raw crimes, aggregate them to daily grid rows, and compute features.

    This is the common path shared by training and daily scoring.
    """
    # Guardrails keep the rolling-window contract sane for both workflows.
    if lookback_days < 2:
        raise ValueError("lookback_days must be at least 2")
    if min_history_days < 1:
        raise ValueError("min_history_days must be at least 1")
    if min_history_days > lookback_days:
        raise ValueError("min_history_days cannot exceed lookback_days")

    # Pull extra history before the target range so rolling stats have context.
    history_start_date = start_date - timedelta(days=lookback_days)
    crime_records = _fetch_crime_records(history_start_date, end_date)
    if crime_records.empty:
        return pd.DataFrame()

    # Collapse raw crime rows into one dense row per grid/day.
    daily_frame, category_columns = _prepare_daily_frame(
        crime_records=crime_records,
        full_date_range=_date_range(history_start_date, end_date),
    )

    # Transform daily counts into model-ready numeric features.
    feature_frame = _compute_features(
        daily_frame=daily_frame,
        category_columns=category_columns,
        lookback_days=lookback_days,
    )

    # Keep only the requested slice once the rolling features are available.
    feature_frame = feature_frame[
        (feature_frame["date"] >= pd.Timestamp(start_date))
        & (feature_frame["date"] <= pd.Timestamp(end_date))
        & (feature_frame["history_days"] >= min_history_days)
    ].copy()

    # Return plain ``date`` objects for downstream compatibility.
    feature_frame["date"] = feature_frame["date"].dt.date
    return feature_frame


def _fetch_crime_records(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch raw crimes for the requested range using fallback event timestamps.

    ``occurred_*`` is treated as the preferred signal because it best reflects
    when the incident happened. ``reported_*`` is used as a fallback when the
    occurred fields are missing.
    """
    # Resolve one event date/hour per crime row directly in SQL.
    query = text(
        f"""
        SELECT
            grid_id,
            COALESCE(occurred_date, reported_date)::date AS event_date,
            COALESCE(occurred_hour, reported_hour) AS event_hour,
            offence_category,
            CASE
                WHEN occurred_date IS NULL AND reported_date IS NOT NULL THEN 1
                ELSE 0
            END AS used_reported_date_fallback,
            CASE
                WHEN occurred_hour IS NULL AND reported_hour IS NOT NULL THEN 1
                ELSE 0
            END AS used_reported_hour_fallback
        FROM {CRIME_RECORDS_TABLE}
        WHERE grid_id IS NOT NULL
          AND COALESCE(occurred_date, reported_date) BETWEEN :start_date AND :end_date
        ORDER BY grid_id, COALESCE(occurred_date, reported_date), id
        """
    )
    return pd.read_sql_query(
        sql=query,
        con=engine,
        params={"start_date": start_date, "end_date": end_date},
    )
