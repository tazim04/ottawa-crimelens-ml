from datetime import date, datetime, timedelta

import pandas as pd
from sqlalchemy import text
import logging

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

logger = logging.getLogger(__name__)


##### Core Feature Engineering Layer ######


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
        include_explanation_features=True,
    )
    return features.reset_index(drop=True)


def build_training_features(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    *,
    include_explanation_features: bool = True,
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

    # Build model-ready features
    features = _build_feature_frame(
        start_date=resolved_start_date,
        end_date=resolved_end_date,
        lookback_days=lookback_days,
        min_history_days=min_history_days,
        include_explanation_features=include_explanation_features,
    )
    return features.reset_index(drop=True)


def _build_feature_frame(
    start_date: date,
    end_date: date,
    lookback_days: int,
    min_history_days: int,
    *,
    include_explanation_features: bool = True,
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
    logger.info(
        "Building features for %s to %s with lookback_days=%s min_history_days=%s (history window starts %s)",
        start_date,
        end_date,
        lookback_days,
        min_history_days,
        history_start_date,
    )
    crime_records = _fetch_crime_records(history_start_date, end_date)
    if crime_records.empty:
        logger.warning(
            "No crime records found between %s and %s; feature frame will be empty",
            history_start_date,
            end_date,
        )
        return pd.DataFrame()
    logger.info(
        "Fetched %s crime records across %s grids and %s event dates",
        len(crime_records),
        crime_records["grid_id"].nunique(dropna=True),
        pd.to_datetime(crime_records["event_date"]).nunique(dropna=True),
    )

    # Collapse raw crime rows into one dense row per grid/day.
    daily_frame, category_columns = _prepare_daily_frame(
        crime_records=crime_records,
        full_date_range=_date_range(history_start_date, end_date),
    )
    logger.info(
        "Prepared dense daily frame with %s rows across %s grids",
        len(daily_frame),
        daily_frame["grid_id"].nunique(dropna=True),
    )

    # Transform daily counts into model-ready numeric features.
    feature_frame = _compute_features(
        daily_frame=daily_frame,
        category_columns=category_columns,
        lookback_days=lookback_days,
        include_explanation_features=include_explanation_features,
    )
    logger.info(
        "Computed feature candidates: %s rows before target-date/history filtering",
        len(feature_frame),
    )

    target_slice = feature_frame[
        (feature_frame["date"] >= pd.Timestamp(start_date))
        & (feature_frame["date"] <= pd.Timestamp(end_date))
    ].copy()
    if target_slice.empty:
        logger.warning(
            "No feature rows matched requested date slice %s to %s before history filtering",
            start_date,
            end_date,
        )
    else:
        eligible_rows = int((target_slice["history_days"] >= min_history_days).sum())
        logger.info(
            "Target-date feature candidates: %s rows, %s eligible after history filter, history_days range=[%s, %s]",
            len(target_slice),
            eligible_rows,
            int(target_slice["history_days"].min()),
            int(target_slice["history_days"].max()),
        )

    # Keep only the requested slice once the rolling features are available.
    feature_frame = target_slice[
        target_slice["history_days"] >= min_history_days
    ].copy()
    if feature_frame.empty:
        logger.warning(
            "Feature frame is empty after applying history_days >= %s for %s to %s",
            min_history_days,
            start_date,
            end_date,
        )
    else:
        logger.info("Returning %s feature rows after filtering", len(feature_frame))

    # Return plain ``date`` objects for downstream compatibility.
    feature_frame["date"] = feature_frame["date"].dt.date
    return feature_frame


def _fetch_crime_records(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch raw crimes for the requested range using guarded fallback event timestamps.

    ``occurred_date`` is preferred when it looks plausible. ``reported_date`` is
    used when the occurred date is missing or falls after the reported date.
    """
    # Resolve one event date/hour per crime row directly in SQL for cleaner downstream processing. The query also flags when fallbacks were used
    query = text(
        f"""
        SELECT
            grid_id,
            CASE
                WHEN occurred_date IS NULL THEN reported_date
                WHEN reported_date IS NULL THEN occurred_date
                WHEN occurred_date > reported_date THEN reported_date
                ELSE occurred_date
            END::date AS event_date,
            COALESCE(occurred_hour, reported_hour) AS event_hour,
            offence_category,
            CASE
                WHEN occurred_date IS NULL AND reported_date IS NOT NULL THEN 1
                WHEN reported_date IS NOT NULL AND occurred_date > reported_date THEN 1
                ELSE 0
            END AS used_reported_date_fallback,
            CASE
                WHEN occurred_hour IS NULL AND reported_hour IS NOT NULL THEN 1
                ELSE 0
            END AS used_reported_hour_fallback
        FROM {CRIME_RECORDS_TABLE}
        WHERE grid_id IS NOT NULL
          AND (
              CASE
                  WHEN occurred_date IS NULL THEN reported_date
                  WHEN reported_date IS NULL THEN occurred_date
                  WHEN occurred_date > reported_date THEN reported_date
                  ELSE occurred_date
              END
          ) BETWEEN :start_date AND :end_date
        ORDER BY
            grid_id,
            (
                CASE
                    WHEN occurred_date IS NULL THEN reported_date
                    WHEN reported_date IS NULL THEN occurred_date
                    WHEN occurred_date > reported_date THEN reported_date
                    ELSE occurred_date
                END
            ),
            id
        """
    )
    return pd.read_sql_query(
        sql=query, con=engine, params={"start_date": start_date, "end_date": end_date}
    )
