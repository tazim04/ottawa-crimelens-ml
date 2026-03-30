import numpy as np
import pandas as pd

from app.features.constants import (
    CATEGORY_FEATURE_COLUMNS,
    OFFENCE_CATEGORIES,
    TIME_BUCKET_COLUMNS,
    TIME_FALLBACK_COLUMNS,
)
from app.features.utils import category_to_feature_name, grouped_rolling


def prepare_daily_frame(
    crime_records: pd.DataFrame,
    full_date_range: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregate raw crime rows into a dense daily grid feature frame.
    The returned frame has one row per grid cell per day, with zero-filled counts for missing days.
    Used in compute_features to build actual model-facing features.
    """
    # Copy the input frame to avoid mutating caller data
    frame = crime_records.copy()

    # Normalize the raw columns into a predictable aggregation schema
    frame["grid_id"] = frame["grid_id"].astype(str)
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    frame["event_hour"] = pd.to_numeric(frame["event_hour"], errors="coerce")
    frame["used_reported_date_fallback"] = pd.to_numeric(
        frame["used_reported_date_fallback"], errors="coerce"
    ).fillna(0.0)
    frame["used_reported_hour_fallback"] = pd.to_numeric(
        frame["used_reported_hour_fallback"], errors="coerce"
    ).fillna(0.0)
    frame["offence_category"] = (
        frame.get("offence_category", pd.Series(index=frame.index, dtype="object"))
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    # Build one row per grid/day before adding dense zero-filled dates
    grouped = frame.groupby(["grid_id", "event_date"], sort=True)
    daily_totals = grouped.size().rename("total_crimes")
    daily_time = grouped.apply(summarize_time_distribution).reset_index()

    category_counts = (
        frame.groupby(["grid_id", "event_date", "offence_category"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    category_counts["count"] = category_counts["count"].astype(float)
    category_columns = CATEGORY_FEATURE_COLUMNS.copy()

    # Expand the fixed offence-category vocabulary into stable feature columns
    if OFFENCE_CATEGORIES:
        category_counts = category_counts[
            category_counts["offence_category"].isin(OFFENCE_CATEGORIES)
        ].copy()

        category_counts["feature_name"] = category_counts["offence_category"].map(
            category_to_feature_name
        )

        category_frame = category_counts.pivot_table(
            index=["grid_id", "event_date"],
            columns="feature_name",
            values="count",
            aggfunc="sum",
            fill_value=0.0,
        )

        # Materialize zero columns for categories absent in this slice in one pass
        category_frame = category_frame.reindex(
            columns=category_columns, fill_value=0.0
        )
        category_frame = category_frame.reset_index()
        category_feature_columns = category_columns
    else:
        category_frame = pd.DataFrame(columns=["grid_id", "event_date"])
        category_feature_columns = []

    # Merge totals, time buckets, and category counts into one daily frame
    daily_frame = daily_totals.reset_index().merge(
        daily_time,
        on=["grid_id", "event_date"],
        how="left",
    )
    if category_feature_columns:
        daily_frame = daily_frame.merge(
            category_frame,
            on=["grid_id", "event_date"],
            how="left",
        )

    # Track which rows came from source data before reindexing fills gaps
    daily_frame["has_source_row"] = 1.0
    daily_frame = daily_frame.rename(columns={"event_date": "date"})

    # Reindex across the full grid/date matrix so zero-crime days stay explicit
    grid_ids = sorted(daily_frame["grid_id"].unique())
    full_index = pd.MultiIndex.from_product(
        [grid_ids, full_date_range],
        names=["grid_id", "date"],
    )
    daily_frame = (
        daily_frame.set_index(["grid_id", "date"]).reindex(full_index).reset_index()
    )

    # Fill all engineered base columns so later math can stay purely numeric
    fill_columns = [
        "total_crimes",
        *TIME_BUCKET_COLUMNS,
        *TIME_FALLBACK_COLUMNS,
        *category_feature_columns,
        "has_source_row",
    ]
    for column in fill_columns:
        daily_frame[column] = pd.to_numeric(
            daily_frame[column], errors="coerce"
        ).fillna(0.0)

    return daily_frame, category_feature_columns


def compute_features(
    daily_frame: pd.DataFrame,
    category_columns: list[str],
    lookback_days: int,
    *,
    include_explanation_features: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling, share, fallback-rate, and calendar features.

    Returns model-facing feature frame with one row per grid/day, with static
    features such as rolling averages and share metrics.

    ``history_days`` counts prior observed grid/day rows only. Zero-filled
    dates remain useful for rolling totals but do not count as observed history.
    """
    # Sort within each grid so shifts and rolling windows line up correctly
    frame = daily_frame.sort_values(["grid_id", "date"]).copy()
    grouped_total = frame.groupby("grid_id")["total_crimes"]
    prior_total = grouped_total.shift(1)
    prior_observed = frame.groupby("grid_id")["has_source_row"].shift(1).fillna(0.0)
    window_suffix = f"{lookback_days}d"
    derived_columns: dict[str, pd.Series] = {}

    # Rolling stats use only prior days to avoid leaking the current day
    history_days = prior_observed.groupby(frame["grid_id"]).cumsum()
    rolling_mean = grouped_rolling(prior_total, frame["grid_id"], lookback_days, "mean")
    rolling_std = grouped_rolling(
        prior_total, frame["grid_id"], lookback_days, "std"
    ).fillna(0.0)

    derived_columns["history_days"] = history_days
    derived_columns[f"rolling_mean_{window_suffix}"] = rolling_mean
    derived_columns[f"rolling_std_{window_suffix}"] = rolling_std
    if include_explanation_features:
        derived_columns[f"rolling_min_{window_suffix}"] = grouped_rolling(
            prior_total,
            frame["grid_id"],
            lookback_days,
            "min",
        )
        derived_columns[f"rolling_max_{window_suffix}"] = grouped_rolling(
            prior_total,
            frame["grid_id"],
            lookback_days,
            "max",
        )
        derived_columns[f"rolling_sum_{window_suffix}"] = grouped_rolling(
            prior_total,
            frame["grid_id"],
            lookback_days,
            "sum",
        )

    # Express the current day relative to its recent local baseline
    count_delta = frame["total_crimes"] - rolling_mean.fillna(0.0)
    safe_std = rolling_std.replace(0.0, np.nan)
    if include_explanation_features:
        derived_columns["count_delta_from_mean"] = count_delta
    derived_columns[f"count_zscore_{window_suffix}"] = (count_delta / safe_std).fillna(
        0.0
    )

    # Track category-specific baselines so explanations can name the crime type that changed
    for column in category_columns:
        grouped_category = frame.groupby("grid_id")[column]
        prior_category = grouped_category.shift(1)
        rolling_mean_column = f"{column}_rolling_mean_{window_suffix}"
        rolling_std_column = f"{column}_rolling_std_{window_suffix}"
        delta_column = f"{column}_delta_from_mean"
        zscore_column = f"{column}_zscore_{window_suffix}"

        category_rolling_mean = grouped_rolling(
            prior_category, frame["grid_id"], lookback_days, "mean"
        ).fillna(0.0)
        category_rolling_std = grouped_rolling(
            prior_category, frame["grid_id"], lookback_days, "std"
        ).fillna(0.0)
        category_delta = frame[column] - category_rolling_mean
        safe_category_std = category_rolling_std.replace(0.0, np.nan)

        derived_columns[rolling_mean_column] = category_rolling_mean
        if include_explanation_features:
            derived_columns[rolling_std_column] = category_rolling_std
            derived_columns[delta_column] = category_delta
        derived_columns[zscore_column] = (category_delta / safe_category_std).fillna(
            0.0
        )

    # Track time-bucket baselines so explanations can describe shifts in incident timing
    for column in TIME_BUCKET_COLUMNS:
        grouped_bucket = frame.groupby("grid_id")[column]
        prior_bucket = grouped_bucket.shift(1)
        rolling_mean_column = f"{column}_rolling_mean_{window_suffix}"
        rolling_std_column = f"{column}_rolling_std_{window_suffix}"
        delta_column = f"{column}_delta_from_mean"
        zscore_column = f"{column}_zscore_{window_suffix}"

        bucket_rolling_mean = grouped_rolling(
            prior_bucket, frame["grid_id"], lookback_days, "mean"
        ).fillna(0.0)
        bucket_rolling_std = grouped_rolling(
            prior_bucket, frame["grid_id"], lookback_days, "std"
        ).fillna(0.0)
        bucket_delta = frame[column] - bucket_rolling_mean
        safe_bucket_std = bucket_rolling_std.replace(0.0, np.nan)

        derived_columns[rolling_mean_column] = bucket_rolling_mean
        if include_explanation_features:
            derived_columns[rolling_std_column] = bucket_rolling_std
            derived_columns[delta_column] = bucket_delta
        derived_columns[zscore_column] = (bucket_delta / safe_bucket_std).fillna(0.0)

    # Convert raw counts into per-day composition features
    total_nonzero = frame["total_crimes"].replace(0.0, np.nan)
    for column in category_columns:
        derived_columns[f"{column}_share"] = (frame[column] / total_nonzero).fillna(0.0)

    bucket_total = frame[TIME_BUCKET_COLUMNS].sum(axis=1).replace(0.0, np.nan)
    for column in TIME_BUCKET_COLUMNS:
        derived_columns[f"{column}_share"] = (frame[column] / bucket_total).fillna(0.0)

    # Keep fallback usage as explicit signal instead of hiding imputations
    derived_columns["reported_date_fallback_rate"] = (
        frame["used_reported_date_fallback_count"] / total_nonzero
    ).fillna(0.0)
    derived_columns["reported_hour_fallback_rate"] = (
        frame["used_reported_hour_fallback_count"] / total_nonzero
    ).fillna(0.0)

    # Add cyclical calendar context for weekly patterns
    day_of_week = frame["date"].dt.dayofweek.astype(float)
    derived_columns["day_of_week_sin"] = pd.Series(
        np.sin(2 * np.pi * day_of_week / 7.0),
        index=frame.index,
    )
    derived_columns["day_of_week_cos"] = pd.Series(
        np.cos(2 * np.pi * day_of_week / 7.0),
        index=frame.index,
    )
    derived_columns["is_weekend"] = day_of_week.isin([5.0, 6.0]).astype(float)
    if include_explanation_features:
        derived_columns["day_of_week"] = day_of_week

    frame = pd.concat(
        [frame, pd.DataFrame(derived_columns, index=frame.index)],
        axis=1,
    )

    # Return only the model-facing columns from the wider working frame
    feature_columns = [
        "grid_id",
        "date",
        "total_crimes",
        "history_days",
        f"rolling_mean_{window_suffix}",
        f"rolling_std_{window_suffix}",
        f"count_zscore_{window_suffix}",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "reported_date_fallback_rate",
        "reported_hour_fallback_rate",
        *category_columns,
        *TIME_BUCKET_COLUMNS,
        *[f"{column}_rolling_mean_{window_suffix}" for column in category_columns],
        *[f"{column}_zscore_{window_suffix}" for column in category_columns],
        *[f"{column}_rolling_mean_{window_suffix}" for column in TIME_BUCKET_COLUMNS],
        *[f"{column}_zscore_{window_suffix}" for column in TIME_BUCKET_COLUMNS],
        *[f"{column}_share" for column in category_columns],
        *[f"{column}_share" for column in TIME_BUCKET_COLUMNS],
    ]
    if include_explanation_features:
        feature_columns.extend(
            [
                f"rolling_min_{window_suffix}",
                f"rolling_max_{window_suffix}",
                f"rolling_sum_{window_suffix}",
                "count_delta_from_mean",
                "day_of_week",
                *[
                    f"{column}_rolling_std_{window_suffix}"
                    for column in category_columns
                ],
                *[f"{column}_delta_from_mean" for column in category_columns],
                *[
                    f"{column}_rolling_std_{window_suffix}"
                    for column in TIME_BUCKET_COLUMNS
                ],
                *[f"{column}_delta_from_mean" for column in TIME_BUCKET_COLUMNS],
            ]
        )
    feature_frame = frame[feature_columns].copy()
    numeric_columns = [
        column for column in feature_columns if column not in {"grid_id", "date"}
    ]
    feature_frame[numeric_columns] = feature_frame[numeric_columns].fillna(0.0)
    return feature_frame


def summarize_time_distribution(group: pd.DataFrame) -> pd.Series:
    """
    Summarize time-of-day buckets and fallback counts for one grid/day group.
    """
    # Bucket HHMM-style event hours into coarse time-of-day counts
    event_hours = pd.to_numeric(group["event_hour"], errors="coerce")
    return pd.Series(
        {
            "night_crimes": ((event_hours // 100).between(0, 5)).sum(),
            "morning_crimes": ((event_hours // 100).between(6, 11)).sum(),
            "afternoon_crimes": ((event_hours // 100).between(12, 17)).sum(),
            "evening_crimes": ((event_hours // 100).between(18, 23)).sum(),
            "unknown_hour_crimes": event_hours.isna().sum(),
            "used_reported_date_fallback_count": group[
                "used_reported_date_fallback"
            ].sum(),
            "used_reported_hour_fallback_count": group[
                "used_reported_hour_fallback"
            ].sum(),
        }
    )
