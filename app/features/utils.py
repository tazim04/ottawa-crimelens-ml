from datetime import date, datetime

import pandas as pd


def coerce_date(value: str | date | datetime) -> date:
    """
    Convert accepted date-like inputs to ``datetime.date``.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def build_date_range(start_date: date, end_date: date) -> pd.DatetimeIndex:
    """
    Build an inclusive daily date range for reindexing and filtering.
    """
    # Use a dense daily index so missing grid/day combinations can be filled.
    return pd.date_range(start=start_date, end=end_date, freq="D")


def category_to_feature_name(category: str) -> str:
    """
    Normalize an offence category label into a stable feature column name.
    
    eg: "Theft $5000 and Under" -> "category_theft_5000_and_under"
    """
    # Collapse punctuation and spacing differences into predictable column ids
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(category).strip()
    )
    normalized = "_".join(part for part in normalized.split("_") if part)
    if not normalized:
        normalized = "unknown"
    return f"category_{normalized}"


def grouped_rolling(
    series: pd.Series,
    group_keys: pd.Series,
    window: int,
    operation: str,
) -> pd.Series:
    """
    Apply a rolling aggregation independently within each grid_id series.
    """
    # Reset the group index so the result lines up with the caller's frame
    grouped = series.groupby(group_keys)
    rolling = grouped.rolling(window=window, min_periods=1)
    result = getattr(rolling, operation)()
    return result.reset_index(level=0, drop=True)
