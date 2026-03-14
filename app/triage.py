from __future__ import annotations

import pandas as pd

DEFAULT_TRIAGE_HIGH_PERCENTILE = 0.9
DEFAULT_TRIAGE_MEDIUM_PERCENTILE = 0.75

_CATEGORY_LABELS = {
    "category_arson_share": "arson",
    "category_assaults_share": "assaults",
    "category_attempted_murder_share": "attempted murder",
    "category_break_and_enter_share": "break and enter",
    "category_criminal_harassment_share": "criminal harassment",
    "category_homicide_share": "homicide",
    "category_indecent_or_harassing_communications_share": "indecent or harassing communications",
    "category_mischief_share": "mischief",
    "category_robbery_share": "robbery",
    "category_theft_5000_and_under_share": "theft $5000 and under",
    "category_theft_over_5000_share": "theft over $5000",
    "category_theft_of_motor_vehicle_share": "theft of motor vehicle",
    "category_uttering_threats_share": "uttering threats",
}
_TIME_BUCKET_LABELS = {
    "night_crimes_share": "night",
    "morning_crimes_share": "morning",
    "afternoon_crimes_share": "afternoon",
    "evening_crimes_share": "evening",
    "unknown_hour_crimes_share": "unknown-hour",
}


def add_triage_explanations(
    triaged_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Attach human-readable explanation text for each row's triage outcome.

    The explanation is heuristic: it summarizes the strongest observable feature
    signals behind the anomaly score rather than attempting an exact model
    decomposition.
    """
    # If there are no triaged rows, return an empty explanation column with the right type
    if triaged_frame.empty:
        explained_frame = triaged_frame.copy()
        explained_frame["triage_explanation"] = pd.Series(dtype="object")
        return explained_frame

    # Build explanations by comparing the scored features against recent historical baselines and highlighting dominant labels
    window_suffix = f"{lookback_days}d"
    explanation_source = feature_frame.copy()
    explanation_source = explanation_source.merge(
        triaged_frame[
            ["grid_id", "date", "anomaly_score", "triage_percentile", "triage_label"]
        ],
        on=["grid_id", "date"],
        how="left",
    )
    explanation_source["triage_explanation"] = explanation_source.apply(
        lambda row: _build_triage_explanation(row, window_suffix=window_suffix),
        axis=1,
    )

    return triaged_frame.merge(
        explanation_source[["grid_id", "date", "triage_explanation"]],
        on=["grid_id", "date"],
        how="left",
    )


def assign_triage_labels(
    scored_frame: pd.DataFrame,
    *,
    high_percentile: float = DEFAULT_TRIAGE_HIGH_PERCENTILE,
    medium_percentile: float = DEFAULT_TRIAGE_MEDIUM_PERCENTILE,
) -> pd.DataFrame:
    """
    Add triage labels derived from relative anomaly-score severity.
    """
    if scored_frame.empty:
        triaged_frame = scored_frame.copy()
        triaged_frame["triage_label"] = pd.Series(dtype="object")
        triaged_frame["triage_percentile"] = pd.Series(dtype="float64")
        return triaged_frame

    if not 0 < medium_percentile < high_percentile < 1:
        raise ValueError(
            "triage percentiles must satisfy 0 < medium_percentile < high_percentile < 1"
        )

    triaged_frame = scored_frame.copy()
    triaged_frame["triage_percentile"] = triaged_frame["anomaly_score"].rank(
        method="average", pct=True
    )
    triaged_frame["triage_label"] = "low"
    triaged_frame.loc[
        triaged_frame["triage_percentile"] >= medium_percentile,
        "triage_label",
    ] = "medium"
    triaged_frame.loc[
        triaged_frame["triage_percentile"] >= high_percentile,
        "triage_label",
    ] = "high"
    return triaged_frame


def _build_triage_explanation(row: pd.Series, *, window_suffix: str) -> str:
    """Summarize the clearest observable signals for one scored grid/day row."""
    triage_label = str(row.get("triage_label", "unknown")).lower()
    anomaly_score = float(row.get("anomaly_score", 0.0))
    triage_percentile = float(row.get("triage_percentile", 0.0))
    total_crimes = float(row.get("total_crimes", 0.0))
    rolling_mean = float(row.get(f"rolling_mean_{window_suffix}", 0.0))
    count_delta = float(row.get("count_delta_from_mean", 0.0))
    count_zscore = float(row.get(f"count_zscore_{window_suffix}", 0.0))

    # Start the explanation with the triage outcome and anomaly score severity
    parts = [
        (
            f"{triage_label.capitalize()} triage: anomaly score {anomaly_score:.3f} "
            f"({triage_percentile:.0%} percentile)."
        )
    ]

    # Compare today's activity to the recent rolling baseline
    if total_crimes or rolling_mean:
        parts.append(
            (
                f"Observed {total_crimes:.0f} crimes versus a recent average of "
                f"{rolling_mean:.1f} ({count_delta:+.1f})."
            )
        )

    # Highlight large deviations, or note when low triage looks stable
    if abs(count_zscore) >= 2:
        direction = "above" if count_zscore > 0 else "below"
        parts.append(
            f"That is {abs(count_zscore):.1f} standard deviations {direction} baseline."
        )
    elif abs(count_delta) <= 1 and triage_label == "low":
        parts.append("Activity is close to the recent baseline.")

    # Add dominant mix signals when a category or time bucket clearly stands out
    dominant_category = _top_share_label(row, _CATEGORY_LABELS)
    if dominant_category is not None:
        parts.append(
            f"Crime mix was led by {dominant_category[0]} ({dominant_category[1]:.0%} of incidents)."
        )

    dominant_time_bucket = _top_share_label(row, _TIME_BUCKET_LABELS)
    if dominant_time_bucket is not None and dominant_time_bucket[1] >= 0.5:
        parts.append(
            f"Incidents were concentrated in the {dominant_time_bucket[0]} ({dominant_time_bucket[1]:.0%})."
        )

    # Surface heavy fallback usage because it affects confidence in the source data
    reported_date_fallback_rate = float(row.get("reported_date_fallback_rate", 0.0))
    reported_hour_fallback_rate = float(row.get("reported_hour_fallback_rate", 0.0))
    if reported_date_fallback_rate >= 0.25:
        parts.append(
            f"{reported_date_fallback_rate:.0%} of rows used reported-date fallback."
        )
    if reported_hour_fallback_rate >= 0.25:
        parts.append(
            f"{reported_hour_fallback_rate:.0%} of rows used reported-hour fallback."
        )

    return " ".join(parts)


def _top_share_label(
    row: pd.Series,
    label_map: dict[str, str],
) -> tuple[str, float] | None:
    """Return the strongest share-based feature label when it is meaningful."""
    best_column = None
    best_value = 0.0
    for column in label_map:
        value = float(row.get(column, 0.0))
        if value > best_value:
            best_column = column
            best_value = value

    if best_column is None or best_value < 0.35:
        return None
    return label_map[best_column], best_value
