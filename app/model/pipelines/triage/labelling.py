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
    "night_crimes": "night",
    "morning_crimes": "morning",
    "afternoon_crimes": "afternoon",
    "evening_crimes": "evening",
    "unknown_hour_crimes": "unknown-hour",
}
_CATEGORY_COUNT_LABELS = {
    "category_arson": "arson",
    "category_assaults": "assaults",
    "category_attempted_murder": "attempted murder",
    "category_break_and_enter": "break and enter",
    "category_criminal_harassment": "criminal harassment",
    "category_homicide": "homicide",
    "category_indecent_or_harassing_communications": "indecent or harassing communications",
    "category_mischief": "mischief",
    "category_robbery": "robbery",
    "category_theft_5000_and_under": "theft $5000 and under",
    "category_theft_over_5000": "theft over $5000",
    "category_theft_of_motor_vehicle": "theft of motor vehicle",
    "category_uttering_threats": "uttering threats",
}


def add_triage_explanations(
    triaged_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Attach a human-readable explanation for each row's triage outcome.

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
        lambda row: _build_triage_explanation(
            row,
            window_suffix=window_suffix,
        ),
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


def _build_triage_explanation(
    row: pd.Series,
    *,
    window_suffix: str,
) -> str:
    """Summarize the meaningful observable signals for one scored grid/day row."""

    triage_label = str(row.get("triage_label", "unknown")).lower()
    # anomaly_score = float(row.get("anomaly_score", 0.0))
    # triage_percentile = float(row.get("triage_percentile", 0.0))
    total_crimes = float(row.get("total_crimes", 0.0))
    rolling_mean = float(row.get(f"rolling_mean_{window_suffix}", 0.0))
    count_delta = float(row.get("count_delta_from_mean", 0.0))
    count_zscore = float(row.get(f"count_zscore_{window_suffix}", 0.0))
    category_shifts = _category_shifts(row, window_suffix=window_suffix)
    time_bucket_shifts = _time_bucket_shifts(row, window_suffix=window_suffix)
    is_close_to_baseline = abs(count_zscore) < 1.0 and abs(count_delta) <= max(
        1.0, rolling_mean * 0.25
    )
    
    # List holding multi-part explanation components, which will be joined together at the end. Each part should be a complete sentence or clause.
    parts: list[str] = []

    # Percentile
    # percentile_as_rank = max(1, int(round(triage_percentile * 100)))
    # parts = [
    #     (
    #         f"{triage_label.capitalize()} triage: anomaly score {anomaly_score:.3f}. "
    #         f"This is higher than about {percentile_as_rank}% of scored locations."
    #     )
    # ]

    # Crime volume delta
    if total_crimes or rolling_mean:
        parts.append(
            (
                f"Observed {total_crimes:.0f} crimes versus a recent average of "
                f"{rolling_mean:.1f}, which is {_format_delta(count_delta)}."
            )
        )

    # Overall signal strength
    overall_shift_level = _shift_strength_level(
        count_zscore,
        count_delta,
        rolling_mean,
    )
    if overall_shift_level > 0:
        direction_text = "above" if count_delta > 0 else "below"
        parts.append(
            f"Overall crime volume is {_describe_shift_strength(overall_shift_level)} {direction_text} usual for this area."
        )
    elif abs(count_delta) <= 1 and triage_label == "low":
        parts.append("Activity is close to the usual level for this area.")

    # Category and time-bucket signals
    for (
        category_label,
        direction,
        category_count,
        category_mean,
        category_zscore,
    ) in category_shifts[:3]:
        parts.append(
            (
                f"{category_label.capitalize()} was {category_count:.0f} compared with the usual "
                f"{category_mean:.1f}, a {_describe_directional_shift(direction, category_zscore, category_count - category_mean, category_mean)}."
            )
        )

    for (
        bucket_label,
        direction,
        bucket_count,
        bucket_mean,
        bucket_zscore,
    ) in time_bucket_shifts[:2]:
        parts.append(
            (
                f"{bucket_label.capitalize()} incidents were {bucket_count:.0f} compared with the usual "
                f"{bucket_mean:.1f}, a {_describe_directional_shift(direction, bucket_zscore, bucket_count - bucket_mean, bucket_mean)}."
            )
        )

    # Fallback explanations
    if len(parts) == 1:
        if triage_label == "low" and is_close_to_baseline:
            parts.append("No unusual activity.")
        else:
            parts.append(
                "There were mild changes, but no single clear driver stood out."
            )

    return " ".join(parts)


def _category_shifts(
    row: pd.Series,
    *,
    window_suffix: str,
) -> list[tuple[str, str, float, float, float]]:
    """
    Return meaningful category-level changes versus rolling baselines.
    """
    shifts: list[tuple[float, str, str, float, float, float]] = []

    for column, label in _CATEGORY_COUNT_LABELS.items():
        category_count = float(row.get(column, 0.0))
        category_mean = float(row.get(f"{column}_rolling_mean_{window_suffix}", 0.0))
        category_delta = float(row.get(f"{column}_delta_from_mean", 0.0))
        category_zscore = float(row.get(f"{column}_zscore_{window_suffix}", 0.0))

        strength = abs(category_zscore)
        if strength < 1.5 and abs(category_delta) < max(2.0, category_mean * 0.75):
            continue
        shifts.append(
            (
                strength,
                label,
                "up" if category_delta >= 0 else "down",
                category_count,
                category_mean,
                category_zscore,
            )
        )

    shifts.sort(key=lambda item: item[0], reverse=True)
    return [shift[1:] for shift in shifts]


def _format_delta(delta: float) -> str:
    """Render a plain-language difference from the recent average."""
    if delta > 0:
        return f"{abs(delta):.1f} more than usual"
    if delta < 0:
        return f"{abs(delta):.1f} fewer than usual"
    return "right in line with the recent average"


def _describe_shift_strength(level: int) -> str:
    """Translate a severity bucket into plainer overall wording."""
    if level >= 2:
        return "far"
    if level == 1:
        return "well"
    return "slightly"


def _relative_change(delta: float, baseline: float) -> float:
    """Return the proportional change versus baseline while handling zero baselines."""
    if baseline <= 0:
        return float("inf") if abs(delta) > 0 else 0.0
    return abs(delta) / baseline


def _shift_strength_level(zscore: float, delta: float, baseline: float) -> int:
    """
    Blend statistical distance with absolute and proportional change.

    This prevents complete drop-to-zero or spike-from-baseline explanations from
    being understated in noisier areas, while staying conservative for tiny
    baselines.
    """
    strength = abs(zscore)
    abs_delta = abs(delta)
    relative_change = _relative_change(delta, baseline)

    if (
        strength >= 2.5
        or abs_delta >= max(5.0, baseline)
        or (baseline >= 2.0 and relative_change >= 0.9 and abs_delta >= 2.0)
    ):
        return 2
    if (
        strength >= 1.5
        or abs_delta >= max(3.0, baseline * 0.75)
        or (baseline >= 1.5 and relative_change >= 0.6 and abs_delta >= 1.5)
    ):
        return 1
    return 0


def _describe_directional_shift(
    direction: str,
    zscore: float,
    delta: float,
    baseline: float,
) -> str:
    """Describe an up/down shift without exposing statistical jargon."""
    level = _shift_strength_level(zscore, delta, baseline)
    if level >= 2:
        modifier = "sharp"
    elif level == 1:
        modifier = "clear"
    else:
        modifier = "small"
    noun = "increase" if direction == "up" else "drop"
    return f"{modifier} {noun}"


def _time_bucket_shifts(
    row: pd.Series,
    *,
    window_suffix: str,
) -> list[tuple[str, str, float, float, float]]:
    """
    Return meaningful time-bucket count shifts versus rolling baselines.
    """
    shifts: list[tuple[float, str, str, float, float, float]] = []

    for column, label in _TIME_BUCKET_LABELS.items():
        bucket_count = float(row.get(column, 0.0))
        bucket_mean = float(row.get(f"{column}_rolling_mean_{window_suffix}", 0.0))
        bucket_delta = float(row.get(f"{column}_delta_from_mean", 0.0))
        bucket_zscore = float(row.get(f"{column}_zscore_{window_suffix}", 0.0))

        strength = abs(bucket_zscore)
        if strength < 1.5 and abs(bucket_delta) < max(1.0, bucket_mean * 0.75):
            continue
        shifts.append(
            (
                strength,
                label,
                "up" if bucket_delta >= 0 else "down",
                bucket_count,
                bucket_mean,
                bucket_zscore,
            )
        )

    shifts.sort(key=lambda item: item[0], reverse=True)
    return [shift[1:] for shift in shifts]
