import pandas as pd

DEFAULT_TRIAGE_HIGH_PERCENTILE = 0.9
DEFAULT_TRIAGE_MEDIUM_PERCENTILE = 0.75


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
