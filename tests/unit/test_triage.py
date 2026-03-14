from datetime import date

import pandas as pd

from app import triage


def test_assign_triage_labels_adds_percentiles_and_labels() -> None:
    """Test that triage assignment adds both percentile metadata and severity labels."""
    scored_frame = pd.DataFrame(
        [
            {"grid_id": "g1", "date": date(2026, 3, 13), "anomaly_score": 0.1},
            {"grid_id": "g2", "date": date(2026, 3, 13), "anomaly_score": 0.5},
            {"grid_id": "g3", "date": date(2026, 3, 13), "anomaly_score": 0.9},
        ]
    )

    triaged_frame = triage.assign_triage_labels(
        scored_frame,
        medium_percentile=0.5,
        high_percentile=0.8,
    )

    assert triaged_frame["triage_label"].tolist() == ["low", "medium", "high"]
    assert "triage_percentile" in triaged_frame.columns


def test_add_triage_explanations_summarizes_feature_signals() -> None:
    """Test that explanation text reflects baseline, mix, and timing signals."""
    triaged_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "anomaly_score": 0.9,
                "triage_percentile": 0.95,
                "triage_label": "high",
            }
        ]
    )
    feature_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "total_crimes": 8.0,
                "rolling_mean_30d": 3.0,
                "count_delta_from_mean": 5.0,
                "count_zscore_30d": 2.6,
                "reported_date_fallback_rate": 0.0,
                "reported_hour_fallback_rate": 0.0,
                "category_assaults_share": 0.625,
                "evening_crimes_share": 0.75,
            }
        ]
    )

    explained_frame = triage.add_triage_explanations(
        triaged_frame,
        feature_frame,
        lookback_days=30,
    )

    explanation = explained_frame.loc[0, "triage_explanation"]
    assert "95% percentile" in explanation
    assert "Observed 8 crimes versus a recent average of 3.0" in explanation
    assert "2.6 standard deviations above baseline" in explanation
    assert "assaults (62% of incidents)" in explanation
    assert "evening (75%)" in explanation
