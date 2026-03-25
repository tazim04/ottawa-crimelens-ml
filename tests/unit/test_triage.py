from datetime import date

import pandas as pd

from app.model.pipelines.triage import labelling


def test_assign_triage_labels_adds_percentiles_and_labels() -> None:
    """Test that triage assignment adds both percentile metadata and severity labels."""
    scored_frame = pd.DataFrame(
        [
            {"grid_id": "g1", "date": date(2026, 3, 13), "anomaly_score": 0.1},
            {"grid_id": "g2", "date": date(2026, 3, 13), "anomaly_score": 0.5},
            {"grid_id": "g3", "date": date(2026, 3, 13), "anomaly_score": 0.9},
        ]
    )

    triaged_frame = labelling.assign_triage_labels(
        scored_frame,
        medium_percentile=0.5,
        high_percentile=0.8,
    )

    assert triaged_frame["triage_label"].tolist() == ["low", "medium", "high"]
    assert "triage_percentile" in triaged_frame.columns


def test_add_triage_explanations_summarizes_feature_signals() -> None:
    """Test that explanation text reflects baseline, category, and timing signals."""
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
                "category_assaults": 5.0,
                "category_assaults_rolling_mean_30d": 1.5,
                "category_assaults_delta_from_mean": 3.5,
                "category_assaults_zscore_30d": 2.8,
                "evening_crimes": 6.0,
                "evening_crimes_rolling_mean_30d": 2.0,
                "evening_crimes_delta_from_mean": 4.0,
                "evening_crimes_zscore_30d": 2.1,
            }
        ]
    )

    explained_frame = labelling.add_triage_explanations(
        triaged_frame,
        feature_frame,
        lookback_days=30,
    )

    explanation = explained_frame.loc[0, "triage_explanation"]
    assert explanation.startswith(
        "High triage: anomaly score 0.900. This is higher than about 95% of scored locations."
    )
    assert (
        "Observed 8 crimes versus a recent average of 3.0, which is 5.0 more than usual."
        in explanation
    )
    assert "Overall crime volume is far above usual for this area." in explanation
    assert (
        "Assaults was 5 compared with the usual 1.5, a sharp increase." in explanation
    )
    assert (
        "Evening incidents were 6 compared with the usual 2.0, a sharp increase."
        in explanation
    )


def test_add_triage_explanations_surfaces_large_drop_in_activity() -> None:
    """Test that a large drop describes the below-baseline change in the explanation."""
    triaged_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "anomaly_score": 0.7,
                "triage_percentile": 1.0,
                "triage_label": "high",
            }
        ]
    )
    feature_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "total_crimes": 0.0,
                "rolling_mean_30d": 11.4,
                "count_delta_from_mean": -11.4,
                "count_zscore_30d": -2.8,
                "reported_date_fallback_rate": 0.0,
                "reported_hour_fallback_rate": 0.0,
                "category_assaults": 0.0,
                "category_assaults_rolling_mean_30d": 2.0,
                "category_assaults_delta_from_mean": -2.0,
                "category_assaults_zscore_30d": -2.2,
            }
        ]
    )

    explained_frame = labelling.add_triage_explanations(
        triaged_frame,
        feature_frame,
        lookback_days=30,
    )

    explanation = explained_frame.loc[0, "triage_explanation"]
    assert explanation.startswith(
        "High triage: anomaly score 0.700. This is higher than about 100% of scored locations."
    )
    assert (
        "Observed 0 crimes versus a recent average of 11.4, which is 11.4 fewer than usual."
        in explanation
    )
    assert "Overall crime volume is far below usual for this area." in explanation
    assert "Assaults was 0 compared with the usual 2.0, a sharp drop." in explanation


def test_add_triage_explanations_uses_proportional_drop_language_for_high_triage() -> (
    None
):
    """Test that high-triage rows with a full drop do not understate the shift."""
    triaged_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "anomaly_score": 0.662,
                "triage_percentile": 1.0,
                "triage_label": "high",
            }
        ]
    )
    feature_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "total_crimes": 0.0,
                "rolling_mean_30d": 6.2,
                "count_delta_from_mean": -6.2,
                "count_zscore_30d": -1.1,
                "reported_date_fallback_rate": 0.0,
                "reported_hour_fallback_rate": 0.0,
                "category_theft_5000_and_under": 0.0,
                "category_theft_5000_and_under_rolling_mean_30d": 4.2,
                "category_theft_5000_and_under_delta_from_mean": -4.2,
                "category_theft_5000_and_under_zscore_30d": -1.0,
                "afternoon_crimes": 0.0,
                "afternoon_crimes_rolling_mean_30d": 3.2,
                "afternoon_crimes_delta_from_mean": -3.2,
                "afternoon_crimes_zscore_30d": -1.0,
                "evening_crimes": 0.0,
                "evening_crimes_rolling_mean_30d": 1.7,
                "evening_crimes_delta_from_mean": -1.7,
                "evening_crimes_zscore_30d": -0.9,
            }
        ]
    )

    explained_frame = labelling.add_triage_explanations(
        triaged_frame,
        feature_frame,
        lookback_days=30,
    )

    explanation = explained_frame.loc[0, "triage_explanation"]
    assert "Overall crime volume is far below usual for this area." in explanation
    assert (
        "Theft $5000 and under was 0 compared with the usual 4.2, a sharp drop."
        in explanation
    )
    assert (
        "Afternoon incidents were 0 compared with the usual 3.2, a sharp drop."
        in explanation
    )
    assert (
        "Evening incidents were 0 compared with the usual 1.7, a clear drop."
        in explanation
    )


def test_add_triage_explanations_uses_close_to_baseline_wording_for_low_rows() -> None:
    """Test that low-triage rows near baseline stay conservative in the explanation."""
    triaged_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "anomaly_score": 0.2,
                "triage_percentile": 0.2,
                "triage_label": "low",
            }
        ]
    )
    feature_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "total_crimes": 0.0,
                "rolling_mean_30d": 0.8,
                "count_delta_from_mean": -0.8,
                "count_zscore_30d": -0.4,
                "reported_date_fallback_rate": 0.0,
                "reported_hour_fallback_rate": 0.0,
            }
        ]
    )

    explained_frame = labelling.add_triage_explanations(
        triaged_frame,
        feature_frame,
        lookback_days=30,
    )

    explanation = explained_frame.loc[0, "triage_explanation"]
    assert explanation.startswith(
        "Low triage: anomaly score 0.200. This is higher than about 20% of scored locations."
    )
    assert explanation.endswith("Activity is close to the usual level for this area.")


def test_add_triage_explanations_keeps_high_triage_rows_out_of_low_baseline_wording() -> (
    None
):
    """Test that high-triage rows do not reuse the low-severity baseline sentence."""
    triaged_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "anomaly_score": 0.61,
                "triage_percentile": 0.94,
                "triage_label": "high",
            }
        ]
    )
    feature_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "total_crimes": 0.0,
                "rolling_mean_30d": 1.3,
                "count_delta_from_mean": -1.3,
                "count_zscore_30d": -0.9,
                "reported_date_fallback_rate": 0.0,
                "reported_hour_fallback_rate": 0.0,
            }
        ]
    )

    explained_frame = labelling.add_triage_explanations(
        triaged_frame,
        feature_frame,
        lookback_days=30,
    )

    explanation = explained_frame.loc[0, "triage_explanation"]
    assert explanation == (
        "High triage: anomaly score 0.610. This is higher than about 94% of scored locations. "
        "Observed 0 crimes versus a recent average of 1.3, which is 1.3 fewer than usual."
    )


def test_add_triage_explanations_keeps_tiny_baseline_high_triage_rows_minimal() -> None:
    """Test that a high-triage row with a tiny baseline does not invent extra drivers."""
    triaged_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "anomaly_score": 0.561,
                "triage_percentile": 0.9027777777777778,
                "triage_label": "high",
            }
        ]
    )
    feature_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "total_crimes": 0.0,
                "rolling_mean_30d": 0.6,
                "count_delta_from_mean": -0.6,
                "count_zscore_30d": -0.9,
                "reported_date_fallback_rate": 0.0,
                "reported_hour_fallback_rate": 0.0,
            }
        ]
    )

    explained_frame = labelling.add_triage_explanations(
        triaged_frame,
        feature_frame,
        lookback_days=30,
    )

    explanation = explained_frame.loc[0, "triage_explanation"]
    assert explanation == (
        "High triage: anomaly score 0.561. This is higher than about 90% of scored locations. "
        "Observed 0 crimes versus a recent average of 0.6, which is 0.6 fewer than usual."
    )
