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
