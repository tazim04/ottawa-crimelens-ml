from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from app import scoring


def test_resolve_scoring_date_normalizes_supported_inputs() -> None:
    """Test that supported date-like inputs normalize to a plain ``date``."""
    assert scoring.resolve_scoring_date("2026-03-13") == date(2026, 3, 13)
    assert scoring.resolve_scoring_date(date(2026, 3, 13)) == date(2026, 3, 13)
    assert scoring.resolve_scoring_date(datetime(2026, 3, 13, 9, 30)) == date(
        2026, 3, 13
    )


def test_resolve_model_artifact_path_uses_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the model artifact path falls back to the environment variable."""
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", "models/from-env.joblib")

    assert scoring.resolve_model_artifact_path() == Path("models/from-env.joblib")


def test_score_daily_features_uses_builder_and_model_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that daily scoring orchestrates feature building, model loading, and triage."""
    feature_frame = pd.DataFrame(
        [
            {"grid_id": "g1", "date": date(2026, 3, 13), "total_crimes": 2.0},
            {"grid_id": "g2", "date": date(2026, 3, 13), "total_crimes": 7.0},
        ]
    )
    scored_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "anomaly_score": 0.1,
                "model_version": "v1",
            },
            {
                "grid_id": "g2",
                "date": date(2026, 3, 13),
                "anomaly_score": 0.9,
                "model_version": "v1",
            },
        ]
    )
    artifact_sentinel = object()

    # Stub each dependency so this test only verifies scoring-layer orchestration.
    monkeypatch.setattr(
        scoring, "build_scoring_features", lambda **kwargs: feature_frame
    )
    monkeypatch.setattr(scoring, "load_model_artifact", lambda path: artifact_sentinel)
    monkeypatch.setattr(
        scoring,
        "score_feature_frame",
        lambda features, artifact: (
            scored_frame
            if features.equals(feature_frame) and artifact is artifact_sentinel
            else pd.DataFrame()
        ),
    )

    result = scoring.score_daily_features(
        target_date="2026-03-13",
        model_artifact_path="models/current.joblib",
        medium_percentile=0.5,
        high_percentile=0.8,
    )

    # The lower of two rows lands on the medium percentile threshold here.
    assert result["triage_label"].tolist() == ["medium", "high"]
    assert result["model_version"].tolist() == ["v1", "v1"]


def test_run_scoring_pipeline_persists_results_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the top-level pipeline persists scored rows when persistence is enabled."""
    scored_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": date(2026, 3, 13),
                "anomaly_score": 0.42,
                "model_version": "v1",
                "triage_percentile": 1.0,
                "triage_label": "high",
            }
        ]
    )
    persisted: dict[str, object] = {}

    # Capture persistence inputs so the test can assert the pipeline forwards them correctly.
    monkeypatch.setattr(scoring, "score_daily_features", lambda **kwargs: scored_frame)
    monkeypatch.setattr(
        scoring,
        "persist_scored_results",
        lambda frame, table_name, if_exists: (
            persisted.update(
                {
                    "frame": frame.copy(),
                    "table_name": table_name,
                    "if_exists": if_exists,
                }
            )
            or len(frame)
        ),
    )

    result = scoring.run_scoring_pipeline(
        target_date="2026-03-13",
        persist_results=True,
        results_table="daily_scores",
        if_exists="replace",
    )

    assert result.equals(scored_frame)
    assert persisted["table_name"] == "daily_scores"
    assert persisted["if_exists"] == "replace"
