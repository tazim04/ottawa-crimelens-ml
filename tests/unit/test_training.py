from pathlib import Path

import pandas as pd
import pytest

from app import training


def test_resolve_model_artifact_path_uses_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that training artifact path resolution falls back to the environment."""
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", "models/training-from-env.joblib")

    assert training.resolve_model_artifact_path() == Path(
        "models/training-from-env.joblib"
    )


def test_parse_contamination_normalizes_cli_inputs() -> None:
    """Test that contamination parsing preserves ``auto`` and converts numeric strings."""
    assert training.parse_contamination("auto") == "auto"
    assert training.parse_contamination("0.15") == 0.15
    assert training.parse_contamination(0.2) == 0.2


def test_run_training_pipeline_builds_trains_and_saves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the training pipeline orchestrates feature building, fitting, and saving."""
    training_frame = pd.DataFrame(
        [
            {"grid_id": "g1", "date": "2026-01-01", "total_crimes": 3.0},
            {"grid_id": "g2", "date": "2026-01-01", "total_crimes": 6.0},
        ]
    )
    artifact_sentinel = object()

    monkeypatch.setattr(
        training,
        "build_training_dataset",
        lambda **kwargs: training_frame,
    )
    monkeypatch.setattr(
        training,
        "train_isolation_forest",
        lambda feature_frame, **kwargs: (
            artifact_sentinel if feature_frame.equals(training_frame) else None
        ),
    )
    monkeypatch.setattr(
        training,
        "save_model_artifact",
        lambda artifact, output_path: (
            Path(output_path) if artifact is artifact_sentinel else Path("unexpected")
        ),
    )

    artifact, saved_path = training.run_training_pipeline(
        start_date="2026-01-01",
        end_date="2026-01-31",
        contamination="0.05",
        output_path="artifacts/custom.joblib",
    )

    assert artifact is artifact_sentinel
    assert saved_path == Path("artifacts/custom.joblib")


def test_run_training_pipeline_raises_when_training_dataset_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that training fails fast when the requested date range yields no rows."""
    monkeypatch.setattr(
        training,
        "build_training_dataset",
        lambda **kwargs: pd.DataFrame(),
    )

    with pytest.raises(ValueError):
        training.run_training_pipeline(
            start_date="2026-01-01",
            end_date="2026-01-31",
        )
