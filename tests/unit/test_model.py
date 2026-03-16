from pathlib import Path
import shutil

import pandas as pd
import pytest

from app.model.model import (
    DEFAULT_MODEL_VERSION,
    ModelArtifact,
    infer_feature_columns,
    load_model_artifact,
    prepare_model_matrix,
    save_model_artifact,
    score_feature_frame,
    train_isolation_forest,
)


@pytest.fixture
def sample_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": "2026-01-01",
                "total_crimes": 3.0,
                "rolling_mean_30d": 1.5,
                "reported_hour_fallback_rate": 0.0,
                "category_assaults_share": 0.67,
            },
            {
                "grid_id": "g1",
                "date": "2026-01-02",
                "total_crimes": 4.0,
                "rolling_mean_30d": 2.0,
                "reported_hour_fallback_rate": 0.25,
                "category_assaults_share": 0.50,
            },
            {
                "grid_id": "g2",
                "date": "2026-01-01",
                "total_crimes": 1.0,
                "rolling_mean_30d": 0.5,
                "reported_hour_fallback_rate": 0.0,
                "category_assaults_share": 0.00,
            },
            {
                "grid_id": "g2",
                "date": "2026-01-02",
                "total_crimes": 7.0,
                "rolling_mean_30d": 2.5,
                "reported_hour_fallback_rate": 0.50,
                "category_assaults_share": 0.85,
            },
        ]
    )


def test_infer_feature_columns_excludes_metadata_and_keeps_numeric_columns(
    sample_feature_frame: pd.DataFrame,
) -> None:
    """Test that feature inference excludes metadata and keeps numeric columns."""
    feature_columns = infer_feature_columns(sample_feature_frame)

    assert feature_columns == [
        "total_crimes",
        "rolling_mean_30d",
        "reported_hour_fallback_rate",
        "category_assaults_share",
    ]
    assert "grid_id" not in feature_columns
    assert "date" not in feature_columns


def test_prepare_model_matrix_aligns_and_coerces_numeric_values() -> None:
    """Test that model matrix preparation coerces values to floats and fills gaps."""
    # Dummy frame
    feature_frame = pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "date": "2026-01-01",
                "total_crimes": "3",
                "rolling_mean_30d": None,
                "reported_hour_fallback_rate": "0.25",
            }
        ]
    )

    # Use the function to prepare the model matrix, specifying the expected feature columns
    matrix = prepare_model_matrix(
        feature_frame,
        ["total_crimes", "rolling_mean_30d", "reported_hour_fallback_rate"],
    )

    # Verify that the resulting matrix has the correct columns, data types, and values
    assert matrix.dtypes.tolist() == [float, float, float]
    assert matrix.iloc[0].to_dict() == {
        "total_crimes": 3.0,
        "rolling_mean_30d": 0.0,
        "reported_hour_fallback_rate": 0.25,
    }


def test_prepare_model_matrix_raises_for_missing_columns(
    sample_feature_frame: pd.DataFrame,
) -> None:
    """Test that model matrix preparation raises for missing feature columns."""
    with pytest.raises(ValueError):
        prepare_model_matrix(sample_feature_frame, ["total_crimes", "missing_feature"])


def test_train_isolation_forest_returns_artifact_with_schema(
    sample_feature_frame: pd.DataFrame,
) -> None:
    """Test that Isolation Forest training returns an artifact with schema metadata."""
    artifact = train_isolation_forest(
        sample_feature_frame,
        n_estimators=25,
        random_state=7,
    )

    assert isinstance(artifact, ModelArtifact)
    assert artifact.model_version == DEFAULT_MODEL_VERSION
    assert artifact.training_row_count == 4
    assert artifact.feature_columns == [
        "total_crimes",
        "rolling_mean_30d",
        "reported_hour_fallback_rate",
        "category_assaults_share",
    ]
    assert artifact.model_params == {
        "contamination": "auto",
        "n_estimators": 25,
        "random_state": 7,
    }


def test_save_and_load_model_artifact_round_trip(
    sample_feature_frame: pd.DataFrame,
) -> None:
    """Test that saving and loading a model artifact preserves its metadata."""
    artifact = train_isolation_forest(sample_feature_frame, n_estimators=10)
    output_dir = Path("tests/.artifacts")
    output_path = output_dir / "crime_model.joblib"

    # Use a workspace-local path to avoid pytest temp-dir permission issues.
    if output_dir.exists():
        shutil.rmtree(output_dir)

    saved_path = save_model_artifact(artifact, output_path)
    loaded_artifact = load_model_artifact(saved_path)

    assert saved_path == output_path
    assert saved_path.exists()
    assert loaded_artifact.model_version == artifact.model_version
    assert loaded_artifact.feature_columns == artifact.feature_columns
    assert loaded_artifact.training_row_count == artifact.training_row_count

    shutil.rmtree(output_dir)


def test_score_feature_frame_returns_scores_with_metadata(
    sample_feature_frame: pd.DataFrame,
) -> None:
    """Test that scoring returns anomaly scores alongside metadata columns."""
    artifact = train_isolation_forest(sample_feature_frame, n_estimators=20)

    scored_frame = score_feature_frame(sample_feature_frame, artifact)

    assert list(scored_frame.columns) == [
        "grid_id",
        "date",
        "anomaly_score",
        "model_version",
    ]
    assert len(scored_frame) == len(sample_feature_frame)
    assert (scored_frame["anomaly_score"] >= 0).all()
    assert (scored_frame["model_version"] == artifact.model_version).all()
