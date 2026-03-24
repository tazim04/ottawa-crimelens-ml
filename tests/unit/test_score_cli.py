import argparse

import pandas as pd
import pytest

import score


def test_main_allows_model_artifact_path_to_fall_back_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the scoring CLI does not require the model path flag when env/default fallback is available."""

    class _ParserStub:
        def parse_args(self) -> argparse.Namespace:
            return parsed_args

    parsed_args = argparse.Namespace(
        target_date="2026-03-13",
        model_artifact_path=None,
        lookback_days=None,
        min_history_days=None,
        persist_results=False,
        results_table="crime_anomaly_scores",
        if_exists="delete_rows",
    )
    captured_kwargs: dict[str, object] = {}

    monkeypatch.setattr(score, "configure_logging", lambda: None)
    monkeypatch.setattr(score, "build_parser", lambda: _ParserStub())
    monkeypatch.setattr(
        score,
        "run_scoring_pipeline",
        lambda **kwargs: captured_kwargs.update(kwargs) or pd.DataFrame(),
    )

    exit_code = score.main()

    assert exit_code == 0
    assert captured_kwargs["model_artifact_path"] is None
