from datetime import date
from typing import cast

import numpy as np
import pandas as pd
import pytest

from app import feature_builder
from app.features.aggregation import compute_features, prepare_daily_frame


def _sample_crime_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "grid_id": "g1",
                "event_date": "2026-01-01",
                "event_hour": 100,
                "offence_category": "Assaults",
                "used_reported_date_fallback": 1,
                "used_reported_hour_fallback": 0,
            },
            {
                "grid_id": "g1",
                "event_date": "2026-01-01",
                "event_hour": 900,
                "offence_category": "Assaults",
                "used_reported_date_fallback": 0,
                "used_reported_hour_fallback": 1,
            },
            {
                "grid_id": "g1",
                "event_date": "2026-01-01",
                "event_hour": 1300,
                "offence_category": "Theft $5000 and Under",
                "used_reported_date_fallback": 0,
                "used_reported_hour_fallback": 0,
            },
            {
                "grid_id": "g1",
                "event_date": "2026-01-03",
                "event_hour": 700,
                "offence_category": "Assaults",
                "used_reported_date_fallback": 0,
                "used_reported_hour_fallback": 1,
            },
            {
                "grid_id": "g1",
                "event_date": "2026-01-03",
                "event_hour": 1100,
                "offence_category": "Theft $5000 and Under",
                "used_reported_date_fallback": 0,
                "used_reported_hour_fallback": 1,
            },
            {
                "grid_id": "g1",
                "event_date": "2026-01-03",
                "event_hour": 1500,
                "offence_category": "Theft $5000 and Under",
                "used_reported_date_fallback": 0,
                "used_reported_hour_fallback": 0,
            },
            {
                "grid_id": "g1",
                "event_date": "2026-01-03",
                "event_hour": 2200,
                "offence_category": "Robbery",
                "used_reported_date_fallback": 0,
                "used_reported_hour_fallback": 0,
            },
            {
                "grid_id": "g2",
                "event_date": "2026-01-01",
                "event_hour": 100,
                "offence_category": "Theft $5000 and Under",
                "used_reported_date_fallback": 0,
                "used_reported_hour_fallback": 0,
            },
            {
                "grid_id": "g2",
                "event_date": "2026-01-03",
                "event_hour": 100,
                "offence_category": "Theft $5000 and Under",
                "used_reported_date_fallback": 0,
                "used_reported_hour_fallback": 1,
            },
            {
                "grid_id": "g2",
                "event_date": "2026-01-03",
                "event_hour": None,
                "offence_category": "Mischief",
                "used_reported_date_fallback": 0,
                "used_reported_hour_fallback": 0,
            },
        ]
    )


@pytest.fixture
def sample_crime_records() -> pd.DataFrame:
    return _sample_crime_records()


@pytest.fixture
def prepared_daily_frame(
    sample_crime_records: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    return prepare_daily_frame(
        crime_records=sample_crime_records,
        full_date_range=pd.date_range("2026-01-01", "2026-01-03", freq="D"),
    )


def test_prepare_daily_frame_aggregates_raw_records_to_daily_grid_rows(
    prepared_daily_frame: tuple[pd.DataFrame, list[str]],
) -> None:
    frame, category_columns = prepared_daily_frame

    assert set(category_columns) >= {
        "category_assaults",
        "category_mischief",
        "category_robbery",
    }
    assert "category_theft_5000_and_under" in category_columns
    assert len(frame) == 6

    g1_day1 = frame[
        (frame["grid_id"] == "g1") & (frame["date"] == pd.Timestamp("2026-01-01"))
    ].iloc[0]
    assert g1_day1["total_crimes"] == 3.0
    assert g1_day1["night_crimes"] == 1.0
    assert g1_day1["morning_crimes"] == 1.0
    assert g1_day1["afternoon_crimes"] == 1.0
    assert g1_day1["category_assaults"] == 2.0
    assert g1_day1["category_theft_5000_and_under"] == 1.0
    assert g1_day1["used_reported_date_fallback_count"] == 1.0
    assert g1_day1["used_reported_hour_fallback_count"] == 1.0

    g2_missing_day = frame[
        (frame["grid_id"] == "g2") & (frame["date"] == pd.Timestamp("2026-01-02"))
    ].iloc[0]
    assert g2_missing_day["total_crimes"] == 0.0
    assert g2_missing_day["has_source_row"] == 0.0


def test_compute_features_builds_rolling_and_share_features(
    prepared_daily_frame: tuple[pd.DataFrame, list[str]],
) -> None:
    daily_frame, category_columns = prepared_daily_frame
    features = compute_features(
        daily_frame=daily_frame,
        category_columns=category_columns,
        lookback_days=2,
    )

    g1_day3 = features[
        (features["grid_id"] == "g1") & (features["date"] == pd.Timestamp("2026-01-03"))
    ].iloc[0]
    assert g1_day3["history_days"] == 1
    assert g1_day3["rolling_mean_2d"] == pytest.approx(1.5)
    assert g1_day3["rolling_sum_2d"] == pytest.approx(3.0)
    assert g1_day3["category_assaults_share"] == pytest.approx(0.25)
    assert g1_day3["category_theft_5000_and_under_share"] == pytest.approx(0.5)
    assert g1_day3["category_robbery_share"] == pytest.approx(0.25)
    assert g1_day3["morning_crimes_share"] == pytest.approx(0.5)
    assert g1_day3["afternoon_crimes_share"] == pytest.approx(0.25)
    assert g1_day3["evening_crimes_share"] == pytest.approx(0.25)
    assert g1_day3["reported_hour_fallback_rate"] == pytest.approx(0.5)
    assert np.isclose(g1_day3["count_zscore_2d"], 1.1785113019775793)

    g2_day3 = features[
        (features["grid_id"] == "g2") & (features["date"] == pd.Timestamp("2026-01-03"))
    ].iloc[0]
    assert g2_day3["unknown_hour_crimes_share"] == pytest.approx(0.5)
    assert g2_day3["reported_hour_fallback_rate"] == pytest.approx(0.5)


def test_compute_features_handles_zero_total_crimes_without_nan(
    prepared_daily_frame: tuple[pd.DataFrame, list[str]],
) -> None:
    daily_frame, category_columns = prepared_daily_frame
    features = compute_features(
        daily_frame=daily_frame,
        category_columns=category_columns,
        lookback_days=2,
    )

    g1_day2 = features[
        (features["grid_id"] == "g1") & (features["date"] == pd.Timestamp("2026-01-02"))
    ].iloc[0]
    assert g1_day2["total_crimes"] == 0.0
    assert g1_day2["category_assaults_share"] == 0.0
    assert g1_day2["category_theft_5000_and_under_share"] == 0.0
    assert g1_day2["unknown_hour_crimes_share"] == 0.0
    assert g1_day2["reported_date_fallback_rate"] == 0.0


@pytest.mark.parametrize(
    ("lookback_days", "min_history_days"),
    [
        (1, 1),
        (3, 4),
    ],
)
def test_build_feature_frame_validates_parameters(
    lookback_days: int,
    min_history_days: int,
) -> None:
    with pytest.raises(ValueError):
        feature_builder._build_feature_frame(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 2),
            lookback_days=lookback_days,
            min_history_days=min_history_days,
        )


def test_build_daily_features_filters_to_target_date(
    monkeypatch: pytest.MonkeyPatch,
    sample_crime_records: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        feature_builder,
        "_fetch_crime_records",
        lambda start_date, end_date: sample_crime_records,
    )

    features = feature_builder.build_daily_features(
        target_date="2026-01-03",
        lookback_days=2,
        min_history_days=1,
    )

    assert sorted(features["date"].unique()) == [date(2026, 1, 3)]
    assert set(features["grid_id"]) == {"g1", "g2"}


def test_build_training_features_filters_by_history_and_date_range(
    monkeypatch: pytest.MonkeyPatch,
    sample_crime_records: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        feature_builder,
        "_fetch_crime_records",
        lambda start_date, end_date: sample_crime_records,
    )

    features = feature_builder.build_training_features(
        start_date="2026-01-01",
        end_date="2026-01-03",
        lookback_days=2,
        min_history_days=1,
    )

    assert features["date"].tolist() == [
        date(2026, 1, 2),
        date(2026, 1, 3),
        date(2026, 1, 2),
        date(2026, 1, 3),
    ]
    assert features["grid_id"].tolist() == ["g1", "g1", "g2", "g2"]


def test_build_training_features_rejects_inverted_dates() -> None:
    with pytest.raises(ValueError):
        feature_builder.build_training_features(
            start_date="2026-01-03",
            end_date="2026-01-01",
        )


def test_fetch_crime_records_uses_guarded_reported_date_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that fetch SQL falls back to reported dates for clearly invalid occurred dates."""
    captured: dict[str, object] = {}

    def fake_read_sql_query(sql, con, params):
        captured["sql"] = str(sql)
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(feature_builder.pd, "read_sql_query", fake_read_sql_query)

    feature_builder._fetch_crime_records(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
    )

    query_text = cast(str, captured["sql"])
    assert "WHEN occurred_date > reported_date THEN reported_date" in query_text
    assert "WHEN occurred_date IS NULL THEN reported_date" in query_text
    assert "INTERVAL" not in query_text
    assert "AS used_reported_date_fallback" in query_text
    assert cast(dict[str, date], captured["params"]) == {
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 1, 31),
    }
