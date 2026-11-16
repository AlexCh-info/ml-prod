"""Сценарии спеки zone-features: окно строго по прошлым дням и event_timestamp."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from taxi_duration.errors import DataError
from taxi_duration.features.zone_stats import build_zone_stats, zone_stats_schema


def _trips(*rows: tuple[str, int, float, float]) -> pd.DataFrame:
    """Поездки: (pickup, pu_zone, duration_min, trip_distance)."""
    df = pd.DataFrame(rows, columns=["pickup_datetime", "pu_zone", "duration_min", "trip_distance"])
    return df.astype(
        {
            "pickup_datetime": "datetime64[us]",
            "pu_zone": "int32",
            "duration_min": "float64",
            "trip_distance": "float64",
        }
    )


def _row(stats: pd.DataFrame, zone: int, day: str) -> pd.Series:
    match = stats[(stats["pu_zone"] == zone) & (stats["event_timestamp"] == pd.Timestamp(day))]
    assert len(match) == 1, f"expected one row for zone={zone} day={day}, got {len(match)}"
    return match.iloc[0]


# --- Текущий день не попадает в окно ----------------------------------------------------------


def test_current_day_is_not_in_window() -> None:
    stats = build_zone_stats(_trips(*[("2024-01-10 12:00", 74, 10.0, 1.0)] * 7))
    assert _row(stats, 74, "2024-01-10")["zone_trips_7d"] == 0
    # На следующий день поездки 10-го уже известны.
    assert _row(stats, 74, "2024-01-11")["zone_trips_7d"] == 7


def test_late_trip_on_same_day_does_not_leak() -> None:
    # Поездка в 23:59 дня D не должна влиять на строку дня D.
    base = _trips(("2024-01-09 10:00", 1, 10.0, 1.0))
    with_late = _trips(("2024-01-09 10:00", 1, 10.0, 1.0), ("2024-01-10 23:59", 1, 50.0, 9.0))
    before = _row(build_zone_stats(base), 1, "2024-01-10")
    after = _row(build_zone_stats(with_late), 1, "2024-01-10")
    pd.testing.assert_series_equal(before, after)


# --- Метка времени доступности -----------------------------------------------------------------


def test_event_timestamp_is_day_after_window_end() -> None:
    # Окно, которое заканчивается 2024-01-09, видно в строке с event_timestamp 2024-01-10T00:00.
    stats = build_zone_stats(_trips(("2024-01-09 18:30", 5, 12.0, 2.0)))
    row = _row(stats, 5, "2024-01-10")
    assert row["event_timestamp"] == pd.Timestamp("2024-01-10T00:00:00")
    assert row["zone_trips_7d"] == 1


def test_event_timestamps_are_midnights() -> None:
    stats = build_zone_stats(_trips(("2024-01-01 13:17", 1, 10.0, 1.0)))
    assert (stats["event_timestamp"] == stats["event_timestamp"].dt.floor("D")).all()


# --- Границы окна --------------------------------------------------------------------------------


def test_window_covers_exactly_7_previous_days() -> None:
    stats = build_zone_stats(
        _trips(("2024-01-01 10:00", 1, 10.0, 1.0), ("2024-01-15 10:00", 1, 10.0, 1.0))
    )
    # Строка 01-08: окно 01-01..01-07 — поездка 01-01 внутри.
    assert _row(stats, 1, "2024-01-08")["zone_trips_7d"] == 1
    # Строка 01-09: окно 01-02..01-08 — поездка 01-01 уже вышла.
    assert _row(stats, 1, "2024-01-09")["zone_trips_7d"] == 0


def test_days_without_trips_still_count_as_days() -> None:
    # Если бы окно шло по строкам, а не по дням, редкие поездки «растягивали» бы его.
    stats = build_zone_stats(
        _trips(("2024-01-01 10:00", 1, 10.0, 1.0), ("2024-01-20 10:00", 1, 30.0, 3.0))
    )
    assert _row(stats, 1, "2024-01-21")["zone_trips_7d"] == 1
    assert _row(stats, 1, "2024-01-21")["zone_mean_duration_7d"] == pytest.approx(30.0)


def test_custom_window_changes_columns_and_span() -> None:
    stats = build_zone_stats(
        _trips(("2024-01-01 10:00", 1, 10.0, 1.0), ("2024-01-05 10:00", 1, 10.0, 1.0)),
        window_days=2,
    )
    assert list(stats.columns) == list(zone_stats_schema(2))
    assert _row(stats, 1, "2024-01-03")["zone_trips_2d"] == 1
    assert _row(stats, 1, "2024-01-04")["zone_trips_2d"] == 0


def test_window_crosses_month_boundary() -> None:
    stats = build_zone_stats(
        _trips(("2024-01-31 22:00", 1, 10.0, 1.0), ("2024-02-03 10:00", 1, 10.0, 1.0))
    )
    assert _row(stats, 1, "2024-02-01")["zone_trips_7d"] == 1


# --- Средние -----------------------------------------------------------------------------------


def test_means_are_weighted_by_trips() -> None:
    stats = build_zone_stats(
        _trips(
            ("2024-01-01 10:00", 1, 10.0, 1.0),
            ("2024-01-01 11:00", 1, 20.0, 2.0),
            ("2024-01-02 10:00", 1, 60.0, 9.0),
        )
    )
    row = _row(stats, 1, "2024-01-03")
    assert row["zone_trips_7d"] == 3
    # (10 + 20 + 60) / 3, а не среднее дневных средних (15 + 60) / 2.
    assert row["zone_mean_duration_7d"] == pytest.approx(30.0)
    assert row["zone_mean_distance_7d"] == pytest.approx(4.0)


def test_means_are_nan_without_trips_in_window() -> None:
    stats = build_zone_stats(_trips(("2024-01-10 10:00", 1, 10.0, 1.0)))
    row = _row(stats, 1, "2024-01-10")
    assert math.isnan(row["zone_mean_duration_7d"])
    assert math.isnan(row["zone_mean_distance_7d"])


def test_zones_are_independent() -> None:
    stats = build_zone_stats(
        _trips(("2024-01-01 10:00", 1, 10.0, 1.0), ("2024-01-01 10:00", 2, 40.0, 4.0))
    )
    assert _row(stats, 1, "2024-01-02")["zone_mean_duration_7d"] == pytest.approx(10.0)
    assert _row(stats, 2, "2024-01-02")["zone_mean_duration_7d"] == pytest.approx(40.0)


# --- Форма результата ---------------------------------------------------------------------------


def test_grid_spans_first_day_to_day_after_last() -> None:
    stats = build_zone_stats(
        _trips(("2024-01-01 10:00", 1, 10.0, 1.0), ("2024-01-05 10:00", 2, 10.0, 1.0))
    )
    expected_days = list(pd.date_range("2024-01-01", "2024-01-06", freq="D"))
    for zone in (1, 2):
        days = stats.loc[stats["pu_zone"] == zone, "event_timestamp"].tolist()
        assert days == expected_days


def test_schema_order_and_uniqueness() -> None:
    stats = build_zone_stats(
        _trips(("2024-01-03 10:00", 2, 10.0, 1.0), ("2024-01-01 10:00", 1, 10.0, 1.0))
    )
    assert {col: str(dtype) for col, dtype in stats.dtypes.items()} == zone_stats_schema(7)
    assert list(stats.columns) == list(zone_stats_schema(7))
    assert not stats.duplicated(["pu_zone", "event_timestamp"]).any()
    assert stats.equals(stats.sort_values(["pu_zone", "event_timestamp"], ignore_index=True))
    assert isinstance(stats.index, pd.RangeIndex)


def test_empty_input_gives_empty_result_with_schema() -> None:
    stats = build_zone_stats(_trips())
    assert stats.empty
    assert {col: str(dtype) for col, dtype in stats.dtypes.items()} == zone_stats_schema(7)


def test_missing_column_raises_data_error() -> None:
    with pytest.raises(DataError, match="duration_min"):
        build_zone_stats(_trips().drop(columns=["duration_min"]))


def test_invalid_window_raises_data_error() -> None:
    with pytest.raises(DataError, match="window_days"):
        build_zone_stats(_trips(), window_days=0)


# --- Свойство: будущее не влияет на прошлое ------------------------------------------------------


def test_future_trips_never_change_past_rows() -> None:
    rng = np.random.default_rng(42)
    n = 500
    trips = pd.DataFrame(
        {
            "pickup_datetime": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 31 * 24 * 60, n), unit="min"),
            "pu_zone": rng.integers(1, 6, n).astype("int32"),
            "duration_min": rng.uniform(1, 60, n),
            "trip_distance": rng.uniform(0, 10, n),
        }
    ).astype({"pickup_datetime": "datetime64[us]"})
    cutoff = pd.Timestamp("2024-01-15")

    full = build_zone_stats(trips)
    past_only = build_zone_stats(trips[trips["pickup_datetime"] < cutoff])

    # Строки с event_timestamp <= cutoff зависят только от поездок до cutoff.
    full_past = full[full["event_timestamp"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_past, past_only.reset_index(drop=True))
