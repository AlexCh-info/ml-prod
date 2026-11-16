"""Сценарии спеки trip-preparation и контракт ``prepare_trips`` из docstring."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from taxi_duration.config import FilterSettings
from taxi_duration.data.prepare import PREPARED_SCHEMA, RAW_REQUIRED_COLUMNS, prepare_trips
from taxi_duration.errors import DataError

MONTH = "2024-01"
DEFAULT_FILTERS = FilterSettings(min_duration=1, max_duration=60)


def _trip(
    pickup: str = "2024-01-10 10:00:00",
    minutes: float = 15.0,
    pu: int = 74,
    do: int = 75,
    distance: float = 2.5,
) -> dict[str, Any]:
    start = pd.Timestamp(pickup)
    return {
        "VendorID": 2,
        "lpep_pickup_datetime": start,
        "lpep_dropoff_datetime": start + pd.Timedelta(minutes=minutes),
        "PULocationID": pu,
        "DOLocationID": do,
        "passenger_count": 1.0,
        "trip_distance": distance,
        "fare_amount": 12.1,
    }


def _raw(*trips: dict[str, Any]) -> pd.DataFrame:
    """Сырой DataFrame с dtype как в файле TLC (int32, datetime64[us])."""
    return pd.DataFrame(list(trips)).astype(
        {
            "lpep_pickup_datetime": "datetime64[us]",
            "lpep_dropoff_datetime": "datetime64[us]",
            "PULocationID": "int32",
            "DOLocationID": "int32",
            "trip_distance": "float64",
        }
    )


def _prepare(raw: pd.DataFrame, filters: FilterSettings = DEFAULT_FILTERS) -> pd.DataFrame:
    return prepare_trips(raw, MONTH, filters)


# --- Расчёт таргета ---------------------------------------------------------------------


def test_duration_15_minutes() -> None:
    result = _prepare(_raw(_trip(pickup="2024-01-10 10:00:00", minutes=15)))
    assert result["duration_min"].tolist() == [15.0]


def test_duration_is_fractional() -> None:
    result = _prepare(_raw(_trip(minutes=15.5)))
    assert result["duration_min"].tolist() == [pytest.approx(15.5)]


# --- Фильтрация выбросов -----------------------------------------------------------------


def test_too_short_trip_is_dropped() -> None:
    result = _prepare(_raw(_trip(minutes=0.5), _trip(minutes=10)))
    assert result["duration_min"].tolist() == [10.0]


def test_too_long_trip_is_dropped() -> None:
    result = _prepare(_raw(_trip(minutes=61), _trip(minutes=10)))
    assert result["duration_min"].tolist() == [10.0]


def test_negative_duration_is_dropped() -> None:
    result = _prepare(_raw(_trip(minutes=-5), _trip(minutes=10)))
    assert result["duration_min"].tolist() == [10.0]


def test_duration_bounds_are_inclusive() -> None:
    result = _prepare(_raw(_trip(minutes=1), _trip(minutes=60)))
    assert result["duration_min"].tolist() == [1.0, 60.0]


def test_filters_come_from_settings() -> None:
    result = _prepare(
        _raw(_trip(minutes=3), _trip(minutes=20), _trip(minutes=40)),
        FilterSettings(min_duration=5, max_duration=30),
    )
    assert result["duration_min"].tolist() == [20.0]


# --- Поездки из другого месяца -------------------------------------------------------------


def test_trip_from_2009_is_dropped() -> None:
    result = _prepare(_raw(_trip(pickup="2009-01-01 10:00:00"), _trip()))
    assert len(result) == 1
    assert result["pickup_datetime"].iloc[0] == pd.Timestamp("2024-01-10 10:00:00")


@pytest.mark.parametrize(
    ("pickup", "kept"),
    [
        ("2023-12-31 23:59:59", False),
        ("2024-01-01 00:00:00", True),
        ("2024-01-31 23:59:59", True),
        ("2024-02-01 00:00:00", False),
    ],
)
def test_month_boundaries(pickup: str, kept: bool) -> None:
    result = _prepare(_raw(_trip(pickup=pickup, minutes=10)))
    assert (len(result) == 1) is kept


def test_month_is_taken_from_argument() -> None:
    raw = _raw(_trip(pickup="2024-01-10 10:00:00"), _trip(pickup="2024-02-10 10:00:00"))
    result = prepare_trips(raw, "2024-02", DEFAULT_FILTERS)
    assert result["pickup_datetime"].tolist() == [pd.Timestamp("2024-02-10 10:00:00")]


# --- Пропуски -------------------------------------------------------------------------------


@pytest.mark.parametrize("column", RAW_REQUIRED_COLUMNS)
def test_rows_with_missing_required_value_are_dropped(column: str) -> None:
    raw = _raw(_trip(minutes=10), _trip(minutes=20))
    if column in {"PULocationID", "DOLocationID"}:
        raw[column] = raw[column].astype("Int32")  # nullable int: так пропуск выглядит в zone-ID
    raw.loc[0, column] = None  # NaT / NaN / <NA> в зависимости от dtype
    result = _prepare(raw)
    assert result["duration_min"].tolist() == [20.0]


# --- Схема ----------------------------------------------------------------------------------


def test_schema_matches_contract() -> None:
    result = _prepare(_raw(_trip(), _trip(minutes=30)))
    assert list(result.columns) == list(PREPARED_SCHEMA)
    assert {col: str(dtype) for col, dtype in result.dtypes.items()} == PREPARED_SCHEMA


def test_schema_of_empty_result() -> None:
    result = _prepare(_raw(_trip(minutes=0.1)))
    assert result.empty
    assert {col: str(dtype) for col, dtype in result.dtypes.items()} == PREPARED_SCHEMA


def test_columns_are_mapped_from_raw() -> None:
    # 2024-01-01 — понедельник.
    result = _prepare(_raw(_trip(pickup="2024-01-01 17:45:00", pu=10, do=20, distance=3.3)))
    row = result.iloc[0]
    assert row["pickup_datetime"] == pd.Timestamp("2024-01-01 17:45:00")
    assert (row["pu_zone"], row["do_zone"]) == (10, 20)
    assert row["trip_distance"] == pytest.approx(3.3)
    assert row["pickup_hour"] == 17
    assert row["pickup_weekday"] == 0


def test_sunday_is_6() -> None:
    result = _prepare(_raw(_trip(pickup="2024-01-07 08:00:00")))
    assert result["pickup_weekday"].tolist() == [6]


def test_index_is_reset_and_order_kept() -> None:
    result = _prepare(_raw(_trip(minutes=0.1), _trip(minutes=30), _trip(minutes=0.2), _trip()))
    assert isinstance(result.index, pd.RangeIndex)
    assert result.index.tolist() == [0, 1]
    assert result["duration_min"].tolist() == [30.0, 15.0]


# --- Ошибки и чистота -----------------------------------------------------------------------


def test_missing_column_raises_data_error_with_names() -> None:
    raw = _raw(_trip()).drop(columns=["PULocationID", "trip_distance"])
    with pytest.raises(DataError, match="PULocationID") as exc_info:
        _prepare(raw)
    assert "trip_distance" in str(exc_info.value)


def test_invalid_month_raises_data_error() -> None:
    with pytest.raises(DataError):
        prepare_trips(_raw(_trip()), "2024-13", DEFAULT_FILTERS)


def test_input_is_not_mutated() -> None:
    raw = _raw(_trip(minutes=0.5), _trip(), _trip(pickup="2009-01-01 10:00:00"))
    snapshot = raw.copy(deep=True)
    _prepare(raw)
    pd.testing.assert_frame_equal(raw, snapshot)
