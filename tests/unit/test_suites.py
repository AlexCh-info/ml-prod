"""Контракт наборов ожиданий (задачи 2.2 и 2.3 [YOU]): структура suite, без данных.

Проверяется, ЧТО описано в контракте (типы ожиданий, колонки, severity, параметры),
а не как оно срабатывает на данных — это интеграционный тест (задача 2.4).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import great_expectations as gx
import pytest

from taxi_duration.data.prepare import PREPARED_SCHEMA, RAW_REQUIRED_COLUMNS
from taxi_duration.validation.runner import CRITICAL, WARNING, Stage
from taxi_duration.validation.suites import (
    PREPARED_TRIPS_SUITE,
    RAW_TRIPS_SUITE,
    build_prepared_trips_suite,
    build_raw_trips_suite,
)

Expectation = dict[str, Any]

MONTH_START = {"$PARAMETER": "month_start"}
MONTH_END = {"$PARAMETER": "month_end"}
RAW_KEY_FIELDS = ("lpep_pickup_datetime", "lpep_dropoff_datetime", "PULocationID", "DOLocationID")


@pytest.fixture(autouse=True)
def gx_context() -> Iterator[None]:
    # ExpectationSuite требует активный контекст; эфемерный не пишет на диск.
    gx.get_context(mode="ephemeral")
    yield


def expectations(builder: Callable[[], gx.ExpectationSuite]) -> list[Expectation]:
    items = builder().to_json_dict()["expectations"]
    assert isinstance(items, list)
    return [item for item in items if isinstance(item, dict)]


def find(
    items: list[Expectation], type_: str, *, column: str | None = None, **kwargs: Any
) -> list[Expectation]:
    """Ожидания нужного типа (и колонки), у которых kwargs содержат заданные значения."""
    return [
        item
        for item in items
        if item["type"] == type_
        and (column is None or item["kwargs"].get("column") == column)
        and all(item["kwargs"].get(key) == value for key, value in kwargs.items())
    ]


def one(items: list[Expectation], type_: str, **kwargs: Any) -> Expectation:
    found = find(items, type_, **kwargs)
    assert found, f"no {type_} with {kwargs}"
    return found[0]


def test_suite_names_match_stages() -> None:
    # runner ищет suite по имени стадии: расхождение = дубли в gx/ и отказ запуска.
    assert (
        Stage.RAW.suite_name,
        Stage.PREPARED.suite_name,
    ) == (RAW_TRIPS_SUITE, PREPARED_TRIPS_SUITE)


# --- 2.2 raw_trips ------------------------------------------------------------------------


@pytest.fixture
def raw() -> list[Expectation]:
    return expectations(build_raw_trips_suite)


def test_raw_suite_name_and_size() -> None:
    suite = build_raw_trips_suite()
    assert suite.name == RAW_TRIPS_SUITE
    assert len(suite.expectations) >= 7


def test_raw_uses_both_severities(raw: list[Expectation]) -> None:
    assert {item["severity"] for item in raw} == {CRITICAL, WARNING}


def test_raw_schema_is_critical(raw: list[Expectation]) -> None:
    covered: set[str] = set()
    for item in raw:
        if item["severity"] != CRITICAL:
            continue
        kwargs = item["kwargs"]
        if item["type"] == "expect_column_to_exist":
            covered.add(kwargs["column"])
        elif item["type"] == "expect_table_columns_to_match_set":
            covered.update(kwargs["column_set"])
        elif item["type"] == "expect_table_columns_to_match_ordered_list":
            covered.update(kwargs["column_list"])
    missing = set(RAW_REQUIRED_COLUMNS) - covered
    assert not missing, f"schema expectations do not cover: {sorted(missing)}"


@pytest.mark.parametrize("column", RAW_KEY_FIELDS)
def test_raw_key_fields_not_null_critical(raw: list[Expectation], column: str) -> None:
    item = one(raw, "expect_column_values_to_not_be_null", column=column)
    assert item["severity"] == CRITICAL


def test_raw_dropoff_not_before_pickup_critical(raw: list[Expectation]) -> None:
    item = one(
        raw,
        "expect_column_pair_values_a_to_be_greater_than_b",
        column_A="lpep_dropoff_datetime",
        column_B="lpep_pickup_datetime",
    )
    assert item["kwargs"].get("or_equal") is True
    assert item["severity"] == CRITICAL


def test_raw_row_count_critical(raw: list[Expectation]) -> None:
    item = one(raw, "expect_table_row_count_to_be_between")
    assert item["severity"] == CRITICAL
    assert (item["kwargs"].get("min_value") or 0) > 0, "empty file must fail"


def test_raw_pickup_in_month_warning(raw: list[Expectation]) -> None:
    item = one(
        raw,
        "expect_column_values_to_be_between",
        column="lpep_pickup_datetime",
        min_value=MONTH_START,
        max_value=MONTH_END,
    )
    assert item["severity"] == WARNING
    assert item["kwargs"].get("strict_max") is True, "month_end is the next month start"
    assert 0 < item["kwargs"].get("mostly", 1) < 1


@pytest.mark.parametrize("column", ["trip_distance", "passenger_count"])
def test_raw_ranges_warning(raw: list[Expectation], column: str) -> None:
    item = one(raw, "expect_column_values_to_be_between", column=column)
    assert item["severity"] == WARNING
    assert 0 < item["kwargs"].get("mostly", 1) < 1


# --- 2.3 prepared_trips -------------------------------------------------------------------


@pytest.fixture
def prepared() -> list[Expectation]:
    return expectations(build_prepared_trips_suite)


def test_prepared_suite_name() -> None:
    assert build_prepared_trips_suite().name == PREPARED_TRIPS_SUITE


def test_prepared_schema_is_exact_ordered_critical(prepared: list[Expectation]) -> None:
    item = one(prepared, "expect_table_columns_to_match_ordered_list")
    assert item["kwargs"]["column_list"] == list(PREPARED_SCHEMA)
    assert item["severity"] == CRITICAL


@pytest.mark.parametrize(("column", "dtype"), list(PREPARED_SCHEMA.items()))
def test_prepared_column_types_critical(
    prepared: list[Expectation], column: str, dtype: str
) -> None:
    item = one(prepared, "expect_column_values_to_be_of_type", column=column)
    assert item["kwargs"]["type_"] == dtype
    assert item["severity"] == CRITICAL


@pytest.mark.parametrize(
    ("column", "low", "high"), [("duration_min", 1, 60), ("pickup_hour", 0, 23)]
)
def test_prepared_ranges_strict_critical(
    prepared: list[Expectation], column: str, low: int, high: int
) -> None:
    item = one(prepared, "expect_column_values_to_be_between", column=column)
    kwargs = item["kwargs"]
    assert (kwargs.get("min_value"), kwargs.get("max_value")) == (low, high)
    assert kwargs.get("mostly", 1) == 1, "prepare guarantees the range: any violation is a bug"
    assert item["severity"] == CRITICAL


def test_prepared_has_no_uniqueness_checks(prepared: list[Expectation]) -> None:
    unique_types = {
        "expect_column_values_to_be_unique",
        "expect_compound_columns_to_be_unique",
        "expect_select_column_values_to_be_unique_within_record",
    }
    assert not [item for item in prepared if item["type"] in unique_types]
