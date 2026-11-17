"""Разбор результата GX и политика severity (спека data-quality-gate), без запуска GX.

Фикстуры повторяют форму ``ExpectationSuiteValidationResult.to_json_dict()`` GX 1.x.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from taxi_duration.errors import ConfigError
from taxi_duration.storage import validation_report_uri, write_json
from taxi_duration.validation.runner import (
    CRITICAL,
    WARNING,
    ExpectationFailure,
    Stage,
    format_failures,
    run_validation,
    suite_parameters,
    summarize_result,
)

RUN_TIME = datetime(2026, 11, 17, 9, 30, tzinfo=UTC)
REPORT_FIELDS = {"success", "evaluated", "failed_critical", "failed_warning", "run_time"}


def evr(
    expectation: str,
    *,
    success: bool | None,
    severity: str | None = CRITICAL,
    kwargs: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    exception_info: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Один результат ожидания в форме to_json_dict()."""
    config: dict[str, Any] = {"type": expectation, "kwargs": kwargs or {}, "meta": meta or {}}
    if severity is not None:
        config["severity"] = severity
    return {
        "success": success,
        "expectation_config": config,
        "result": result or {},
        "exception_info": exception_info
        or {"raised_exception": False, "exception_message": None, "exception_traceback": None},
    }


def suite_result(*results: dict[str, Any], suite_name: str = "raw_trips") -> dict[str, Any]:
    return {
        "success": all(r["success"] for r in results),
        "suite_name": suite_name,
        "results": list(results),
    }


def summarize(*results: dict[str, Any]) -> Any:
    return summarize_result(
        suite_result(*results), stage=Stage.RAW, month="2024-01", run_time=RUN_TIME
    )


NOT_NULL_OK = evr(
    "expect_column_values_to_not_be_null",
    success=True,
    kwargs={"column": "PULocationID"},
    result={"unexpected_percent": 0.0},
)
OUT_OF_MONTH_3PCT = evr(
    "expect_column_values_to_be_between",
    success=False,
    severity=WARNING,
    kwargs={"column": "lpep_pickup_datetime", "mostly": 0.99},
    result={"unexpected_percent": 3.0},
)
SCHEMA_FAILED = evr(
    "expect_table_columns_to_match_set",
    success=False,
    kwargs={"column_set": ["PULocationID", "trip_distance"]},
    result={"observed_value": ["trip_distance"]},
)


def test_all_passed_is_success() -> None:
    report = summarize(NOT_NULL_OK, NOT_NULL_OK)
    assert report.success
    assert (report.evaluated, report.failed_critical, report.failed_warning) == (2, 0, 0)
    assert report.success_percent == 100.0
    assert report.failures == ()


def test_warning_failure_does_not_fail_gate() -> None:
    # Сценарий спеки: 3% поездок вне месяца при пороге 1% → код 0, в отчёте провал warning.
    report = summarize(NOT_NULL_OK, OUT_OF_MONTH_3PCT)
    assert report.success
    assert (report.failed_critical, report.failed_warning) == (0, 1)
    (failure,) = report.failures
    assert failure.severity == WARNING
    assert not failure.blocking
    assert failure.column == "lpep_pickup_datetime"
    assert failure.unexpected_percent == 3.0
    assert report.success_percent == 50.0


def test_critical_failure_fails_gate() -> None:
    report = summarize(NOT_NULL_OK, OUT_OF_MONTH_3PCT, SCHEMA_FAILED)
    assert not report.success
    assert (report.failed_critical, report.failed_warning) == (1, 1)
    (critical,) = report.critical_failures
    assert critical.expectation == "expect_table_columns_to_match_set"
    assert critical.column is None
    assert critical.observed_value == ["trip_distance"]


def test_missing_severity_defaults_to_critical() -> None:
    # Как в GX: ожидание без severity — critical.
    report = summarize(evr("expect_column_to_exist", success=False, severity=None))
    assert report.failed_critical == 1
    assert not report.success


def test_meta_severity_is_used_when_field_missing() -> None:
    item = evr("expect_column_to_exist", success=False, severity=None, meta={"severity": "warning"})
    assert summarize(item).failed_warning == 1


def test_severity_is_case_insensitive() -> None:
    assert summarize(evr("expect_column_to_exist", success=False, severity="WARNING")).success


def test_info_severity_is_not_blocking() -> None:
    report = summarize(evr("expect_column_to_exist", success=False, severity="info"))
    assert report.success
    assert report.failed_warning == 1


def test_unknown_severity_is_treated_as_critical(caplog: pytest.LogCaptureFixture) -> None:
    # Опечатка в контракте не должна тихо пропускать данные.
    with caplog.at_level(logging.WARNING, logger="taxi_duration"):
        report = summarize(evr("expect_column_to_exist", success=False, severity="critcal"))
    assert report.failed_critical == 1
    assert "critcal" in caplog.text


def test_none_success_counts_as_failure() -> None:
    # GX может вернуть success=None, если ожидание не удалось оценить.
    report = summarize(evr("expect_column_to_exist", success=None, severity=CRITICAL))
    assert report.failed_critical == 1


def test_exception_in_metric_is_reported() -> None:
    # Колонки нет → GX ловит исключение расчёта метрики, ключ — id метрики.
    item = evr(
        "expect_column_values_to_not_be_null",
        success=False,
        kwargs={"column": "PULocationID"},
        exception_info={
            "MetricConfigurationID(metric_name='column_values.nonnull.condition')": {
                "raised_exception": True,
                "exception_message": 'The column "PULocationID" in BatchData does not exist.',
                "exception_traceback": "Traceback ...",
            }
        },
    )
    (failure,) = summarize(item).failures
    assert failure.error is not None
    assert "PULocationID" in failure.error
    assert "error:" in failure.describe()


def test_flat_exception_info_is_reported() -> None:
    item = evr(
        "expect_column_to_exist",
        success=False,
        exception_info={"raised_exception": True, "exception_message": "x" * 1000},
    )
    (failure,) = summarize(item).failures
    assert failure.error is not None
    assert len(failure.error) == 300


def test_pair_expectation_column_is_joined() -> None:
    item = evr(
        "expect_column_pair_values_a_to_be_greater_than_b",
        success=False,
        kwargs={"column_A": "lpep_dropoff_datetime", "column_B": "lpep_pickup_datetime"},
        result={"unexpected_percent": 0.5},
    )
    (failure,) = summarize(item).failures
    assert failure.column == "lpep_dropoff_datetime,lpep_pickup_datetime"


def test_empty_result_is_success() -> None:
    report = summarize()
    assert report.success
    assert (report.evaluated, report.success_percent) == (0, 100.0)


def test_report_dict_has_spec_fields_and_is_json() -> None:
    report = summarize(NOT_NULL_OK, OUT_OF_MONTH_3PCT, SCHEMA_FAILED)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload.keys() >= REPORT_FIELDS
    assert payload["stage"] == "raw"
    assert payload["month"] == "2024-01"
    assert payload["suite_name"] == "raw_trips"
    assert payload["run_time"] == "2026-11-17T09:30:00+00:00"
    assert [f["severity"] for f in payload["failures"]] == ["warning", "critical"]


def test_run_time_is_converted_to_utc() -> None:
    moscow = datetime(2026, 11, 17, 12, 30, tzinfo=timezone(timedelta(hours=3)))
    report = summarize_result(
        suite_result(), stage=Stage.PREPARED, month="2024-01", run_time=moscow
    )
    assert report.run_time == "2026-11-17T09:30:00+00:00"
    assert report.suite_name == "raw_trips"  # из результата GX, а не из стадии


def test_format_failures_lists_each_failure() -> None:
    failures = [
        ExpectationFailure("expect_a", CRITICAL, "col", 1.5, None, None),
        ExpectationFailure("expect_b", WARNING, None, None, 10, None),
    ]
    assert format_failures(failures) == (
        "expect_a(col) [critical]: unexpected 1.500%; expect_b [warning]: observed 10"
    )


@pytest.mark.parametrize(
    ("month", "start", "end"),
    [
        ("2024-01", "2024-01-01T00:00:00", "2024-02-01T00:00:00"),
        ("2024-12", "2024-12-01T00:00:00", "2025-01-01T00:00:00"),
    ],
)
def test_suite_parameters(month: str, start: str, end: str) -> None:
    assert suite_parameters(month) == {"month_start": start, "month_end": end}


def test_suite_parameters_reject_bad_month() -> None:
    with pytest.raises(ValueError, match="YYYY-MM"):
        suite_parameters("2024-13")


def test_stage_suite_names() -> None:
    assert [stage.suite_name for stage in Stage] == ["raw_trips", "prepared_trips"]


def test_run_validation_requires_gx_context(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="GX context not found"):
        run_validation(
            pd.DataFrame({"x": [1]}),
            stage=Stage.RAW,
            month="2024-01",
            project_root=tmp_path,
            report_uri=str(tmp_path / "report.json"),
        )
    # Нельзя молча создавать новый контекст в чужом каталоге.
    assert not (tmp_path / "gx").exists()


def test_validation_report_uri() -> None:
    assert (
        validation_report_uri("reports", "raw", "2024-01") == "reports/validation/raw_2024-01.json"
    )
    assert (
        validation_report_uri("s3://reports/", "prepared", "2024-02")
        == "s3://reports/validation/prepared_2024-02.json"
    )


def test_write_json_creates_dirs_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "reports" / "validation" / "raw_2024-01.json"
    write_json({"success": True, "msg": "привет"}, str(target))
    write_json({"success": False}, str(target))  # перезапись
    assert json.loads(target.read_text(encoding="utf-8")) == {"success": False}
    assert [p.name for p in target.parent.iterdir()] == [target.name]
