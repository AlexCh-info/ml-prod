"""Сценарии спеки data-quality-gate end-to-end: `taxi validate` + настоящий GX + наш контракт.

Данные синтетические и детерминированные: так тест работает в CI без `dvc pull`, а каждая
«порча» затрагивает ровно то, что проверяется. Контекст GX — копия gx/ из репозитория во
временном каталоге: тесты не трогают ни закоммиченный конфиг, ни локальные Data Docs.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from taxi_duration.cli import app
from taxi_duration.config import FilterSettings
from taxi_duration.data.prepare import prepare_trips

pytestmark = pytest.mark.integration

runner = CliRunner()

MONTH = "2024-01"
ROWS = 12_000  # больше нижней границы числа строк в raw_trips
SEED = 42
REPORT_FIELDS = {"success", "evaluated", "failed_critical", "failed_warning", "run_time"}

Invoke = Callable[..., Any]


def make_raw_month(month: str = MONTH, rows: int = ROWS) -> pd.DataFrame:
    """Валидный сырой месяц TLC: всё внутри месяца, длительность 3..45 мин, разумные значения."""
    rng = np.random.default_rng(SEED)
    start = pd.Timestamp(f"{month}-01")
    month_seconds = int(((start + pd.DateOffset(months=1)) - start).total_seconds())
    # Последний час месяца не используем: высадка должна остаться в том же месяце.
    pickup = start + pd.to_timedelta(rng.integers(0, month_seconds - 3600, rows), unit="s")
    duration = pd.to_timedelta(rng.integers(3 * 60, 45 * 60, rows), unit="s")
    return pd.DataFrame(
        {
            "VendorID": rng.integers(1, 3, rows).astype("int32"),
            "lpep_pickup_datetime": pickup.astype("datetime64[us]"),
            "lpep_dropoff_datetime": (pickup + duration).astype("datetime64[us]"),
            "PULocationID": rng.integers(1, 266, rows).astype("int32"),
            "DOLocationID": rng.integers(1, 266, rows).astype("int32"),
            "passenger_count": rng.integers(1, 7, rows).astype("float64"),
            "trip_distance": rng.uniform(0.0, 20.0, rows).round(2),
            "fare_amount": rng.uniform(3.0, 80.0, rows).round(2),
        }
    )


def corrupt_rows(df: pd.DataFrame, share: float) -> pd.Index:
    """Индексы первых ``share`` строк (детерминированно) — для точечной порчи."""
    return df.index[: int(len(df) * share)]


@pytest.fixture
def project(monkeypatch: pytest.MonkeyPatch, repo_root: Path, tmp_path: Path) -> Path:
    """Временный «репозиторий»: конфиг, копия gx/ без производных файлов, пустые data/ и reports/.

    conftest уже сделал tmp_path текущим каталогом: `taxi validate` ищет gx/ в CWD.
    """
    shutil.copytree(
        repo_root / "gx",
        tmp_path / "gx",
        ignore=shutil.ignore_patterns(
            "uncommitted", "checkpoints", "validation_definitions", "*.json"
        ),
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  data_root: {tmp_path / 'data'}\n  reports_dir: {tmp_path / 'reports'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TAXI_CONFIG", str(config))
    monkeypatch.setenv("TAXI__LOGGING__CONFIG_PATH", str(repo_root / "configs" / "logging.yaml"))
    return tmp_path


def write_raw(project: Path, df: pd.DataFrame, month: str = MONTH) -> None:
    path = project / "data" / "raw" / f"green_tripdata_{month}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_prepared(project: Path, df: pd.DataFrame, month: str = MONTH) -> None:
    path = project / "data" / "processed" / f"trips_{month}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def validate(stage: str, *extra: str) -> Any:
    return runner.invoke(app, ["validate", "--month", MONTH, "--stage", stage, "--no-docs", *extra])


def read_report(project: Path, stage: str) -> dict[str, Any]:
    path = project / "reports" / "validation" / f"{stage}_{MONTH}.json"
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return report


# --- Требование: проверка контракта по команде ----------------------------------------------


def test_valid_raw_month_exits_0_and_writes_report(project: Path) -> None:
    write_raw(project, make_raw_month())

    # Без --no-docs: этот сценарий заодно проверяет обновление HTML Data Docs.
    result = runner.invoke(app, ["validate", "--month", MONTH, "--stage", "raw"])

    assert result.exit_code == 0, result.output
    report = read_report(project, "raw")
    assert report.keys() >= REPORT_FIELDS
    assert report["success"] is True
    assert (report["failed_critical"], report["failed_warning"]) == (0, 0)
    assert report["evaluated"] >= 7
    site = project / "gx" / "uncommitted" / "data_docs" / "local_site"
    assert (site / "index.html").is_file()
    # Прогон назван по стадии и месяцу: в Data Docs видно, какой месяц проверялся.
    assert (site / "validations" / "raw_trips" / f"raw_{MONTH}").is_dir()


def test_valid_prepared_month_exits_0(project: Path) -> None:
    write_prepared(project, prepare_trips(make_raw_month(), MONTH, FilterSettings()))

    result = validate("prepared")

    assert result.exit_code == 0, result.output
    assert read_report(project, "prepared")["success"] is True


def test_missing_input_file_exits_1_with_hint(project: Path) -> None:
    result = validate("raw")
    assert result.exit_code == 1
    assert f"taxi ingest --month {MONTH}" in result.stderr


# --- Требование: критичное нарушение останавливает пайплайн ---------------------------------


def test_missing_column_exits_1_and_names_schema_expectation(project: Path) -> None:
    # Сценарий спеки: из файла удалена колонка PULocationID.
    write_raw(project, make_raw_month().drop(columns=["PULocationID"]))

    result = validate("raw")

    assert result.exit_code == 1
    assert "PULocationID" in result.stderr
    assert "expect_column_to_exist" in result.stderr or "expect_table_columns" in result.stderr
    report = read_report(project, "raw")  # отчёт пишется и при провале
    assert report["success"] is False
    assert report["failed_critical"] >= 1


def test_negative_duration_in_prepared_exits_1(project: Path) -> None:
    # Отрицательная длительность после prepare — баг в коде подготовки, а не шум данных.
    prepared = prepare_trips(make_raw_month(), MONTH, FilterSettings())
    prepared.loc[corrupt_rows(prepared, 0.001), "duration_min"] = -5.0
    write_prepared(project, prepared)

    result = validate("prepared")

    assert result.exit_code == 1
    assert "duration_min" in result.stderr
    (failure,) = read_report(project, "prepared")["failures"]
    assert (failure["column"], failure["severity"]) == ("duration_min", "critical")


def test_dropoff_before_pickup_in_raw_exits_1(project: Path) -> None:
    # Отрицательная длительность в сырых данных: 1% строк «едут назад во времени».
    raw = make_raw_month()
    rows = corrupt_rows(raw, 0.01)
    raw.loc[rows, "lpep_dropoff_datetime"] = raw.loc[rows, "lpep_pickup_datetime"] - pd.Timedelta(
        minutes=5
    )
    write_raw(project, raw)

    result = validate("raw")

    assert result.exit_code == 1
    assert "expect_column_pair_values_a_to_be_greater_than_b" in result.stderr


# --- Требование: предупреждения не останавливают пайплайн -----------------------------------


def test_out_of_month_dates_warn_but_exit_0(project: Path) -> None:
    # Сценарий спеки: 3% поездок с датой вне месяца при пороге 1% (и строже).
    raw = make_raw_month()
    rows = corrupt_rows(raw, 0.03)
    raw.loc[rows, "lpep_pickup_datetime"] = pd.Timestamp("2023-12-31 23:00:00")
    raw.loc[rows, "lpep_dropoff_datetime"] = pd.Timestamp("2023-12-31 23:20:00")
    write_raw(project, raw)

    result = validate("raw")

    assert result.exit_code == 0, result.output
    report = read_report(project, "raw")
    assert report["success"] is True
    assert (report["failed_critical"], report["failed_warning"]) == (0, 1)
    (failure,) = report["failures"]
    assert failure["severity"] == "warning"
    assert failure["column"] == "lpep_pickup_datetime"
    assert failure["unexpected_percent"] == pytest.approx(3.0, abs=0.01)


# --- Отчёт ----------------------------------------------------------------------------------


def test_repeated_run_does_not_rewrite_gx_config(project: Path) -> None:
    # get-or-add в runner: после первой регистрации asset конфиг GX стабилен (нет шума в git).
    write_raw(project, make_raw_month())
    config = project / "gx" / "great_expectations.yml"

    assert validate("raw").exit_code == 0
    before = config.read_bytes()
    assert validate("raw").exit_code == 0

    assert config.read_bytes() == before
