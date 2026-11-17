"""Quality gate: прогнать контракт данных (GX Checkpoint) и решить, пускать ли данные дальше.

Поток одного запуска ``run_validation``::

    файловый контекст gx/ ─► Data Source → Asset → Batch Definition   (get-or-add, стабильно)
                           ─► Suite (из кода, add_or_update)          (источник правды — suites.py)
                           ─► Validation Definition → Checkpoint      (+ action: Data Docs)
                           ─► run(dataframe, month_start/month_end)
                           ─► summarize_result → JSON-отчёт → лог → DataValidationError?

Политика серьёзности (design.md, решение 3):

- провал хотя бы одного ``critical``-ожидания → ``DataValidationError`` → CLI код 1;
- провалы ``warning`` (и ``info``) только попадают в отчёт и лог;
- ожидание без severity GX считает ``critical`` — это поведение GX, мы его не меняем.

Разбор результата (``summarize_result``) работает со словарём ``to_json_dict()``,
а не с объектами GX: так его можно тестировать на JSON-фикстуре без запуска GX.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from taxi_duration.config import parse_month
from taxi_duration.errors import ConfigError, DataValidationError
from taxi_duration.storage import write_json

if TYPE_CHECKING:
    from great_expectations.core.batch_definition import BatchDefinition
    from great_expectations.core.expectation_suite import ExpectationSuite
    from great_expectations.data_context.data_context.abstract_data_context import (
        AbstractDataContext,
    )

logger = logging.getLogger(__name__)

# Имена объектов в контексте стабильны: повторный запуск обновляет их, а не плодит дубли.
DATASOURCE_NAME = "taxi_pandas"
BATCH_DEFINITION_NAME = "whole_dataframe"
DATA_DOCS_ACTION_NAME = "update_data_docs"
GX_CONFIG_RELPATH = Path("gx") / "great_expectations.yml"

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"
_KNOWN_SEVERITIES = frozenset({CRITICAL, WARNING, INFO})

# Трассировки GX длинные; в отчёт и сообщение кладём только начало текста ошибки.
_MAX_ERROR_CHARS = 300


class Stage(StrEnum):
    """Стадия пайплайна, данные которой проверяются. У каждой — свой набор ожиданий."""

    RAW = "raw"
    PREPARED = "prepared"

    @property
    def suite_name(self) -> str:
        """Имя suite, asset, validation definition и checkpoint (``raw_trips``...)."""
        return f"{self.value}_trips"


@dataclass(frozen=True, slots=True)
class ExpectationFailure:
    """Одно проваленное ожидание в отчёте."""

    expectation: str  # тип ожидания GX, например expect_column_values_to_not_be_null
    severity: str  # critical | warning | info
    column: str | None  # колонка (или "A,B" для парных ожиданий); None — ожидание на таблицу
    unexpected_percent: float | None  # доля нарушивших строк, %; None — не построчное
    observed_value: Any  # что увидел GX (число строк, список колонок...), если применимо
    error: str | None  # текст исключения, если ожидание не смогло посчитаться

    @property
    def blocking(self) -> bool:
        """Останавливает ли этот провал пайплайн."""
        return self.severity == CRITICAL

    def describe(self) -> str:
        """Короткая строка для лога и сообщения об ошибке."""
        target = f"({self.column})" if self.column else ""
        if self.error:
            detail = f"error: {self.error}"
        elif self.unexpected_percent is not None:
            detail = f"unexpected {self.unexpected_percent:.3f}%"
        else:
            detail = f"observed {self.observed_value!r}"
        return f"{self.expectation}{target} [{self.severity}]: {detail}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Итог проверки одного месяца одной стадии. ``to_dict()`` — содержимое JSON-отчёта.

    ``success`` — решение quality gate (нет проваленных critical), а не «всё прошло»:
    при одних warning-провалах ``success`` = true, а ``failed_warning`` > 0.
    """

    stage: str
    month: str
    suite_name: str
    success: bool
    evaluated: int
    failed_critical: int
    failed_warning: int
    success_percent: float  # доля прошедших ожиданий, % — удобна в `dvc metrics diff`
    run_time: str  # ISO-8601 UTC начала проверки
    failures: tuple[ExpectationFailure, ...] = ()

    @property
    def critical_failures(self) -> tuple[ExpectationFailure, ...]:
        return tuple(failure for failure in self.failures if failure.blocking)

    @property
    def warning_failures(self) -> tuple[ExpectationFailure, ...]:
        return tuple(failure for failure in self.failures if not failure.blocking)

    def to_dict(self) -> dict[str, Any]:
        """JSON-совместимый словарь; ``failures`` — список словарей."""
        payload = asdict(self)
        payload["failures"] = [asdict(failure) for failure in self.failures]
        return payload


def suite_parameters(month: str) -> dict[str, str]:
    """Параметры suite для месяца: ``{"$PARAMETER": "month_start"}`` в ожиданиях.

    ``month_start`` — начало месяца, ``month_end`` — начало следующего (граница не входит
    в месяц: в ожидании ставь ``strict_max=True``). ISO-строки, а не ``Timestamp``: так они
    сериализуются в Data Docs, а GX сравнивает их с ``datetime64`` колонки корректно.

    Raises:
        ValueError: ``month`` не в формате ``YYYY-MM``.
    """
    start = pd.Timestamp(parse_month(month))
    end = start + pd.DateOffset(months=1)
    return {"month_start": start.isoformat(), "month_end": end.isoformat()}


def run_name(stage: Stage, month: str) -> str:
    """Имя прогона в Data Docs и хранилище результатов GX: ``raw_2024-01``."""
    return f"{stage.value}_{month}"


def summarize_result(
    result: Mapping[str, Any], *, stage: Stage, month: str, run_time: datetime
) -> ValidationReport:
    """Свести результат GX (``ExpectationSuiteValidationResult.to_json_dict()``) в отчёт.

    Ожидание считается проваленным, если ``success`` не ``True`` (в том числе когда GX
    поймал исключение при расчёте метрики, например колонки нет). Severity берётся из
    ``expectation_config.severity``, затем из ``meta.severity``; если её нет — ``critical``
    (как у GX). Неизвестное значение тоже трактуется как ``critical``: безопаснее
    остановить пайплайн, чем пропустить данные из-за опечатки в контракте.
    """
    results: list[Mapping[str, Any]] = list(result.get("results") or [])
    failures = tuple(_failure(item) for item in results if item.get("success") is not True)
    evaluated = len(results)
    failed_critical = sum(failure.blocking for failure in failures)
    passed = evaluated - len(failures)
    return ValidationReport(
        stage=stage.value,
        month=month,
        suite_name=str(result.get("suite_name") or stage.suite_name),
        success=failed_critical == 0,
        evaluated=evaluated,
        failed_critical=failed_critical,
        failed_warning=len(failures) - failed_critical,
        success_percent=round(100.0 * passed / evaluated, 2) if evaluated else 100.0,
        run_time=run_time.astimezone(UTC).isoformat(timespec="seconds"),
        failures=failures,
    )


def format_failures(failures: Iterable[ExpectationFailure]) -> str:
    """Список провалов одной строкой через ``; `` (для сообщения ``DataValidationError``)."""
    return "; ".join(failure.describe() for failure in failures)


def run_validation(
    df: pd.DataFrame,
    *,
    stage: Stage,
    month: str,
    project_root: Path,
    report_uri: str,
    storage_options: dict[str, Any] | None = None,
    update_docs: bool = True,
) -> ValidationReport:
    """Проверить ``df`` контрактом стадии, записать отчёт и применить политику severity.

    Отчёт пишется до решения quality gate: при провале он нужен больше всего.

    Args:
        df: данные месяца (сырые или подготовленные — по ``stage``).
        stage: какой набор ожиданий применять.
        month: проверяемый месяц ``YYYY-MM`` (границы для ожиданий на даты).
        project_root: каталог, где лежит ``gx/great_expectations.yml`` (корень репозитория).
        report_uri: куда записать JSON-отчёт (fsspec URI).
        storage_options: параметры fsspec для ``report_uri``.
        update_docs: перестраивать ли HTML Data Docs (в тестах можно выключить).

    Returns:
        Отчёт; ``success`` = true (провалы warning, если есть, — внутри).

    Raises:
        ConfigError: в ``project_root`` нет контекста GX.
        DataValidationError: провалено хотя бы одно critical-ожидание.
    """
    config_file = project_root / GX_CONFIG_RELPATH
    # Без проверки get_context(mode="file") молча создал бы новый пустой gx/ в чужом каталоге.
    if not config_file.is_file():
        raise ConfigError(f"GX context not found: {config_file} (run from the repository root)")

    # Ленивый импорт: GX нужен только этой команде, а не каждому запуску CLI.
    import great_expectations as gx
    from great_expectations.checkpoint.actions import UpdateDataDocsAction, ValidationAction
    from great_expectations.core.run_identifier import RunIdentifier

    context = gx.get_context(mode="file", project_root_dir=str(project_root))
    batch_definition = _batch_definition(context, stage)
    suite = context.suites.add_or_update(_build_suite(stage))
    validation_definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=stage.suite_name, data=batch_definition, suite=suite)
    )
    actions: list[ValidationAction] = (
        [UpdateDataDocsAction(name=DATA_DOCS_ACTION_NAME)] if update_docs else []
    )
    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name=stage.suite_name,
            validation_definitions=[validation_definition],
            actions=actions,
        )
    )

    run_time = datetime.now(UTC)
    logger.info("Validating %s %s: %d rows against suite %s", stage, month, len(df), suite.name)
    checkpoint_result = checkpoint.run(
        batch_parameters={"dataframe": df},
        expectation_parameters=suite_parameters(month),
        # Batch ID у dataframe-asset одинаковый для всех месяцев: без run_name в Data Docs
        # не видно, какой месяц провалился. То же время, что в JSON-отчёте.
        run_id=RunIdentifier(run_name=run_name(stage, month), run_time=run_time),
    )
    # Один validation definition на checkpoint → ровно один результат.
    (suite_result,) = checkpoint_result.run_results.values()
    # У GX этот метод без аннотаций типов; результат — JSON-совместимый dict.
    payload: Mapping[str, Any] = suite_result.to_json_dict()  # type: ignore[no-untyped-call]
    report = summarize_result(payload, stage=stage, month=month, run_time=run_time)

    write_json(report.to_dict(), report_uri, storage_options)
    _log_report(report, report_uri)
    if not report.success:
        raise DataValidationError(
            f"{stage} {month}: {report.failed_critical} critical expectation(s) failed: "
            f"{format_failures(report.critical_failures)} (report: {report_uri})"
        )
    return report


def _build_suite(stage: Stage) -> ExpectationSuite:
    """Собрать suite стадии из кода и проверить, что имя совпадает с ожидаемым."""
    from taxi_duration.validation import suites

    builders = {
        Stage.RAW: suites.build_raw_trips_suite,
        Stage.PREPARED: suites.build_prepared_trips_suite,
    }
    suite = builders[stage]()
    # Имя — ключ в хранилище GX: другое имя = новый suite рядом со старым (грабли design.md).
    if suite.name != stage.suite_name:
        raise ValueError(
            f"Suite for stage {stage} must be named {stage.suite_name!r}, got {suite.name!r}"
        )
    return suite


def _batch_definition(context: AbstractDataContext, stage: Stage) -> BatchDefinition[Any]:
    """Data Source → Asset → Batch Definition: взять существующие или создать.

    Не ``add_or_update``: он пересоздаёт объекты с новыми id и при каждом запуске
    переписывает закоммиченный ``gx/great_expectations.yml``. get-or-add меняет файл
    только при первой регистрации asset.
    """
    from great_expectations.datasource.fluent.pandas_datasource import (
        DataFrameAsset,
        PandasDatasource,
    )

    try:
        datasource = context.data_sources.get(DATASOURCE_NAME)
    except LookupError:
        datasource = context.data_sources.add_pandas(name=DATASOURCE_NAME)
    # Под тем же именем в gx/great_expectations.yml мог оказаться источник другого типа.
    if not isinstance(datasource, PandasDatasource):
        raise ConfigError(
            f"GX datasource {DATASOURCE_NAME!r} must be pandas, got {datasource.type}"
        )
    try:
        asset = datasource.get_asset(stage.suite_name)
    except LookupError:
        asset = datasource.add_dataframe_asset(stage.suite_name)
    if not isinstance(asset, DataFrameAsset):
        raise ConfigError(f"GX asset {stage.suite_name!r} must be a dataframe asset")
    batch_definition: BatchDefinition[Any]
    try:
        batch_definition = asset.get_batch_definition(BATCH_DEFINITION_NAME)
    except LookupError:
        batch_definition = asset.add_batch_definition_whole_dataframe(BATCH_DEFINITION_NAME)
    return batch_definition


def _failure(item: Mapping[str, Any]) -> ExpectationFailure:
    config: Mapping[str, Any] = item.get("expectation_config") or {}
    kwargs: Mapping[str, Any] = config.get("kwargs") or {}
    details: Mapping[str, Any] = item.get("result") or {}
    unexpected = details.get("unexpected_percent")
    return ExpectationFailure(
        expectation=str(config.get("type", "unknown")),
        severity=_severity(config),
        column=_column(kwargs),
        unexpected_percent=float(unexpected) if unexpected is not None else None,
        observed_value=details.get("observed_value"),
        error=_exception_message(item.get("exception_info")),
    )


def _severity(config: Mapping[str, Any]) -> str:
    meta: Mapping[str, Any] = config.get("meta") or {}
    raw = config.get("severity") or meta.get("severity") or CRITICAL
    severity = str(raw).lower()
    if severity not in _KNOWN_SEVERITIES:
        logger.warning(
            "Unknown severity %r in %s: treating as critical", raw, config.get("type", "unknown")
        )
        return CRITICAL
    return severity


def _column(kwargs: Mapping[str, Any]) -> str | None:
    if kwargs.get("column") is not None:
        return str(kwargs["column"])
    pair = [kwargs.get("column_A"), kwargs.get("column_B")]
    if all(name is not None for name in pair):
        return ",".join(str(name) for name in pair)
    return None


def _exception_message(info: Any) -> str | None:
    """Текст исключения из ``exception_info``.

    GX кладёт его в двух формах: плоский словарь ``{raised_exception, exception_message, ...}``
    или словарь «id метрики → такой же словарь», когда упал расчёт конкретной метрики.
    """
    if not isinstance(info, Mapping) or not info:
        return None
    candidates = [info] if "raised_exception" in info else list(info.values())
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("raised_exception"):
            message = str(candidate.get("exception_message") or "unknown error")
            return message[:_MAX_ERROR_CHARS]
    return None


def _log_report(report: ValidationReport, report_uri: str) -> None:
    for failure in report.warning_failures:
        logger.warning("Expectation failed (non-blocking): %s", failure.describe())
    for failure in report.critical_failures:
        logger.error("Expectation failed (critical): %s", failure.describe())
    logger.info(
        "Validation %s %s: success=%s evaluated=%d failed_critical=%d failed_warning=%d -> %s",
        report.stage,
        report.month,
        report.success,
        report.evaluated,
        report.failed_critical,
        report.failed_warning,
        report_uri,
    )
