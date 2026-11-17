"""Командная точка входа ``taxi``.

Через неё код проекта вызывают человек, DVC, Airflow, Docker и CI. Контракт:

- код возврата 0 — успех;
- ожидаемая ошибка проекта (``TaxiError``) — одна строка ``Error: ...`` в stderr и код 1;
- неожиданное исключение (баг) — запись в лог с трассировкой и код 1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
import yaml
from typer.core import TyperGroup
from typer.exceptions import TyperException

from taxi_duration import __version__
from taxi_duration.config import Settings, load_settings, parse_month
from taxi_duration.data.ingest import ingest_month
from taxi_duration.data.prepare import prepare_trips
from taxi_duration.errors import DataError, TaxiError
from taxi_duration.features.zone_stats import build_zone_stats
from taxi_duration.logging_setup import setup_logging
from taxi_duration.storage import (
    processed_month_uri,
    raw_month_uri,
    read_parquet,
    storage_options,
    validation_report_uri,
    write_parquet,
    zone_stats_uri,
)
from taxi_duration.validation.runner import Stage, run_validation

logger = logging.getLogger(__name__)

# Сигнальные исключения самого typer (--help, ошибки аргументов, typer.Exit):
# у них своя обработка и свои коды возврата, их не трогаем.
_FRAMEWORK_EXCEPTIONS = (typer.Exit, typer.Abort, TyperException)

# Команды, которым не нужны конфиг и логирование: должны работать даже при сломанном конфиге.
_NO_SETUP_COMMANDS = frozenset({"version"})


class _TaxiGroup(TyperGroup):
    """Корневая группа команд с единой обработкой ошибок.

    ``invoke`` выполняет и callback, и выбранную подкоманду (включая вложенные
    группы вроде ``config show``), поэтому одного ``try`` здесь достаточно для всего CLI.
    """

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except _FRAMEWORK_EXCEPTIONS:
            raise
        except TaxiError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            # Если логирование ещё не настроено, сработает logging.lastResort (stderr).
            logger.exception("Unexpected error: %s", exc)
            raise typer.Exit(code=1) from exc


app = typer.Typer(
    cls=_TaxiGroup,
    name="taxi",
    help="taxi-duration: MLOps-пайплайн прогноза длительности поездки NYC Green Taxi.",
    no_args_is_help=True,
    add_completion=False,
    # Не печатать локальные переменные в трассировке: там могут быть секреты из Settings.
    pretty_exceptions_show_locals=False,
)

config_app = typer.Typer(help="Работа с конфигурацией.", no_args_is_help=True)
app.add_typer(config_app, name="config")


@app.callback()
def main(ctx: typer.Context) -> None:
    """Загрузить конфиг и настроить логирование перед любой подкомандой."""
    if ctx.invoked_subcommand in _NO_SETUP_COMMANDS:
        return
    settings = load_settings()
    setup_logging(settings)
    ctx.obj = settings
    logger.debug("CLI started: subcommand=%s", ctx.invoked_subcommand)


def _settings(ctx: typer.Context) -> Settings:
    """Настройки, загруженные корневым callback (ctx.obj наследуется вложенными контекстами)."""
    settings = ctx.find_root().obj
    if not isinstance(settings, Settings):  # pragma: no cover - нарушение инварианта callback
        raise RuntimeError("Settings are not initialised: root callback did not run")
    return settings


@app.command()
def version() -> None:
    """Показать версию пакета."""
    typer.echo(__version__)


def _validate_month(value: str) -> str:
    """Callback опции ``--month``: неверный формат — ошибка использования (код 2)."""
    try:
        parse_month(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return value


MonthOption = Annotated[
    str, typer.Option("--month", help="Месяц в формате YYYY-MM.", callback=_validate_month)
]


@app.command()
def ingest(
    ctx: typer.Context,
    month: MonthOption,
    force: Annotated[
        bool, typer.Option("--force", help="Скачать заново, даже если файл уже есть.")
    ] = False,
) -> None:
    """Скачать месячный parquet NYC TLC Green Taxi в <data_root>/raw/."""
    settings = _settings(ctx)
    data_root = settings.paths.data_root
    ingest_month(
        month,
        data_root=data_root,
        data=settings.data,
        storage_options=storage_options(data_root, settings.s3),
        force=force,
    )


@app.command()
def prepare(ctx: typer.Context, month: MonthOption) -> None:
    """Очистить сырые поездки месяца и посчитать таргет: raw/ → processed/."""
    settings = _settings(ctx)
    data_root = settings.paths.data_root
    options = storage_options(data_root, settings.s3)
    source = raw_month_uri(data_root, month)
    target = processed_month_uri(data_root, month)

    raw = read_parquet(source, options, hint=f"run `taxi ingest --month {month}` first")
    trips = prepare_trips(raw, month, settings.data.filters)
    write_parquet(trips, target, options)
    logger.info(
        "Prepared %s: %d -> %d rows (%d dropped) -> %s",
        month,
        len(raw),
        len(trips),
        len(raw) - len(trips),
        target,
    )


@app.command("build-features")
def build_features(ctx: typer.Context) -> None:
    """Посчитать признаки зон по всем месяцам из data.months: processed/ → features/."""
    settings = _settings(ctx)
    data_root = settings.paths.data_root
    options = storage_options(data_root, settings.s3)
    months = settings.data.months
    if not months:
        raise DataError("data.months is empty: nothing to build features from")

    trips = pd.concat(
        [
            read_parquet(
                processed_month_uri(data_root, month),
                options,
                hint=f"run `taxi prepare --month {month}` first",
            )
            for month in months
        ],
        ignore_index=True,
    )
    stats = build_zone_stats(trips, settings.features.window_days)
    target = zone_stats_uri(data_root)
    write_parquet(stats, target, options)
    logger.info(
        "Built zone stats for %s: %d trips -> %d rows (%d zones) -> %s",
        ", ".join(months),
        len(trips),
        len(stats),
        stats["pu_zone"].nunique(),
        target,
    )


@app.command()
def validate(
    ctx: typer.Context,
    month: MonthOption,
    stage: Annotated[
        Stage, typer.Option("--stage", help="Какие данные проверять: raw/ или processed/.")
    ],
    no_docs: Annotated[
        bool, typer.Option("--no-docs", help="Не перестраивать HTML Data Docs.")
    ] = False,
) -> None:
    """Проверить данные месяца контрактом GX: critical-провал → код 1, отчёт в reports/validation/.

    Запускать из корня репозитория: там лежит контекст gx/.
    """
    settings = _settings(ctx)
    data_root = settings.paths.data_root
    if stage is Stage.RAW:
        source, hint = raw_month_uri(data_root, month), f"run `taxi ingest --month {month}` first"
    else:
        source = processed_month_uri(data_root, month)
        hint = f"run `taxi prepare --month {month}` first"
    df = read_parquet(source, storage_options(data_root, settings.s3), hint=hint)

    reports_dir = settings.paths.reports_dir
    run_validation(
        df,
        stage=stage,
        month=month,
        project_root=Path.cwd(),
        report_uri=validation_report_uri(reports_dir, stage.value, month),
        storage_options=storage_options(reports_dir, settings.s3),
        update_docs=not no_docs,
    )


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Показать итоговый конфиг (YAML + .env + env) с замаскированными секретами."""
    typer.echo(yaml.safe_dump(_settings(ctx).to_safe_dict(), sort_keys=False, allow_unicode=True))
