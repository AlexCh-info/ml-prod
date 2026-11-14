"""Командная точка входа ``taxi``.

Через неё код проекта вызывают человек, DVC, Airflow, Docker и CI. Контракт:

- код возврата 0 — успех;
- ожидаемая ошибка проекта (``TaxiError``) — одна строка ``Error: ...`` в stderr и код 1;
- неожиданное исключение (баг) — запись в лог с трассировкой и код 1.
"""

from __future__ import annotations

import logging
from typing import Any

import typer
import yaml
from typer.core import TyperGroup
from typer.exceptions import TyperException

from taxi_duration import __version__
from taxi_duration.config import Settings, load_settings
from taxi_duration.errors import TaxiError
from taxi_duration.logging_setup import setup_logging

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


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Показать итоговый конфиг (YAML + .env + env) с замаскированными секретами."""
    typer.echo(yaml.safe_dump(_settings(ctx).to_safe_dict(), sort_keys=False, allow_unicode=True))
