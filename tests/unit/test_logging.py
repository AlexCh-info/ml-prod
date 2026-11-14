"""Сценарии спеки structured-logging. Контракт для setup_logging() (задача 3.2 [YOU])."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import pytest

from taxi_duration.config import LoggingSettings, Settings
from taxi_duration.errors import ConfigError
from taxi_duration.logging_setup import setup_logging

LOGGER_NAME = "taxi_duration.data.ingest"


def make_settings(
    repo_root: Path,
    *,
    fmt: Literal["text", "json"] = "text",
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    config_path: Path | None = None,
) -> Settings:
    return Settings(
        logging=LoggingSettings(
            format=fmt,
            level=level,
            config_path=config_path or repo_root / "configs" / "logging.yaml",
        )
    )


def test_json_format_emits_one_valid_json_line(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setup_logging(make_settings(repo_root, fmt="json"))

    logging.getLogger(LOGGER_NAME).info("loaded", extra={"rows": 10})

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert {"timestamp", "level", "logger", "message", "rows"} <= record.keys()
    assert record["level"] == "INFO"
    assert record["logger"] == LOGGER_NAME
    assert record["message"] == "loaded"
    assert record["rows"] == 10


def test_text_format_contains_time_level_logger_message(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setup_logging(make_settings(repo_root, fmt="text"))

    logging.getLogger(LOGGER_NAME).info("loaded")

    line = capsys.readouterr().out.strip()
    assert "INFO" in line
    assert LOGGER_NAME in line
    assert "loaded" in line
    assert line[:4].isdigit()  # строка начинается с времени: 2026-...
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)


def test_level_warning_hides_info(repo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(make_settings(repo_root, level="WARNING"))

    log = logging.getLogger(LOGGER_NAME)
    log.info("hidden")
    log.warning("shown")

    out = capsys.readouterr().out
    assert "hidden" not in out
    assert "shown" in out


def test_level_debug_shows_debug(repo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(make_settings(repo_root, level="DEBUG"))

    logging.getLogger(LOGGER_NAME).debug("details")

    assert "details" in capsys.readouterr().out


def test_logger_created_before_setup_still_works(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Модули создают логгер при импорте — до вызова setup_logging().
    # Наши логгеры (потомки taxi_duration) dictConfig не отключает в любом случае,
    # а вот чужие (библиотеки) при disable_existing_loggers: true замолчали бы целиком.
    early = logging.getLogger("taxi_duration.created_before_setup")
    library = logging.getLogger("some_library.client")
    setup_logging(make_settings(repo_root))

    early.info("still alive")
    library.warning("library warning")

    out = capsys.readouterr().out
    assert "still alive" in out
    assert "library warning" in out


def test_repeated_setup_does_not_duplicate_output(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(repo_root)
    setup_logging(settings)
    setup_logging(settings)

    logging.getLogger(LOGGER_NAME).info("once")

    assert capsys.readouterr().out.count("once") == 1


def test_missing_logging_config_raises_config_error(repo_root: Path, tmp_path: Path) -> None:
    missing = tmp_path / "no-logging.yaml"
    with pytest.raises(ConfigError, match="no-logging.yaml"):
        setup_logging(make_settings(repo_root, config_path=missing))


def test_setup_does_not_mutate_template_between_calls(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Сначала json, потом text: второй вызов не должен «залипнуть» на первом формате.
    setup_logging(make_settings(repo_root, fmt="json"))
    setup_logging(make_settings(repo_root, fmt="text"))

    logging.getLogger(LOGGER_NAME).info("plain")

    line = capsys.readouterr().out.strip()
    assert "plain" in line
    assert not line.startswith("{")
