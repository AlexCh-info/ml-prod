"""Сценарии спеки cli (и сквозные сценарии runtime-config через `taxi config show`)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from taxi_duration import __version__, cli
from taxi_duration.cli import app

WriteConfig = Callable[..., Path]

runner = CliRunner()


@pytest.fixture(autouse=True)
def logging_template(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    # Тесты работают в tmp_path, а logging.config_path по умолчанию относительный.
    monkeypatch.setenv("TAXI__LOGGING__CONFIG_PATH", str(repo_root / "configs" / "logging.yaml"))


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.stdout
    assert "config" in result.stdout


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_version_works_without_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TAXI_CONFIG", str(tmp_path / "nope.yaml"))
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0


def test_config_show_uses_repo_base_yaml(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    monkeypatch.chdir(repo_root)
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.output
    shown = yaml.safe_load(result.stdout)
    assert shown["app"]["name"] == "taxi-duration"
    assert shown["mlflow"]["tracking_uri"] == "http://localhost:5001"


def test_config_show_uses_alternative_file(write_config: WriteConfig) -> None:
    write_config("app:\n  name: alt-config\n")
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.stdout)["app"]["name"] == "alt-config"


def test_config_show_applies_env_override(
    write_config: WriteConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config("mlflow:\n  tracking_uri: http://from-yaml:5001\n")
    monkeypatch.setenv("TAXI__MLFLOW__TRACKING_URI", "http://mlflow:5000")
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.stdout)["mlflow"]["tracking_uri"] == "http://mlflow:5000"


def test_config_show_masks_secrets(
    write_config: WriteConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config("app:\n  name: x\n")
    monkeypatch.setenv("TAXI__S3__SECRET_ACCESS_KEY", "super-secret")
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert "super-secret" not in result.output
    assert yaml.safe_load(result.stdout)["s3"]["secret_access_key"] == "**********"


def test_missing_config_gives_one_line_error_and_exit_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TAXI_CONFIG", str(tmp_path / "nope.yaml"))
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 1
    lines = result.stderr.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("Error:")
    assert "nope.yaml" in lines[0]
    assert "Traceback" not in result.output


def test_wrong_type_gives_error_with_field_name(write_config: WriteConfig) -> None:
    write_config("app:\n  seed: abc\n")
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 1
    assert "app.seed" in result.stderr
    assert "Traceback" not in result.output


def test_unexpected_error_is_logged_with_traceback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom() -> None:
        raise RuntimeError("bug in our code")

    monkeypatch.setattr(cli, "load_settings", boom)
    with caplog.at_level(logging.ERROR, logger="taxi_duration"):
        result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 1
    records = [r for r in caplog.records if r.name == "taxi_duration.cli"]
    assert records, "unexpected error must be logged"
    assert records[0].exc_info is not None  # трассировка прикреплена к записи
    assert "bug in our code" in records[0].getMessage()


def test_unknown_command_is_usage_error() -> None:
    result = runner.invoke(app, ["nope"])
    assert result.exit_code == 2  # код click для ошибок использования, не наш 1
