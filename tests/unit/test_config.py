"""Сценарии спеки runtime-config."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from taxi_duration.config import load_settings
from taxi_duration.errors import ConfigError, TaxiError

WriteConfig = Callable[..., Path]

SAMPLE_YAML = """
app:
  name: from-yaml
  seed: 7
logging:
  level: WARNING
mlflow:
  tracking_uri: http://yaml-mlflow:5001
"""


def test_repo_base_yaml_is_valid(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    # Конфиг по умолчанию из репозитория должен проходить валидацию.
    monkeypatch.chdir(repo_root)
    settings = load_settings()
    assert settings.app.name == "taxi-duration"
    assert settings.mlflow.tracking_uri == "http://localhost:5001"


def test_values_loaded_from_yaml(write_config: WriteConfig) -> None:
    write_config(SAMPLE_YAML)
    settings = load_settings()
    assert settings.app.name == "from-yaml"
    assert settings.app.seed == 7
    assert settings.logging.level == "WARNING"
    # Поле, которого нет в YAML, берётся из дефолта модели.
    assert settings.logging.format == "text"


def test_env_overrides_nested_yaml_field(
    write_config: WriteConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(SAMPLE_YAML)
    monkeypatch.setenv("TAXI__MLFLOW__TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("TAXI__LOGGING__LEVEL", "debug")
    settings = load_settings()
    assert settings.mlflow.tracking_uri == "http://mlflow:5000"
    assert settings.logging.level == "DEBUG"
    # Остальные поля секции остаются из YAML: секции сливаются по полям.
    assert settings.app.name == "from-yaml"


def test_dotenv_overrides_yaml_but_env_wins(
    write_config: WriteConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_config(SAMPLE_YAML)
    (tmp_path / ".env").write_text("TAXI__APP__SEED=100\nTAXI__APP__NAME=from-dotenv\n")
    monkeypatch.setenv("TAXI__APP__NAME", "from-env")
    settings = load_settings()
    assert settings.app.seed == 100
    assert settings.app.name == "from-env"


def test_wrong_type_raises_config_error_with_field_name(write_config: WriteConfig) -> None:
    write_config("app:\n  seed: abc\n")
    with pytest.raises(ConfigError, match=r"app\.seed"):
        load_settings()


def test_unknown_key_is_rejected(write_config: WriteConfig) -> None:
    write_config("loging:\n  level: DEBUG\n")
    with pytest.raises(ConfigError, match="loging"):
        load_settings()


def test_missing_file_raises_config_error_with_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "nope.yaml"
    monkeypatch.setenv("TAXI_CONFIG", str(missing))
    with pytest.raises(ConfigError, match=r"nope\.yaml"):
        load_settings()


def test_broken_yaml_raises_config_error(write_config: WriteConfig) -> None:
    write_config("app: [unclosed\n")
    with pytest.raises(ConfigError, match="Cannot parse YAML"):
        load_settings()


def test_config_error_is_taxi_error() -> None:
    assert issubclass(ConfigError, TaxiError)


def test_secrets_are_masked(write_config: WriteConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    write_config(SAMPLE_YAML)
    monkeypatch.setenv("TAXI__S3__ACCESS_KEY_ID", "minioadmin")
    monkeypatch.setenv("TAXI__S3__SECRET_ACCESS_KEY", "super-secret")
    settings = load_settings()

    # Само значение доступно коду явно...
    assert settings.s3.secret_access_key is not None
    assert settings.s3.secret_access_key.get_secret_value() == "super-secret"

    # ...но не утекает ни в дамп, ни в repr.
    dumped = json.dumps(settings.to_safe_dict())
    assert "super-secret" not in dumped
    assert "minioadmin" not in dumped
    assert settings.to_safe_dict()["s3"]["secret_access_key"] == "**********"
    assert "super-secret" not in repr(settings)


def test_settings_are_cached(write_config: WriteConfig) -> None:
    write_config(SAMPLE_YAML)
    assert load_settings() is load_settings()
