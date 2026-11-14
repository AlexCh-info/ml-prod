"""Типизированная конфигурация приложения.

Источники настроек (сверху — выше приоритет, первый нашедший значение побеждает):

1. аргументы ``Settings(...)`` (init) — для тестов и явных переопределений в коде;
2. переменные окружения ``TAXI__<СЕКЦИЯ>__<ПОЛЕ>`` — Docker, K8s, CI;
3. файл ``.env`` в текущем каталоге — локальные секреты разработчика;
4. secrets-каталог (Docker/K8s secrets), если задан;
5. YAML-файл из ``TAXI_CONFIG`` (по умолчанию ``configs/base.yaml``);
6. значения по умолчанию в моделях ниже.

Вложенные секции объединяются по полям: env может переопределить одно поле
секции, остальные придут из YAML.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from taxi_duration.errors import ConfigError

logger = logging.getLogger(__name__)

CONFIG_ENV_VAR = "TAXI_CONFIG"
DEFAULT_CONFIG_PATH = Path("configs/base.yaml")

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class _Section(BaseModel):
    """База для секций: неизвестные ключи в YAML — ошибка, а не тихий игнор опечатки."""

    model_config = ConfigDict(extra="forbid")


class AppSettings(_Section):
    name: str = "taxi-duration"
    environment: Literal["local", "dev", "prod"] = "local"
    seed: int = 42


class LoggingSettings(_Section):
    level: LogLevel = "INFO"
    format: Literal["text", "json"] = "text"
    config_path: Path = Path("configs/logging.yaml")

    @field_validator("level", mode="before")
    @classmethod
    def _upper_level(cls, value: Any) -> Any:
        # Разрешаем TAXI__LOGGING__LEVEL=debug: в env регистр легко перепутать.
        return value.upper() if isinstance(value, str) else value


class PathsSettings(_Section):
    # Строки, а не Path: это fsspec URI (data или s3://lake/...), Path сломал бы "s3://".
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"


class MlflowSettings(_Section):
    tracking_uri: str = "http://localhost:5001"


class S3Settings(_Section):
    endpoint_url: str = "http://localhost:9000"
    # SecretStr: repr/str/JSON-дамп показывают "**********" вместо значения.
    access_key_id: SecretStr | None = None
    secret_access_key: SecretStr | None = None


def config_path() -> Path:
    """Путь к YAML-конфигу: ``$TAXI_CONFIG`` или ``configs/base.yaml`` относительно CWD."""
    return Path(os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TAXI__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    mlflow: MlflowSettings = Field(default_factory=MlflowSettings)
    s3: S3Settings = Field(default_factory=S3Settings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Порядок в кортеже = приоритет. YAML последним: это «база», которую
        # окружение переопределяет без правки файлов (12-factor).
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=config_path())
        return init_settings, env_settings, dotenv_settings, file_secret_settings, yaml_settings

    def to_safe_dict(self) -> dict[str, Any]:
        """JSON-совместимый словарь, где секреты заменены на ``**********``."""
        return self.model_dump(mode="json")


def _format_validation_error(exc: ValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()
    )
    return f"Invalid configuration: {problems}"


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Загрузить и проверить конфиг один раз за процесс.

    Raises:
        ConfigError: файл не найден, YAML не парсится или значения не проходят валидацию.
    """
    path = config_path()
    # YamlConfigSettingsSource молча пропускает отсутствующий файл —
    # для нас это ошибка: опечатка в TAXI_CONFIG не должна давать «конфиг из дефолтов».
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        settings = Settings()
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Cannot parse YAML config {path}: {exc}") from exc

    logger.debug("Configuration loaded from %s", path)
    return settings
