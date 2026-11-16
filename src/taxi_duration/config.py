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
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
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

_MONTH_RE = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


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
    data_root: str = "data"
    artifacts_dir: str = "artifacts"


class FilterSettings(_Section):
    """Пороги очистки поездок (этап prepare). Ключ ``data.filters`` — параметр DVC-стадии."""

    min_duration: float = Field(default=1.0, ge=0)
    max_duration: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _check_range(self) -> FilterSettings:
        if self.min_duration >= self.max_duration:
            raise ValueError(
                f"min_duration ({self.min_duration}) must be less than "
                f"max_duration ({self.max_duration})"
            )
        return self


class DownloadSettings(_Section):
    """Сетевые параметры загрузки сырых файлов (этап ingest)."""

    timeout_s: float = Field(default=60.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)  # всего попыток, включая первую
    backoff_base_s: float = Field(default=1.0, ge=0)  # пауза перед повтором: base * 2**(n-1)


class DataSettings(_Section):
    # {month} подставляется как YYYY-MM.
    url_template: str = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{month}.parquet"
    )
    # Месяцы, которые собирает DVC-пайплайн (foreach в dvc.yaml).
    months: list[str] = Field(default_factory=lambda: ["2024-01"])
    filters: FilterSettings = Field(default_factory=FilterSettings)
    download: DownloadSettings = Field(default_factory=DownloadSettings)

    @field_validator("url_template")
    @classmethod
    def _template_has_month(cls, value: str) -> str:
        if "{month}" not in value:
            raise ValueError("url_template must contain the {month} placeholder")
        return value

    @field_validator("months")
    @classmethod
    def _valid_months(cls, value: list[str]) -> list[str]:
        for month in value:
            parse_month(month)
        if len(set(value)) != len(value):
            raise ValueError(f"months must be unique: {value}")
        return value


def parse_month(month: str) -> date:
    """Проверить строку ``YYYY-MM`` и вернуть первый день месяца.

    Raises:
        ValueError: формат не ``YYYY-MM`` или месяц вне 01..12.
    """
    if not _MONTH_RE.fullmatch(month):
        raise ValueError(f"month must be in YYYY-MM format, got {month!r}")
    return date(int(month[:4]), int(month[5:]), 1)


class MlflowSettings(_Section):
    tracking_uri: str = "http://localhost:5001"


class S3BucketsSettings(_Section):
    """Имена buckets; их создаёт сервис rustfs-init в docker-compose.yml."""

    mlflow: str = "mlflow"  # артефакты MLflow (пишет только MLflow server)
    dvc: str = "dvc"  # remote DVC
    lake: str = "lake"  # данные для Airflow
    reports: str = "reports"  # отчёты GX / Evidently


class S3Settings(_Section):
    # С хоста http://localhost:9000, из контейнеров TAXI__S3__ENDPOINT_URL=http://rustfs:9000.
    endpoint_url: str = "http://localhost:9000"
    buckets: S3BucketsSettings = Field(default_factory=S3BucketsSettings)
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
        # .env общий с docker compose (POSTGRES_PASSWORD, RUSTFS_ACCESS_KEY, ...).
        # match_prefix: из .env берутся только TAXI__*, чужие ключи не ошибка,
        # а опечатка в TAXI__* по-прежнему ловится через extra="forbid".
        dotenv_filtering="match_prefix",
        extra="forbid",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    data: DataSettings = Field(default_factory=DataSettings)
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
