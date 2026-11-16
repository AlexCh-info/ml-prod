"""Работа с путями данных как с fsspec URI.

Один и тот же код читает и пишет и локальный ``data/...``, и ``s3://lake/...`` (Airflow,
этап 07). Разница только в ``storage_options``: для S3 нужны endpoint RustFS и ключи.
"""

from __future__ import annotations

from typing import Any

from taxi_duration.config import S3Settings

RAW_DIR = "raw"
PROCESSED_DIR = "processed"
FEATURES_DIR = "features"


def join_uri(root: str, *parts: str) -> str:
    """Склеить URI через ``/``; в отличие от ``os.path.join`` не трогает ``s3://``."""
    return "/".join([root.rstrip("/"), *(part.strip("/") for part in parts)])


def raw_month_uri(data_root: str, month: str) -> str:
    """``<data_root>/raw/green_tripdata_YYYY-MM.parquet`` — имя как у источника TLC."""
    return join_uri(data_root, RAW_DIR, f"green_tripdata_{month}.parquet")


def processed_month_uri(data_root: str, month: str) -> str:
    """``<data_root>/processed/trips_YYYY-MM.parquet`` — очищенные поездки месяца."""
    return join_uri(data_root, PROCESSED_DIR, f"trips_{month}.parquet")


def zone_stats_uri(data_root: str) -> str:
    """``<data_root>/features/zone_stats.parquet`` — признаки зон по всем месяцам."""
    return join_uri(data_root, FEATURES_DIR, "zone_stats.parquet")


def storage_options(uri: str, s3: S3Settings) -> dict[str, Any]:
    """Параметры fsspec для ``uri``: для ``s3://`` — endpoint и ключи, иначе пусто.

    Если ключи не заданы, s3fs ищет их сам по цепочке boto (env ``AWS_*``, ``~/.aws``).
    """
    if not uri.startswith("s3://"):
        return {}
    options: dict[str, Any] = {"client_kwargs": {"endpoint_url": s3.endpoint_url}}
    if s3.access_key_id is not None:
        options["key"] = s3.access_key_id.get_secret_value()
    if s3.secret_access_key is not None:
        options["secret"] = s3.secret_access_key.get_secret_value()
    return options
