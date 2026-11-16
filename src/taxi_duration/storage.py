"""Работа с путями данных как с fsspec URI.

Один и тот же код читает и пишет и локальный ``data/...``, и ``s3://lake/...`` (Airflow,
этап 07). Разница только в ``storage_options``: для S3 нужны endpoint RustFS и ключи.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

from taxi_duration.config import S3Settings
from taxi_duration.errors import DataError

_FILE_MODE = 0o644  # mkstemp даёт 0600; данные должны читать и другие пользователи

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


def read_parquet(
    uri: str, storage_options: dict[str, Any] | None = None, *, hint: str = ""
) -> pd.DataFrame:
    """Прочитать parquet по URI.

    Raises:
        DataError: файла нет (в сообщении — ``hint``, например какую команду запустить).
    """
    fs, path = url_to_fs(uri, **(storage_options or {}))
    if not fs.exists(path):
        suffix = f" ({hint})" if hint else ""
        raise DataError(f"Input file not found: {uri}{suffix}")
    return pd.read_parquet(uri, storage_options=storage_options or None)


def write_parquet(
    df: pd.DataFrame, uri: str, storage_options: dict[str, Any] | None = None
) -> None:
    """Записать parquet атомарно: читатель видит либо старый файл, либо новый целиком.

    Локально — временный файл рядом с целью и ``os.replace``; для S3 — одна загрузка
    объекта (он появляется только после её завершения).
    """
    fs, path = url_to_fs(uri, **(storage_options or {}))
    if not isinstance(fs, LocalFileSystem):
        df.to_parquet(uri, index=False, storage_options=storage_options or None)
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".part")
    os.close(fd)
    try:
        df.to_parquet(tmp_name, index=False)
        os.chmod(tmp_name, _FILE_MODE)
        os.replace(tmp_name, target)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
