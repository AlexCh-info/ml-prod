"""Загрузка месячного parquet NYC TLC Green Taxi в ``<data_root>/raw/``.

Гарантии (спека data-ingestion):

- **атомарность**: файл сначала качается во временный ``.part``, затем переименовывается
  (``os.replace`` локально, одна PUT-операция для S3). Оборванная загрузка не оставляет
  файла по целевому пути, и DVC не запомнит хеш мусора;
- **повторы**: сетевые ошибки и 5xx повторяются до ``max_attempts`` раз с паузой
  ``backoff_base_s * 2**(n-1)``; 404 означает «месяц не опубликован» и не повторяется;
- **идемпотентность**: существующий файл без ``force`` не скачивается заново
  (важно для повторных запусков задач Airflow на этапе 07).
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

from taxi_duration import __version__
from taxi_duration.config import DataSettings, DownloadSettings, parse_month
from taxi_duration.errors import DataError
from taxi_duration.storage import raw_month_uri

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1 << 20  # 1 MiB: память не зависит от размера файла
_FILE_MODE = 0o644  # rw-r--r--: как у обычного файла при umask 022


class _RetryableError(Exception):
    """Ошибка, после которой есть смысл повторить запрос (сеть, 5xx)."""


def make_client(download: DownloadSettings) -> httpx.Client:
    """HTTP-клиент для источника. Вынесен в функцию, чтобы тесты подменяли транспорт."""
    return httpx.Client(
        timeout=download.timeout_s,
        follow_redirects=True,
        headers={"User-Agent": f"taxi-duration/{__version__}"},
    )


def ingest_month(
    month: str,
    *,
    data_root: str,
    data: DataSettings,
    storage_options: dict[str, Any] | None = None,
    force: bool = False,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Скачать месяц ``YYYY-MM`` в ``<data_root>/raw/green_tripdata_YYYY-MM.parquet``.

    Args:
        month: месяц в формате ``YYYY-MM``.
        data_root: корень данных (fsspec URI: ``data`` или ``s3://lake``).
        data: секция ``data`` конфига (URL-шаблон и параметры загрузки).
        storage_options: параметры fsspec для ``data_root`` (endpoint и ключи S3).
        force: скачать заново, даже если файл уже есть.
        client: HTTP-клиент; по умолчанию создаётся ``make_client``.
        sleep: функция паузы между повторами (в тестах подменяется, чтобы не ждать).

    Returns:
        URI целевого файла.

    Raises:
        DataError: месяц не опубликован (404), ответ не parquet, неожиданный 4xx
            или загрузка не удалась после всех попыток.
    """
    try:
        parse_month(month)
    except ValueError as exc:
        raise DataError(str(exc)) from exc

    target = raw_month_uri(data_root, month)
    fs, target_path = url_to_fs(target, **(storage_options or {}))

    if fs.exists(target_path) and not force:
        logger.info("Raw file for %s already exists, skipping download: %s", month, target)
        return target

    url = data.url_template.format(month=month)
    logger.info("Downloading %s -> %s", url, target)

    with nullcontext(client) if client is not None else make_client(data.download) as http:
        _download_with_retries(
            http, url, month, data.download, fs, target_path, is_local=_is_local(fs), sleep=sleep
        )

    logger.info("Saved %s (%d bytes)", target, fs.size(target_path))
    return target


def _is_local(fs: Any) -> bool:
    return isinstance(fs, LocalFileSystem)


def _download_with_retries(
    client: httpx.Client,
    url: str,
    month: str,
    download: DownloadSettings,
    fs: Any,
    target_path: str,
    *,
    is_local: bool,
    sleep: Callable[[float], None],
) -> None:
    for attempt in range(1, download.max_attempts + 1):
        try:
            _download_once(client, url, month, fs, target_path, is_local=is_local)
            return
        except _RetryableError as exc:
            if attempt == download.max_attempts:
                raise DataError(
                    f"Download of {month} failed after {attempt} attempts: {exc}"
                ) from exc
            delay = download.backoff_base_s * 2 ** (attempt - 1)
            logger.warning(
                "Attempt %d/%d for %s failed: %s; retrying in %.1fs",
                attempt,
                download.max_attempts,
                month,
                exc,
                delay,
            )
            sleep(delay)


def _download_once(
    client: httpx.Client,
    url: str,
    month: str,
    fs: Any,
    target_path: str,
    *,
    is_local: bool,
) -> None:
    """Одна попытка: поток в ``.part``, проверка parquet, атомарная публикация."""
    # Локально .part лежит рядом с целью: os.replace атомарен только в пределах одной ФС.
    # Для S3 качаем во временный каталог ОС и публикуем одним PUT (объект появляется целиком).
    if is_local:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_dir: str | None = str(Path(target_path).parent)
    else:
        tmp_dir = None
    fd, tmp_name = tempfile.mkstemp(
        dir=tmp_dir, prefix=f".{Path(target_path).name}.", suffix=".part"
    )
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            _stream_to(client, url, month, tmp_file)
        _check_parquet(tmp_name, month)
        if is_local:
            # mkstemp создаёт файл с правами 0600, а os.replace их сохраняет:
            # данные стали бы нечитаемы для других пользователей (например, в контейнере).
            os.chmod(tmp_name, _FILE_MODE)
            os.replace(tmp_name, target_path)
        else:
            fs.put_file(tmp_name, target_path)
    finally:
        # После успешного os.replace файла уже нет; во всех остальных случаях — убираем.
        Path(tmp_name).unlink(missing_ok=True)


def _stream_to(client: httpx.Client, url: str, month: str, out: Any) -> None:
    try:
        with client.stream("GET", url) as response:
            status = response.status_code
            if status == httpx.codes.NOT_FOUND:
                raise DataError(f"Month {month} is not published at the source (404): {url}")
            if status >= httpx.codes.INTERNAL_SERVER_ERROR:
                raise _RetryableError(f"HTTP {status}")
            if status != httpx.codes.OK:
                raise DataError(f"Unexpected HTTP {status} for {month}: {url}")
            for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                out.write(chunk)
    except httpx.TransportError as exc:
        # Таймауты, обрыв соединения, неполное тело ответа и т. п.
        raise _RetryableError(f"{type(exc).__name__}: {exc}") from exc


def _check_parquet(path: str, month: str) -> None:
    """Убедиться, что скачан именно parquet: читается footer с метаданными."""
    try:
        metadata = pq.read_metadata(path)
    except Exception as exc:  # pyarrow бросает ArrowInvalid/OSError на мусор
        raise DataError(f"Downloaded file for {month} is not a valid parquet: {exc}") from exc
    logger.info("Downloaded %s: %d rows", month, metadata.num_rows)
