"""Сценарии спеки data-ingestion. Сеть не нужна: источник подменяет httpx.MockTransport."""

from __future__ import annotations

import io
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import fsspec
import httpx
import pandas as pd
import pytest
from typer.testing import CliRunner

from taxi_duration.cli import app
from taxi_duration.config import DataSettings, DownloadSettings
from taxi_duration.data import ingest
from taxi_duration.data.ingest import ingest_month
from taxi_duration.errors import DataError

MONTH = "2024-01"
URL = f"https://source.test/green_tripdata_{MONTH}.parquet"
Handler = Callable[[httpx.Request], httpx.Response]

runner = CliRunner()


def _parquet_bytes(rows: int = 3) -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame({"PULocationID": range(rows)}).to_parquet(buffer)
    return buffer.getvalue()


PARQUET = _parquet_bytes()


class _Source:
    """Поддельный источник: отдаёт ответы из списка по очереди и считает запросы.

    Последний элемент повторяется. Потоковое тело читается один раз, поэтому для него
    передаётся фабрика ``() -> Response``: новый ответ на каждую попытку.
    """

    def __init__(
        self, *responses: httpx.Response | Exception | Callable[[], httpx.Response]
    ) -> None:
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        item = self._responses[min(self.calls, len(self._responses)) - 1]
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, httpx.Response) else item()

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


class _BrokenStream(httpx.SyncByteStream):
    """Тело ответа, которое обрывается после первого куска."""

    def __iter__(self) -> Iterator[bytes]:
        yield PARQUET[:10]
        raise httpx.ReadError("connection reset by peer")


@pytest.fixture
def data() -> DataSettings:
    return DataSettings(
        url_template="https://source.test/green_tripdata_{month}.parquet",
        download=DownloadSettings(max_attempts=3, backoff_base_s=1.0),
    )


@pytest.fixture
def sleeps() -> list[float]:
    return []


def _run(
    source: _Source,
    tmp_path: Path,
    data: DataSettings,
    sleeps: list[float],
    *,
    force: bool = False,
) -> str:
    return ingest_month(
        MONTH,
        data_root=str(tmp_path / "data"),
        data=data,
        force=force,
        client=source.client(),
        sleep=sleeps.append,
    )


def _target(tmp_path: Path) -> Path:
    return tmp_path / "data" / "raw" / f"green_tripdata_{MONTH}.parquet"


def _leftovers(tmp_path: Path) -> list[Path]:
    raw = tmp_path / "data" / "raw"
    return list(raw.glob("*.part")) if raw.exists() else []


# --- Успешная загрузка ------------------------------------------------------------------


def test_download_saves_readable_parquet(
    tmp_path: Path, data: DataSettings, sleeps: list[float]
) -> None:
    source = _Source(httpx.Response(200, content=PARQUET))
    target = _run(source, tmp_path, data, sleeps)

    assert Path(target) == _target(tmp_path)
    assert len(pd.read_parquet(target)) == 3
    assert source.calls == 1
    assert _leftovers(tmp_path) == []


def test_downloaded_file_is_world_readable(
    tmp_path: Path, data: DataSettings, sleeps: list[float]
) -> None:
    # mkstemp даёт 0600; файл данных должны читать и другие пользователи (контейнеры).
    _run(_Source(httpx.Response(200, content=PARQUET)), tmp_path, data, sleeps)
    assert _target(tmp_path).stat().st_mode & 0o777 == 0o644


def test_requests_url_from_template(
    tmp_path: Path, data: DataSettings, sleeps: list[float]
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=PARQUET)

    ingest_month(
        MONTH,
        data_root=str(tmp_path / "data"),
        data=data,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
    )
    assert seen == [URL]


# --- Идемпотентность ---------------------------------------------------------------------


def test_existing_file_is_not_downloaded_again(
    tmp_path: Path, data: DataSettings, sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    target = _target(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    source = _Source(httpx.Response(200, content=PARQUET))

    with caplog.at_level(logging.INFO, logger="taxi_duration"):
        _run(source, tmp_path, data, sleeps)

    assert source.calls == 0
    assert target.read_bytes() == b"existing"
    assert "already exists" in caplog.text


def test_force_downloads_again(tmp_path: Path, data: DataSettings, sleeps: list[float]) -> None:
    target = _target(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"stale")
    source = _Source(httpx.Response(200, content=PARQUET))

    _run(source, tmp_path, data, sleeps, force=True)

    assert source.calls == 1
    assert target.read_bytes() == PARQUET


# --- Повторы -----------------------------------------------------------------------------


def test_transient_5xx_is_retried(tmp_path: Path, data: DataSettings, sleeps: list[float]) -> None:
    source = _Source(httpx.Response(503), httpx.Response(200, content=PARQUET))
    _run(source, tmp_path, data, sleeps)

    assert source.calls == 2
    assert sleeps == [1.0]
    assert _target(tmp_path).exists()


def test_network_error_is_retried(tmp_path: Path, data: DataSettings, sleeps: list[float]) -> None:
    source = _Source(httpx.ConnectError("refused"), httpx.Response(200, content=PARQUET))
    _run(source, tmp_path, data, sleeps)

    assert source.calls == 2
    assert _target(tmp_path).exists()


def test_5xx_on_every_attempt_leaves_no_file(
    tmp_path: Path, data: DataSettings, sleeps: list[float]
) -> None:
    source = _Source(httpx.Response(500))

    with pytest.raises(DataError, match="after 3 attempts"):
        _run(source, tmp_path, data, sleeps)

    assert source.calls == 3
    assert sleeps == [1.0, 2.0]  # экспоненциальная пауза, после последней попытки не ждём
    assert not _target(tmp_path).exists()
    assert _leftovers(tmp_path) == []


def test_broken_stream_leaves_no_file(
    tmp_path: Path, data: DataSettings, sleeps: list[float]
) -> None:
    source = _Source(lambda: httpx.Response(200, stream=_BrokenStream()))

    with pytest.raises(DataError):
        _run(source, tmp_path, data, sleeps)

    assert source.calls == 3
    assert not _target(tmp_path).exists()
    assert _leftovers(tmp_path) == []


# --- Месяц не опубликован и мусор вместо parquet ------------------------------------------


def test_404_fails_immediately_without_retries(
    tmp_path: Path, data: DataSettings, sleeps: list[float]
) -> None:
    source = _Source(httpx.Response(404))

    with pytest.raises(DataError, match=r"2024-01 is not published"):
        _run(source, tmp_path, data, sleeps)

    assert source.calls == 1
    assert sleeps == []
    assert not _target(tmp_path).exists()


def test_non_parquet_body_is_rejected(
    tmp_path: Path, data: DataSettings, sleeps: list[float]
) -> None:
    source = _Source(httpx.Response(200, content=b"<html>Access Denied</html>"))

    with pytest.raises(DataError, match="not a valid parquet"):
        _run(source, tmp_path, data, sleeps)

    assert not _target(tmp_path).exists()
    assert _leftovers(tmp_path) == []


def test_invalid_month_is_data_error(tmp_path: Path, data: DataSettings) -> None:
    with pytest.raises(DataError, match="YYYY-MM"):
        ingest_month("2024-13", data_root=str(tmp_path), data=data)


# --- Удалённое хранилище (как s3://lake в Airflow) -----------------------------------------


def test_remote_root_is_written_via_fsspec(data: DataSettings, sleeps: list[float]) -> None:
    fs = fsspec.filesystem("memory")
    source = _Source(httpx.Response(200, content=PARQUET))

    target = ingest_month(
        MONTH, data_root="memory://lake", data=data, client=source.client(), sleep=sleeps.append
    )

    assert target == f"memory://lake/raw/green_tripdata_{MONTH}.parquet"
    assert fs.cat_file(f"/lake/raw/green_tripdata_{MONTH}.parquet") == PARQUET
    fs.rm("/lake", recursive=True)


# --- CLI: коды возврата ---------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, repo_root: Path, tmp_path: Path) -> Path:
    monkeypatch.setenv("TAXI_CONFIG", str(repo_root / "configs" / "base.yaml"))
    monkeypatch.setenv("TAXI__LOGGING__CONFIG_PATH", str(repo_root / "configs" / "logging.yaml"))
    monkeypatch.setenv("TAXI__PATHS__DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("TAXI__DATA__DOWNLOAD__BACKOFF_BASE_S", "0")
    return tmp_path


def _patch_source(monkeypatch: pytest.MonkeyPatch, source: _Source) -> None:
    monkeypatch.setattr(ingest, "make_client", lambda _download: source.client())


def test_cli_ingest_success(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_source(monkeypatch, _Source(httpx.Response(200, content=PARQUET)))
    result = runner.invoke(app, ["ingest", "--month", MONTH])
    assert result.exit_code == 0, result.output
    assert _target(cli_env).exists()


def test_cli_ingest_rerun_exits_0(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _target(cli_env)
    target.parent.mkdir(parents=True)
    target.write_bytes(PARQUET)
    source = _Source(httpx.Response(500))
    _patch_source(monkeypatch, source)

    result = runner.invoke(app, ["ingest", "--month", MONTH])

    assert result.exit_code == 0, result.output
    assert source.calls == 0
    assert "already exists" in result.output


def test_cli_ingest_5xx_exits_1(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_source(monkeypatch, _Source(httpx.Response(502)))
    result = runner.invoke(app, ["ingest", "--month", MONTH])
    assert result.exit_code == 1
    assert result.stderr.startswith("Error:")
    assert not _target(cli_env).exists()


def test_cli_ingest_404_exits_1_with_message(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _Source(httpx.Response(404))
    _patch_source(monkeypatch, source)
    result = runner.invoke(app, ["ingest", "--month", MONTH])
    assert result.exit_code == 1
    assert source.calls == 1
    assert "not published" in result.stderr


def test_cli_ingest_rejects_bad_month(cli_env: Path) -> None:
    result = runner.invoke(app, ["ingest", "--month", "2024-1"])
    assert result.exit_code == 2  # ошибка использования, до любой сети
