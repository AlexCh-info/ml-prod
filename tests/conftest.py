"""Общие фикстуры: изоляция тестов от окружения разработчика."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from taxi_duration.config import load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Каждый тест — с чистым окружением.

    - убираем TAXI_* из env, чтобы настройки разработчика не влияли на тесты;
    - переходим в tmp_path, чтобы не подхватить локальный .env;
    - сбрасываем кеш load_settings() до и после теста.
    """
    for name in list(os.environ):
        if name.startswith("TAXI_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


@pytest.fixture
def repo_root() -> Path:
    """Корень репозитория (там лежат configs/)."""
    return REPO_ROOT


@pytest.fixture
def write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Path]:
    """Записать YAML-конфиг во временный файл и указать на него TAXI_CONFIG."""

    def _write(content: str, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        monkeypatch.setenv("TAXI_CONFIG", str(path))
        return path

    return _write
