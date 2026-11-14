# taxi-duration

Учебный end-to-end MLOps-проект: модель предсказывает длительность поездки NYC Green Taxi.
Модель здесь только повод: проект проходит полный жизненный цикл ML-системы, от пакета и
конфигов до CI, serving и мониторинга.

## Требования

- Python 3.12 (версия закреплена в `.python-version`, uv установит её сам)
- [uv](https://docs.astral.sh/uv/) 0.13+
- GNU Make (на macOS и Linux есть из коробки)

## Установка

```bash
git clone git@github.com:AlexCh-info/ml-prod.git
cd ml-prod
make install
uv run pre-commit install
```

`make install` выполняет `uv sync --locked`: создаёт `.venv`, ставит пакет `taxi-duration` в
editable-режиме, runtime- и dev-зависимости строго по `uv.lock`. Если `uv.lock` устарел
относительно `pyproject.toml`, команда падает.

`uv run pre-commit install` ставит git-хук: ruff, mypy и проверки файлов запускаются перед
каждым коммитом.

## Запуск

```bash
uv run taxi --help          # список команд
uv run taxi version         # версия пакета
uv run taxi config show     # итоговый конфиг (секреты замаскированы)
```

## Конфигурация

Настройки собираются из нескольких источников. Каждый следующий в списке перекрывает
предыдущие:

1. значения по умолчанию в коде (`src/taxi_duration/config.py`);
2. YAML-файл: `configs/base.yaml` или путь из переменной `TAXI_CONFIG`;
3. файл `.env` в корне проекта;
4. переменные окружения `TAXI__<СЕКЦИЯ>__<ПОЛЕ>`.

```bash
cp .env.example .env                                  # локальные переопределения, не в git
TAXI__LOGGING__FORMAT=json uv run taxi config show    # разовое переопределение
TAXI_CONFIG=configs/base.yaml uv run taxi config show # явный выбор YAML-файла
```

Секреты (например, `TAXI__S3__SECRET_ACCESS_KEY`) задаются только через окружение или `.env`
и никогда не пишутся в `configs/*.yaml`.

Логирование настраивается шаблоном `configs/logging.yaml`. Уровень и формат (`text` или `json`)
берутся из секции `logging` конфига.

## Make-цели

| Цель | Что делает |
|---|---|
| `make install` | ставит окружение по `uv.lock` |
| `make fmt` | исправляет код: `ruff check --fix` и `ruff format` |
| `make lint` | только проверяет: `ruff check`, `ruff format --check`, `mypy` (strict) |
| `make test` | запускает `pytest` |
| `make check` | `lint` + `test`; это же запускается в CI |
| `make clean` | удаляет кеши инструментов и `__pycache__` |

Обычный цикл: написал код → `make fmt` → `make check` → коммит.

## Структура

```
configs/              YAML-конфиги приложения и логирования
src/taxi_duration/    пакет (src-layout)
  cli.py              точка входа `taxi`
  config.py           типизированные настройки (pydantic-settings)
  logging_setup.py    настройка логирования из шаблона
  errors.py           иерархия ожидаемых ошибок
tests/unit/           юнит-тесты
```

## Процесс разработки

GitHub Flow: ветка от `main` → коммиты в формате
[Conventional Commits](https://www.conventionalcommits.org/) → PR → ревью и CI → Squash and merge.
Каждый завершённый этап отмечается тегом `v0.N.0`.
