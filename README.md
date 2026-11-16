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

## Данные

Источник: открытые месячные parquet NYC TLC Green Taxi (около 1–2 MB и 50–70 тысяч поездок
в месяц). Шаги запускаются по порядку, каждый читает результат предыдущего:

```bash
uv run taxi ingest --month 2024-01    # скачать сырой месяц (повторный запуск пропускается)
uv run taxi prepare --month 2024-01   # таргет duration_min, фильтры 1..60 мин и по месяцу
uv run taxi build-features            # признаки зон по всем месяцам из data.months
```

| Шаг | Результат | Что важно |
|---|---|---|
| `ingest` | `data/raw/green_tripdata_YYYY-MM.parquet` | атомарная запись, 3 попытки, `--force` качает заново |
| `prepare` | `data/processed/trips_YYYY-MM.parquet` | схема фиксирована (`PREPARED_SCHEMA`) |
| `build-features` | `data/features/zone_stats.parquet` | окно за прошлые 7 дней, без текущего дня |

Месяцы, пороги фильтров и окно признаков задаются в секциях `data` и `features`
файла `configs/base.yaml`. Корень данных `paths.data_root` может быть и локальной папкой,
и `s3://...`.

Каталог `data/` в git не коммитится: данные версионирует DVC.

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

## Локальный стенд (Docker)

Сервисы профиля `core`: Postgres, RustFS (S3), MLflow, Redis. Нужно около 0.7 GB RAM.

```bash
cp .env.example .env   # один раз; замени все change-me ДО первого make up
make up                # поднимает стенд и ждёт, пока все сервисы станут healthy
```

| Цель | Что делает |
|---|---|
| `make build` | собирает `taxi-duration-train:<git sha>` (и `:dev`) и образ MLflow |
| `make up` | `docker compose --profile core up -d --wait`; падает, если за 120 с стенд не стал healthy |
| `make down` | останавливает стенд, **данные в volumes сохраняются** |
| `make ps` | статус сервисов, включая завершённый `rustfs-init` |
| `make logs` | логи всех сервисов; один сервис — `make logs SERVICE=mlflow` |

| Сервис | С хоста | Из контейнеров |
|---|---|---|
| MLflow UI / API | http://localhost:5001 | http://mlflow:5000 |
| RustFS S3 API | http://localhost:9000 | http://rustfs:9000 |
| Консоль RustFS | http://localhost:9001/rustfs/console/ | — |
| Postgres | localhost:5432 | postgres:5432 |
| Redis | localhost:6379 | redis:6379 |

Полный сброс (удалит все эксперименты MLflow и объекты S3): `docker compose --profile core down -v`.
Пароли из `.env` применяются только при первом старте на пустых volumes.

## Структура

```
configs/              YAML-конфиги приложения и логирования
docker/               Dockerfile образов (train, mlflow)
infra/postgres/       init.sql: базы и пользователи mlflow, airflow
docker-compose.yml    локальный стенд (профиль core)
src/taxi_duration/    пакет (src-layout)
  cli.py              точка входа `taxi`
  config.py           типизированные настройки (pydantic-settings)
  logging_setup.py    настройка логирования из шаблона
  errors.py           иерархия ожидаемых ошибок
  storage.py          пути данных как fsspec URI, атомарная запись parquet
  data/ingest.py      загрузка месяца с повторами
  data/prepare.py     очистка поездок и таргет
  features/zone_stats.py  признаки зон за прошлые дни
data/                 raw/, processed/, features/ (не в git)
tests/unit/           юнит-тесты
```

## Процесс разработки

GitHub Flow: ветка от `main` → коммиты в формате
[Conventional Commits](https://www.conventionalcommits.org/) → PR → ревью и CI → Squash and merge.
Каждый завершённый этап отмечается тегом `v0.N.0`.
