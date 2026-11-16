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

`uv run pre-commit install` ставит три git-хука:
- `pre-commit`: ruff, mypy и проверки файлов перед каждым коммитом;
- `post-checkout` и `pre-push`: синхронизация данных DVC (см. «Версионирование данных»).

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

## Версионирование данных (DVC)

Пайплайн `ingest → prepare → features` описан в `dvc.yaml`, а хеши всех входов и выходов
последнего прогона записаны в `dvc.lock`. В git лежат только эти два файла. Сами parquet
хранятся в локальном кеше `.dvc/cache` и в remote `rustfs` (бакет `s3://dvc` на локальном
стенде).

Один раз после клонирования поднимите стенд (`make up`) и передайте DVC ключи RustFS из `.env`.
Ключи пишутся в `.dvc/config.local`, а он в git не попадает:

```bash
set -a && source .env && set +a
uv run dvc remote modify --local rustfs access_key_id "$RUSTFS_ACCESS_KEY"
uv run dvc remote modify --local rustfs secret_access_key "$RUSTFS_SECRET_KEY"
```

| Команда | Что делает |
|---|---|
| `uv run dvc pull` | скачивает версию данных, записанную в текущем `dvc.lock`, без пересчёта |
| `uv run dvc repro` | перезапускает только стадии, у которых изменились код, данные или параметры |
| `uv run dvc push` | отправляет новые версии из кеша в RustFS |
| `uv run dvc status` | показывает, какие стадии устарели и почему |
| `uv run dvc dag` | рисует граф стадий |

Как изменить данные, например добавить месяц: правка `data.months` в `configs/base.yaml` →
`dvc repro` → `dvc push` → коммит `configs/base.yaml`, `dvc.lock` и `data/*/.gitignore`.
Пересчитываются только новые месяцы и `features`.

Данные следуют за git сами благодаря хукам из `pre-commit install`:
- после `git checkout`/`git switch` хук выполняет `dvc checkout`, и в `data/` оказывается
  версия из `dvc.lock` этого коммита (из локального кеша, без сети);
- перед `git push` хук выполняет `dvc push`, поэтому коммит не уедет без своих данных.
  Нужен поднятый стенд.

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
dvc.yaml, dvc.lock    пайплайн данных и хеши его последнего прогона
.dvc/config           remote DVC (ключи — в config.local, не в git)
data/                 raw/, processed/, features/ (не в git, версии — в DVC)
tests/unit/           юнит-тесты
```

## Процесс разработки

GitHub Flow: ветка от `main` → коммиты в формате
[Conventional Commits](https://www.conventionalcommits.org/) → PR → ревью и CI → Squash and merge.
Каждый завершённый этап отмечается тегом `v0.N.0`.
