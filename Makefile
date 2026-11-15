SRC ?= src tests
RUNNER ?= uv run --frozen

# --- Docker -----------------------------------------------------------------
# Профиль core: postgres, rustfs, rustfs-init, mlflow, redis (см. docker-compose.yml).
PROFILE ?= core
COMPOSE ?= docker compose --profile $(PROFILE)
# Тег образа = git sha (конвенция: не latest). -dirty, если есть незакоммиченные правки:
# образ из грязного дерева не должен выдавать себя за коммит.
GIT_SHA := $(shell git rev-parse --short=7 HEAD 2>/dev/null || echo nogit)$(shell git diff --quiet HEAD 2>/dev/null || echo -dirty)
TRAIN_IMAGE ?= taxi-duration-train
# Сколько секунд ждать healthy в `make up` (сценарий «Холодный старт»: 2 минуты).
WAIT_TIMEOUT ?= 120
# make logs SERVICE=mlflow — логи одного сервиса; без SERVICE — всех.
SERVICE ?=

.PHONY: install fmt lint test check clean build up down ps logs

install:
	uv sync --locked

fmt:
	$(RUNNER) ruff check --fix $(SRC)
	$(RUNNER) ruff format $(SRC)

lint:
	$(RUNNER) ruff check $(SRC)
	$(RUNNER) ruff format --check $(SRC)
	$(RUNNER) mypy

test:
	$(RUNNER) pytest

check: lint test

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage htmlcov dist build
	find . \( -path './.venv' -o -path './.git' \) -prune \
		-o \( -type d -name "__pycache__" -o -type f -name "*.py[co]" \) \
		-exec rm -rf {} +

# .env как файловая цель: если файл есть, рецепт не выполняется (нет пререквизитов,
# значит «актуален»). Если нет — понятная подсказка и выход 1. Сами мы его НЕ создаём:
# пароли должен выбрать человек, а init.sql применит их только при первом старте.
.env:
	@echo "ERROR: файл .env не найден." >&2
	@echo "  cp .env.example .env   # затем замени все change-me на свои пароли" >&2
	@exit 1

# Все цели compose зависят от .env: compose подставляет ${VAR:?} даже для ps/down.
build: .env
	docker build -f docker/train.Dockerfile \
		-t $(TRAIN_IMAGE):$(GIT_SHA) -t $(TRAIN_IMAGE):dev .
	$(COMPOSE) build

# --wait: команда завершится только когда все сервисы healthy (или one-shot завершились
# с кодом 0), иначе — код ошибки. Удобно и для людей, и для CI.
up: .env
	$(COMPOSE) up -d --wait --wait-timeout $(WAIT_TIMEOUT)

# Без -v: volumes с данными MLflow/RustFS/Postgres сохраняются. Полный сброс — вручную:
#   docker compose --profile core down -v   (удалит ВСЕ эксперименты)
down: .env
	$(COMPOSE) down

ps: .env
	$(COMPOSE) ps -a

logs: .env
	$(COMPOSE) logs -f --tail=100 $(SERVICE)
