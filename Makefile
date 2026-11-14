SRC ?= src tests
RUNNER ?= uv run --frozen

.PHONY: install fmt lint test check clean

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
