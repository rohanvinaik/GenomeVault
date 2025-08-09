.PHONY: help build run up down test fmt lint deps deps-upgrade deps-sync

help:
	@echo "Targets:"
	@echo "  build        Build Docker image"
	@echo "  run          Run API in foreground via docker compose"
	@echo "  up           Start API (detached)"
	@echo "  down         Stop stack"
	@echo "  test         Run pytest locally (venv)"
	@echo "  fmt          Format code with black & isort (if installed)"
	@echo "  lint         Run ruff linter and format checker"
	@echo "  deps         Regenerate locked dependency files"
	@echo "  deps-upgrade Upgrade all dependencies to latest versions"
	@echo "  deps-sync    Sync environment to match requirements exactly"

build:
	docker build -t genomevault/api:local .

run:
	docker compose up --build api

up:
	./scripts/dev_up.sh

down:
	./scripts/dev_down.sh

test:
	./scripts/test.sh

fmt:
	@if command -v black >/dev/null; then black .; else echo "black not installed"; fi
	@if command -v isort >/dev/null; then isort .; else echo "isort not installed"; fi

lint:
	@if command -v ruff >/dev/null; then ruff check . && ruff format --check .; else echo "ruff not installed"; fi

deps:
	@echo "Regenerating locked dependency files..."
	@if command -v pip-compile >/dev/null; then \
		pip-compile requirements.in -o requirements.txt --resolver=backtracking && \
		pip-compile requirements-dev.in -o requirements-dev.txt --resolver=backtracking; \
	else \
		echo "pip-tools not installed. Run: pip install pip-tools"; \
	fi

deps-upgrade:
	@echo "Upgrading all dependencies to latest versions..."
	@if command -v pip-compile >/dev/null; then \
		pip-compile --upgrade requirements.in -o requirements.txt --resolver=backtracking && \
		pip-compile --upgrade requirements-dev.in -o requirements-dev.txt --resolver=backtracking; \
	else \
		echo "pip-tools not installed. Run: pip install pip-tools"; \
	fi

deps-sync:
	@echo "Syncing environment to match requirements..."
	@if command -v pip-sync >/dev/null; then \
		pip-sync requirements-dev.txt; \
	else \
		echo "pip-tools not installed. Run: pip install pip-tools"; \
	fi
