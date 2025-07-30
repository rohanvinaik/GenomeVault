.PHONY: help build run up down test fmt

help:
	@echo "Targets:"
	@echo "  build    Build Docker image"
	@echo "  run      Run API in foreground via docker compose"
	@echo "  up       Start API (detached)"
	@echo "  down     Stop stack"
	@echo "  test     Run pytest locally (venv)"
	@echo "  fmt      Format code with black & isort (if installed)"

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
