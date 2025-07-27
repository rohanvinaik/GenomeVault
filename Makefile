.PHONY: help install install-dev test test-unit test-integration test-e2e lint format clean build docs benchmark

# Default target
help:
	@echo "GenomeVault Development Commands"
	@echo "================================"
	@echo "Installation:"
	@echo "  make install       Install package"
	@echo "  make install-dev   Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-e2e      Run end-to-end tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run all linters"
	@echo "  make format        Auto-format code"
	@echo ""
	@echo "Building:"
	@echo "  make clean         Clean build artifacts"
	@echo "  make build         Build distribution packages"
	@echo "  make docs          Build documentation"
	@echo ""
	@echo "Development:"
	@echo "  make benchmark     Run benchmarks"
	@echo "  make dev           Setup development environment"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing targets
test:
	pytest tests/ -v --cov=genomevault --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

# Code quality targets
lint:
	@echo "Running Black..."
	@black --check genomevault/ tests/
	@echo "Running isort..."
	@isort --check genomevault/ tests/
	@echo "Running flake8..."
	@flake8 genomevault/ tests/
	@echo "Running mypy..."
	@mypy genomevault/

format:
	black genomevault/ tests/ examples/ scripts/
	isort genomevault/ tests/ examples/ scripts/

# Build targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

# Documentation
docs:
	cd docs && make clean && make html
	@echo "Documentation built in docs/_build/html/"

# Development targets
benchmark:
	python scripts/benchmarks/run_benchmark.py

dev: install-dev
	@echo "Development environment ready!"

# Shortcuts
fmt: format
check: lint test
all: format lint test build
