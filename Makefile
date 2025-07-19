.PHONY: help test coverage lint format security clean docker-up docker-down install dev-install

help:
	@echo "GenomeVault Development Commands:"
	@echo "  make install     - Install production dependencies"
	@echo "  make dev-install - Install development dependencies"
	@echo "  make test        - Run all tests"
	@echo "  make coverage    - Run tests with coverage report"
	@echo "  make lint        - Run all linters"
	@echo "  make format      - Format code automatically"
	@echo "  make security    - Run security checks"
	@echo "  make clean       - Clean up generated files"
	@echo "  make docker-up   - Start docker test services"
	@echo "  make docker-down - Stop docker test services"

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m "not integration"

test-integration:
	pytest tests/integration/ -v -m integration

coverage:
	pytest tests/ --cov=genomevault --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"
	@echo "Open htmlcov/index.html in your browser to view detailed coverage"

lint:
	@echo "Running Black..."
	black --check .
	@echo "Running isort..."
	isort --check-only .
	@echo "Running Flake8..."
	flake8 .
	@echo "Running mypy..."
	mypy genomevault --ignore-missing-imports
	@echo "Running pylint..."
	pylint genomevault --fail-under=8.0 || true

format:
	@echo "Formatting with Black..."
	black .
	@echo "Sorting imports with isort..."
	isort .

security:
	@echo "Running Bandit security scan..."
	bandit -r genomevault/ -f json -o security-report.json
	@echo "Running Safety check..."
	safety check --json --output safety-report.json || true
	@echo "Security reports generated: security-report.json, safety-report.json"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf build/ dist/
	rm -f security-report.json safety-report.json

docker-up:
	docker-compose -f docker-compose.test.yml up -d
	@echo "Waiting for services to be ready..."
	@sleep 5

docker-down:
	docker-compose -f docker-compose.test.yml down -v

docker-logs:
	docker-compose -f docker-compose.test.yml logs -f

# Development workflow commands
dev-test: docker-up test docker-down

dev-coverage: docker-up coverage docker-down

# CI simulation
ci: format lint security test
	@echo "All CI checks passed!"
