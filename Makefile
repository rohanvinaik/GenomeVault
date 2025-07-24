# GenomeVault Makefile
# Cross-cutting infrastructure for ZK, PIR, and HDC lanes

.PHONY: all test bench clean install lint docs coverage security

# Python interpreter
PYTHON := python3
PIP := pip3

# Directories
SRC_DIR := genomevault
TEST_DIR := tests
BENCH_DIR := benchmarks
DOC_DIR := docs
SCRIPTS_DIR := scripts

# Default target
all: install lint test

# Installation
install:
	@echo "Installing GenomeVault dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"

# Linting and code quality
lint:
	@echo "Running code quality checks..."
	black --check $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	flake8 $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	mypy $(SRC_DIR) --ignore-missing-imports

format:
	@echo "Formatting code..."
	black $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	isort $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)

# Testing
test:
	@echo "Running all tests..."
	pytest $(TEST_DIR) -v --tb=short

test-unit:
	@echo "Running unit tests..."
	pytest $(TEST_DIR)/unit -v

test-integration:
	@echo "Running integration tests..."
	pytest $(TEST_DIR)/integration -v

test-e2e:
	@echo "Running end-to-end tests..."
	pytest $(TEST_DIR)/e2e -v

test-property:
	@echo "Running property-based tests..."
	pytest $(TEST_DIR)/property -v --hypothesis-show-statistics

test-adversarial:
	@echo "Running adversarial tests..."
	pytest $(TEST_DIR)/adversarial -v

# Lane-specific tests
test-zk:
	@echo "Running ZK proof tests..."
	pytest $(TEST_DIR)/zk -v
	pytest $(TEST_DIR)/test_hdc_implementation.py::TestAlgebraicProperties -v

test-pir:
	@echo "Running PIR tests..."
	pytest $(TEST_DIR)/pir -v

test-hdc:
	@echo "Running HDC tests..."
	pytest $(TEST_DIR)/test_hdc_implementation.py -v
	pytest $(TEST_DIR)/test_hdc_quality.py -v

# Coverage
coverage:
	@echo "Running test coverage..."
	pytest --cov=$(SRC_DIR) --cov-report=html --cov-report=term $(TEST_DIR)
	@echo "Coverage report generated in htmlcov/index.html"

coverage-report:
	@echo "Opening coverage report..."
	open htmlcov/index.html

# Benchmarking
bench: bench-hdc bench-pir bench-zk

bench-hdc:
	@echo "Running HDC benchmarks..."
	$(PYTHON) $(SCRIPTS_DIR)/bench.py --lane hdc --output $(BENCH_DIR)

bench-pir:
	@echo "Running PIR benchmarks..."
	$(PYTHON) $(SCRIPTS_DIR)/bench.py --lane pir --output $(BENCH_DIR)

bench-zk:
	@echo "Running ZK benchmarks..."
	$(PYTHON) $(SCRIPTS_DIR)/bench.py --lane zk --output $(BENCH_DIR)

bench-quick:
	@echo "Running quick HDC benchmark..."
	$(PYTHON) $(SCRIPTS_DIR)/bench_hdc.py --quick --output-dir $(BENCH_DIR)/hdc

# Security checks
security:
	@echo "Running security checks..."
	$(PYTHON) $(SCRIPTS_DIR)/security_check.py --project-dir .
	bandit -r $(SRC_DIR) -f json -o security_report.json

threat-scan:
	@echo "Scanning for security threats..."
	$(PYTHON) $(SCRIPTS_DIR)/security_check.py --project-dir .
	@echo "Checking for hardcoded secrets..."
	grep -r "password\|secret\|api_key" $(SRC_DIR) || echo "No hardcoded secrets found"

# Documentation
docs:
	@echo "Building documentation..."
	cd $(DOC_DIR) && make html
	@echo "Documentation built in docs/_build/html/index.html"

docs-serve:
	@echo "Serving documentation..."
	cd $(DOC_DIR)/_build/html && python -m http.server 8000

# Performance monitoring
perf-report:
	@echo "Generating performance report..."
	$(PYTHON) $(SCRIPTS_DIR)/generate_perf_report.py --input $(BENCH_DIR) --output $(DOC_DIR)/perf

# Docker operations
docker-build:
	@echo "Building Docker image..."
	docker build -t genomevault:latest .

docker-run:
	@echo "Running GenomeVault in Docker..."
	docker run -it --rm -p 8000:8000 genomevault:latest

# API operations
api-serve:
	@echo "Starting GenomeVault API server..."
	uvicorn genomevault.api.main:app --reload --host 0.0.0.0 --port 8000

api-test:
	@echo "Testing API endpoints..."
	pytest tests/api -v

# Database operations
db-init:
	@echo "Initializing database..."
	$(PYTHON) -m genomevault.scripts.init_db

db-migrate:
	@echo "Running database migrations..."
	alembic upgrade head

# Cleaning
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	rm -rf $(BENCH_DIR)/*/plots
	rm -f security_report.json

clean-all: clean
	@echo "Deep cleaning..."
	rm -rf build dist
	rm -rf .tox
	rm -rf venv

# Development helpers
dev-setup: install-dev
	@echo "Setting up development environment..."
	pre-commit install
	git config core.hooksPath .git/hooks

update-deps:
	@echo "Updating dependencies..."
	pip-compile requirements.in
	pip-compile requirements-dev.in

# CI/CD helpers
ci-test:
	@echo "Running CI test suite..."
	pytest --tb=short --junitxml=test-results.xml

ci-lint:
	@echo "Running CI linting..."
	black --check $(SRC_DIR)
	flake8 $(SRC_DIR) --format=json --output-file=flake8-report.json

ci-security:
	@echo "Running CI security scan..."
	bandit -r $(SRC_DIR) -f json -o bandit-report.json
	safety check --json > safety-report.json

# Version management
version:
	@echo "Current GenomeVault versions:"
	@$(PYTHON) -c "from genomevault.version import get_version_info; import json; print(json.dumps(get_version_info(), indent=2))"

bump-version:
	@echo "Bumping version..."
	bumpversion patch

# Help
help:
	@echo "GenomeVault Makefile targets:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make bench        - Run all benchmarks"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code"
	@echo "  make coverage     - Generate test coverage report"
	@echo "  make security     - Run security checks"
	@echo "  make docs         - Build documentation"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "Lane-specific targets:"
	@echo "  make test-hdc     - Test HDC implementation"
	@echo "  make test-pir     - Test PIR implementation"
	@echo "  make test-zk      - Test ZK implementation"
	@echo "  make bench-hdc    - Benchmark HDC"
	@echo "  make bench-pir    - Benchmark PIR"
	@echo "  make bench-zk     - Benchmark ZK"
