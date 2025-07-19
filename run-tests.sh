#!/bin/bash
# run-tests.sh - Test runner with various options

set -euo pipefail

# Default to running all tests
ACTION="${1:-all}"

case "$ACTION" in
  unit)
    echo "Running unit tests..."
    pytest tests/unit/ -v
    ;;
  integration)
    echo "Running integration tests..."
    pytest tests/integration/ -v
    ;;
  security)
    echo "Running security tests..."
    pytest tests/security/ -v
    echo "Running Bandit security scan..."
    bandit -r genomevault/
    ;;
  performance)
    echo "Running performance tests..."
    pytest tests/performance/ -v --benchmark-only
    ;;
  coverage)
    echo "Running tests with coverage..."
    pytest tests/ --cov=genomevault --cov-report=html --cov-report=term
    echo "Coverage report available in htmlcov/index.html"
    ;;
  ci)
    echo "Running CI checks..."
    # Run all checks as in CI
    echo "Checking code formatting..."
    black --check .
    echo "Checking import sorting..."
    isort --check-only .
    echo "Running flake8..."
    flake8 .
    echo "Running mypy..."
    mypy genomevault --ignore-missing-imports
    echo "Running tests with coverage..."
    pytest tests/ --cov=genomevault
    ;;
  watch)
    echo "Watching tests..."
    ptw tests/ -- -v
    ;;
  all)
    echo "Running all tests..."
    pytest tests/ -v
    ;;
  *)
    echo "Usage: $0 {unit|integration|security|performance|coverage|ci|watch|all}"
    echo ""
    echo "Options:"
    echo "  unit        - Run unit tests only"
    echo "  integration - Run integration tests only"
    echo "  security    - Run security tests and scans"
    echo "  performance - Run performance benchmarks"
    echo "  coverage    - Run all tests with coverage report"
    echo "  ci          - Run all CI checks (lint, format, tests)"
    echo "  watch       - Watch for changes and rerun tests"
    echo "  all         - Run all tests (default)"
    exit 1
    ;;
esac
