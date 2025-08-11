#!/usr/bin/env bash
set -euo pipefail

# Track overall exit status
EXIT_STATUS=0

echo "========================================="
echo "Starting MVP Test Suite"
echo "========================================="
echo ""

# Run smoke tests first (quiet mode)
echo "Running smoke tests..."
if python -m pytest tests/smoke -q; then
    echo "✓ Smoke tests passed"
else
    echo "✗ Smoke tests failed"
    EXIT_STATUS=1
fi

echo ""
echo "Running detailed smoke tests (verbose)..."
if python -m pytest tests/smoke -v --tb=short; then
    echo "✓ Detailed smoke tests passed"
else
    echo "✗ Detailed smoke tests failed"
    EXIT_STATUS=1
fi

echo ""
echo "========================================="
echo "Running Linters and Type Checkers"
echo "========================================="
echo ""

# Define target modules for linting
TARGET_MODULES=(
    "genomevault/api"
    "genomevault/hypervector"
    "genomevault/hypervector_transform"
    "genomevault/zk"
    "genomevault/core"
)

# Run ruff linter with exclusions
echo "Running ruff linter..."
if python -m ruff check "${TARGET_MODULES[@]}" \
    --select=E,F \
    --extend-exclude="examples/" \
    --extend-exclude="devtools/" \
    --extend-exclude="*.ipynb" \
    --extend-exclude="__pycache__" \
    --extend-exclude=".git"; then
    echo "✓ Ruff linting passed"
else
    echo "✗ Ruff linting failed"
    EXIT_STATUS=1
fi

echo ""
echo "Running mypy type checker..."
if python -m mypy "${TARGET_MODULES[@]}" \
    --ignore-missing-imports \
    --exclude="examples/*" \
    --exclude="devtools/*" \
    --exclude="tests/*" \
    --no-error-summary; then
    echo "✓ MyPy type checking passed"
else
    echo "✗ MyPy type checking failed"
    EXIT_STATUS=1
fi

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ MVP implementation validated!"
    echo "All tests and checks passed successfully."
else
    echo "❌ MVP validation failed!"
    echo "Please fix the issues above before proceeding."
fi

# Exit with proper status code for CI
exit $EXIT_STATUS
