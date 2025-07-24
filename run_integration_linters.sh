#!/bin/bash

echo "Running linters on HDC-PIR integration files..."

# Set up environment
export PYTHONPATH=/Users/rohanvinaik/genomevault:$PYTHONPATH

# Files to check
FILES=(
    "genomevault/pir/client/batched_query_builder.py"
    "genomevault/api/routers/tuned_query.py"
    "tests/test_hdc_pir_integration.py"
    "examples/hdc_pir_integration_demo.py"
)

# Run black
echo "Running black..."
black --line-length=100 "${FILES[@]}"

# Run isort
echo "Running isort..."
isort --profile=black --line-length=100 "${FILES[@]}"

# Run flake8
echo "Running flake8..."
flake8 --max-line-length=100 --extend-ignore=E203,W503 "${FILES[@]}"

# Run mypy
echo "Running mypy..."
mypy --ignore-missing-imports "${FILES[@]}"

echo "Linting complete!"
