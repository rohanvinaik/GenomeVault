#!/bin/bash

# Script to run linters and fix issues for HDC error handling

echo "🔍 Running linters for HDC error handling implementation..."

# Files to check
FILES=(
    "genomevault/hypervector/error_handling.py"
    "genomevault/hypervector/__init__.py"
    "tests/test_hdc_error_handling.py"
    "examples/hdc_error_tuning_example.py"
)

echo "📋 Files to check:"
for file in "${FILES[@]}"; do
    echo "  - $file"
done

# Run black formatter
echo -e "\n⚫ Running Black formatter..."
black "${FILES[@]}"

# Run isort
echo -e "\n📦 Running isort..."
isort "${FILES[@]}"

# Run flake8
echo -e "\n🔍 Running flake8..."
flake8 "${FILES[@]}" --max-line-length=120 --extend-ignore=E203,W503

# Run mypy
echo -e "\n🔎 Running mypy..."
mypy "${FILES[@]}" --ignore-missing-imports

# Run pylint
echo -e "\n🌟 Running pylint..."
pylint "${FILES[@]}" --max-line-length=120 --disable=C0103,R0913,R0914,W0212

echo -e "\n✅ Linting complete!"
