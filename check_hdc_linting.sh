#!/bin/bash

# Comprehensive linting check for HDC error handling files

echo "🔍 Running comprehensive linting checks..."

# Files to check
FILES=(
    "genomevault/hypervector/error_handling.py"
    "genomevault/hypervector/__init__.py"
    "genomevault/api/app.py"
    "genomevault/core/constants.py"
    "tests/test_hdc_error_handling.py"
    "examples/hdc_error_tuning_example.py"
)

echo "📋 Files to check:"
for file in "${FILES[@]}"; do
    echo "  - $file"
done

# Run Black
echo -e "\n⚫ Running Black..."
black --check "${FILES[@]}"
if [ $? -ne 0 ]; then
    echo "❌ Black found formatting issues. Running formatter..."
    black "${FILES[@]}"
    echo "✅ Black formatting applied"
else
    echo "✅ Black check passed"
fi

# Run isort
echo -e "\n📦 Running isort..."
isort --check-only "${FILES[@]}"
if [ $? -ne 0 ]; then
    echo "❌ isort found import ordering issues. Fixing..."
    isort "${FILES[@]}"
    echo "✅ isort formatting applied"
else
    echo "✅ isort check passed"
fi

# Run flake8
echo -e "\n🔍 Running flake8..."
flake8 "${FILES[@]}" --max-line-length=120 --extend-ignore=E203,W503,E501
if [ $? -ne 0 ]; then
    echo "❌ flake8 found issues"
else
    echo "✅ flake8 check passed"
fi

# Run pylint
echo -e "\n🌟 Running pylint..."
for file in "${FILES[@]}"; do
    echo -e "\nChecking $file..."
    pylint "$file" --max-line-length=120 --disable=C0103,R0913,R0914,W0212,R0801,R0903,C0116,C0115,W0613,R0912,R0915,W0622 --errors-only
done

# Run mypy
echo -e "\n🔎 Running mypy..."
mypy "${FILES[@]}" --ignore-missing-imports --no-error-summary

echo -e "\n✅ Linting check complete!"

# Show git diff if any changes were made
if ! git diff --quiet; then
    echo -e "\n📝 Changes made by linters:"
    git diff --stat
fi
