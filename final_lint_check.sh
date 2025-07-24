#!/bin/bash

# Final linting check before push

echo "🔍 Running final linting checks for HDC implementation..."

# Run Black
echo -e "\n⚫ Running Black..."
black genomevault/hypervector/error_handling.py genomevault/api/app.py genomevault/core/constants.py

# Run isort
echo -e "\n📦 Running isort..."
isort genomevault/hypervector/error_handling.py genomevault/api/app.py genomevault/core/constants.py

# Run flake8
echo -e "\n🔍 Running flake8..."
flake8 genomevault/hypervector/error_handling.py genomevault/api/app.py genomevault/core/constants.py --max-line-length=120 --extend-ignore=E203,W503,E501

# Check Black formatting
echo -e "\n⚫ Checking Black formatting..."
black --check .

echo -e "\n✅ Linting complete!"
