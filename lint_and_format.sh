#!/bin/bash

# Script to lint and format the nanopore module code

echo "Running linters and formatters on nanopore module..."

# Change to genomevault directory
cd /Users/rohanvinaik/genomevault

# Format with black
echo "Running black formatter..."
black genomevault/nanopore/ --line-length 100

# Sort imports with isort
echo "Running isort..."
isort genomevault/nanopore/ --profile black --line-length 100

# Run flake8
echo "Running flake8..."
flake8 genomevault/nanopore/ --max-line-length 100 --extend-ignore E203,W503

# Run mypy type checking
echo "Running mypy..."
mypy genomevault/nanopore/ --ignore-missing-imports

# Run pylint
echo "Running pylint..."
pylint genomevault/nanopore/ --max-line-length 100 --disable=C0103,R0913,R0914,W0212

echo "Linting complete!"
