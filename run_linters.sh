#!/bin/bash
# Script to run linters before pushing to GitHub

echo "Running linters for GenomeVault..."

# Black
echo "Running Black..."
black .

# isort
echo "Running isort..."
isort .

# Flake8
echo "Running Flake8..."
flake8 . || true

# Pylint (optional, can be slow)
# echo "Running Pylint..."
# pylint genomevault/ || true

echo "Linting complete! Now you can push to GitHub."
