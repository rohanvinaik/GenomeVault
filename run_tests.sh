#!/bin/bash
# Test script for GenomeVault

echo "Running GenomeVault tests..."

# Change to the genomevault directory
cd /Users/rohanvinaik/genomevault

# Check for stubbed files
echo "1. Checking for stubbed files..."
python scripts/check_no_stubs.py

# Run the e2e tests
echo -e "\n2. Running e2e tests..."
python -m pytest tests/e2e/test_minimal_pipeline.py -v

# Run the adversarial tests
echo -e "\n3. Running adversarial tests..."
python -m pytest tests/adversarial/test_pir_timing.py -v

# Run linters
echo -e "\n4. Running linters..."
echo "   - Black..."
black . --check || echo "Black would make changes"

echo "   - isort..."
isort . --check || echo "isort would make changes"

echo "   - Flake8..."
flake8 . --max-line-length=120 || echo "Flake8 found issues"

echo -e "\nTest run complete!"
