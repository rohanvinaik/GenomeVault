#!/bin/bash
# Fix isort and push PIR implementation

echo "🔧 Fixing isort issues and pushing PIR Implementation"
echo "===================================================="

# Change to project directory
cd /Users/rohanvinaik/genomevault

# Run isort to fix import ordering
echo -e "\n📐 Running isort to fix import ordering..."
isort genomevault/pir/ tests/pir/ scripts/bench_pir.py

# Run black to ensure formatting
echo -e "\n⚫ Running black formatter..."
black genomevault/pir/ tests/pir/ scripts/bench_pir.py

# Check isort compliance
echo -e "\n✅ Verifying isort compliance..."
isort --check-only genomevault/pir/ tests/pir/ scripts/bench_pir.py

# Run other linters to ensure everything passes
echo -e "\n🔍 Running flake8..."
flake8 genomevault/pir/ tests/pir/ scripts/bench_pir.py --config=.flake8

# Add the fixed files
echo -e "\n📁 Adding fixed files..."
git add genomevault/pir/ tests/pir/ scripts/bench_pir.py

# Show what changed
echo -e "\n📊 Import fixes applied:"
git diff --cached --name-only | grep -E "\.py$"

# Commit the isort fixes
echo -e "\n💾 Committing isort fixes..."
git commit -m "fix: Apply isort import ordering to PIR implementation

- Fix import ordering in it_pir_protocol.py
- Fix import ordering in integration_demo.py
- Fix import ordering in coordinator.py
- Fix import ordering in test_pir_protocol.py
- Fix import ordering in bench_pir.py"

# Push to GitHub
echo -e "\n🌐 Pushing to GitHub..."
git push origin main

echo -e "\n✅ Successfully fixed isort issues and pushed to GitHub!"
