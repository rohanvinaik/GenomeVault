#!/bin/bash
# Fix all linting issues for PIR implementation

echo "🔧 Fixing all linting issues for PIR Implementation"
echo "=================================================="

cd /Users/rohanvinaik/genomevault

# Fix imports with isort
echo -e "\n📐 Running isort..."
isort genomevault/pir/ tests/pir/ scripts/bench_pir.py --profile black --line-length 100

# Format with black
echo -e "\n⚫ Running black..."
black genomevault/pir/ tests/pir/ scripts/bench_pir.py --line-length 100

# Verify all linters pass
echo -e "\n✅ Verifying all linters..."

echo -e "\n1. isort check:"
isort --check-only genomevault/pir/ tests/pir/ scripts/bench_pir.py --profile black --line-length 100

echo -e "\n2. black check:"
black --check genomevault/pir/ tests/pir/ scripts/bench_pir.py --line-length 100

echo -e "\n3. flake8 check:"
flake8 genomevault/pir/ tests/pir/ scripts/bench_pir.py --config=.flake8

echo -e "\n✅ All linting fixes applied!"
echo -e "\n📊 Files modified:"
git status --short genomevault/pir/ tests/pir/ scripts/bench_pir.py
