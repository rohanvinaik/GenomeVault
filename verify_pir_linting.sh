#!/bin/bash
# Verify all linting passes for PIR implementation

echo "🔍 Verifying all linting passes for PIR Implementation"
echo "===================================================="

cd /Users/rohanvinaik/genomevault

# Track if all checks pass
ALL_PASS=true

# Check isort
echo -e "\n📐 Checking isort..."
if isort --check-only --profile black --line-length 100 genomevault/pir/ tests/pir/ scripts/bench_pir.py; then
    echo "✅ isort: PASS"
else
    echo "❌ isort: FAIL"
    ALL_PASS=false
fi

# Check black
echo -e "\n⚫ Checking black..."
if black --check --line-length 100 genomevault/pir/ tests/pir/ scripts/bench_pir.py; then
    echo "✅ black: PASS"
else
    echo "❌ black: FAIL"
    ALL_PASS=false
fi

# Check flake8
echo -e "\n🔎 Checking flake8..."
if flake8 genomevault/pir/ tests/pir/ scripts/bench_pir.py --config=.flake8; then
    echo "✅ flake8: PASS"
else
    echo "❌ flake8: FAIL"
    ALL_PASS=false
fi

# Summary
echo -e "\n================================"
if [ "$ALL_PASS" = true ]; then
    echo "✅ All linting checks PASSED!"
    echo -e "\nReady to push to GitHub!"
else
    echo "❌ Some linting checks FAILED!"
    echo -e "\nPlease fix the issues before pushing."
fi
