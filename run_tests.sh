#!/bin/bash
# Test runner for GenomeVault

echo "🧪 Running GenomeVault Tests"
echo "==========================="

cd /Users/rohanvinaik/genomevault

# First run the minimal test
echo "Running minimal test..."
python3 minimal_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Minimal test passed!"
    echo ""
    echo "Running pytest..."
    python -m pytest tests/test_simple.py -v
else
    echo ""
    echo "❌ Minimal test failed"
fi
