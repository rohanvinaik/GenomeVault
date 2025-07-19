#!/bin/bash
# Ultimate fix script for GenomeVault

echo "🚀 GenomeVault Ultimate Fix Script"
echo "=================================="

cd /Users/rohanvinaik/genomevault

# Step 1: Install all dependencies
echo "📦 Installing all dependencies..."
pip install -q scikit-learn biopython pysam pynacl pyyaml uvicorn web3 eth-account seaborn
pip install -q pydantic-settings pydantic>=2.0.0

# Step 2: Fix imports
echo "🔧 Fixing imports..."
python3 fix_imports.py

# Step 3: Run simple test
echo "🧪 Running simple import test..."
python3 simple_test.py

# Step 4: Run pytest if simple test passes
if [ $? -eq 0 ]; then
    echo ""
    echo "🧪 Running pytest..."
    python -m pytest tests/test_simple.py -v
else
    echo "❌ Simple test failed, skipping pytest"
fi

echo ""
echo "✅ Done!"
echo ""
echo "To manually test imports:"
echo "  python3 simple_test.py"
echo ""
echo "To run all tests:"
echo "  python -m pytest tests/ -v"
