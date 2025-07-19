#!/bin/bash
# Quick fix and test for GenomeVault

echo "🚀 GenomeVault Quick Fix & Test"
echo "================================"

# Make scripts executable
chmod +x fix_dependencies.sh
chmod +x test_imports.py
chmod +x debug_genomevault.py

# Run the debug script first
echo "🔍 Running diagnostics..."
python3 debug_genomevault.py

# If diagnostics show issues, run the fix
if [ $? -ne 0 ]; then
    echo ""
    echo "🔧 Running automatic fixes..."
    pip install pydantic-settings pydantic>=2.0.0
    
    echo ""
    echo "🧪 Testing again..."
    python3 test_imports.py
fi

echo ""
echo "✅ Done! Try running: pytest tests/test_simple.py -v"
