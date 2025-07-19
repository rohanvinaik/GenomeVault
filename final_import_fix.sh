#!/bin/bash
# Final fix for all import issues

echo "🚀 Final Import Fix for GenomeVault"
echo "==================================="

cd /Users/rohanvinaik/genomevault

# Fix 1: Create a proper package structure
echo "📦 Creating proper package structure..."
touch genomevault/__init__.py 2>/dev/null || true
echo '__version__ = "0.1.0"' > genomevault/__init__.py

# Fix 2: The core/__init__.py is already fixed

# Fix 3: Add missing hashlib import to binding.py
echo "🔧 Adding hashlib import to binding.py..."
if ! grep -q "import hashlib" hypervector_transform/binding.py; then
    sed -i '' '13a\
import hashlib
' hypervector_transform/binding.py
fi

# Fix 4: Run the diagnostic again
echo -e "\n🔍 Running diagnostic..."
python3 diagnose_imports.py

# Fix 5: Try to run pytest
echo -e "\n🧪 Running pytest..."
python3 -m pytest tests/test_simple.py -v --tb=short

echo -e "\n✅ Fix complete!"
