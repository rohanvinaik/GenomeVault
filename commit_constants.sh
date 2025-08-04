#!/bin/bash

# Script to commit the constant addition and verification improvements

set -e

echo "🔧 Committing HYPERVECTOR_DIMENSIONS constant and verification improvements..."

cd /Users/rohanvinaik/genomevault

# Add the new and modified files
git add genomevault/core/constants.py
git add final_verification.py
git add test_constant.py

# Commit with descriptive message
git commit -m "feat(constants): add HYPERVECTOR_DIMENSIONS constant and verification suite

- Added HYPERVECTOR_DIMENSIONS = 10_000 to core/constants.py  
- Import and re-export constants from utils/constants.py for centralization
- Created comprehensive final_verification.py script
- Added test_constant.py for quick constant verification
- Addresses the verification requirements for constants import

This enables:
- from genomevault.core.constants import HYPERVECTOR_DIMENSIONS
- UnifiedHypervectorEncoder imports with proper constants
- Full ruff check --statistics execution  
- pytest -q -k 'not api and not nanopore' test collection"

echo "✅ Constant and verification changes committed!"

# Run the quick constant test
echo "🧪 Running quick constant verification..."
python test_constant.py

echo "🎉 Constant verification passed!"
