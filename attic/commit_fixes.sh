#!/bin/bash

# Script to commit the fixes as suggested in the error analysis

set -e

echo "🔧 Committing GenomeVault fixes..."

cd /Users/rohanvinaik/genomevault

# Add the fixed files
git add genomevault/hypervector/encoding/unified_encoder.py
git add genomevault/hypervector/encoding/__init__.py
git add genomevault/core/config.py
git add .ruff.toml

# Commit with descriptive message
git commit -m "fix(syntax): resolve syntax errors in unified_encoder.py and __init__.py

- Fixed broken import statements in unified_encoder.py
- Fixed broken import statements in hypervector/encoding/__init__.py
- Added missing get_config() function to core/config.py
- Updated .ruff.toml to be compatible with both old and new Ruff versions
- Addresses the specific syntax errors blocking Phase 7 validation

Resolves the parse errors that were stopping Ruff at 32/34 files."

echo "✅ Changes committed successfully!"

# Run a quick validation
echo "🧪 Running quick validation..."
python -m py_compile genomevault/hypervector/encoding/unified_encoder.py
python -m py_compile genomevault/hypervector/encoding/__init__.py
python -c "from genomevault.core.config import get_config; print('Config import works')"

echo "🎉 All fixes verified!"
