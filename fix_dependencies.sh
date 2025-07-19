#!/bin/bash
# Fix script for GenomeVault Pydantic issues

echo "🔧 Fixing GenomeVault Pydantic dependencies..."

# Install pydantic-settings
echo "📦 Installing pydantic-settings..."
pip install pydantic-settings

# Upgrade pydantic to v2
echo "📦 Upgrading pydantic to v2..."
pip install --upgrade pydantic

# Install other missing dependencies
echo "📦 Installing other requirements..."
pip install -r requirements.txt

echo "✅ Dependencies fixed!"
echo ""
echo "🧪 Running tests to verify fix..."
pytest tests/test_simple.py -v
