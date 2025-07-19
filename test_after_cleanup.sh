#!/bin/bash
# Final test after tail-chasing cleanup

echo "🎯 Testing GenomeVault after tail-chasing cleanup"
echo "================================================"

cd /Users/rohanvinaik/genomevault

# First, let's make sure all dependencies are installed
echo "📦 Ensuring all dependencies are installed..."
pip install -q structlog pydantic-settings

# Run pytest
echo ""
echo "🧪 Running pytest..."
python -m pytest tests/test_simple.py -v

echo ""
echo "✅ Test complete!"
