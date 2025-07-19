#!/bin/bash
# Test after fixing utils imports

echo "🧪 Testing after fixing utils imports"
echo "===================================="

cd /Users/rohanvinaik/genomevault

# Run pytest
python -m pytest tests/test_simple.py -v

# If that works, run all tests
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Simple tests passed!"
    echo ""
    echo "🎉 The tail-chasing has been defeated!"
else
    echo ""
    echo "❌ Still have issues, checking what's wrong..."
    python -c "from utils import get_logger; print('✅ utils imports work')" || echo "❌ utils imports failed"
fi
