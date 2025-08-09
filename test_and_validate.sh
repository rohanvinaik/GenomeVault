#!/bin/bash
cd /Users/rohanvinaik/genomevault

echo "==================================="
echo "Testing and Git Push Validation"
echo "==================================="

# 1. Check git status
echo -e "\n1. Git Status:"
echo "-------------"
git status --short

# 2. Run validation tests
echo -e "\n2. Running validation tests:"
echo "----------------------------"

# Run ruff check
echo -e "\n► Running ruff check..."
if ruff check . --quiet; then
    echo "✅ Ruff check passed"
else
    echo "❌ Ruff check failed"
    ruff check . | head -20
fi

# Run ruff format check
echo -e "\n► Running ruff format check..."
if ruff format --check . --quiet; then
    echo "✅ Code formatting is correct"
else
    echo "❌ Code needs formatting"
    ruff format --diff . | head -20
fi

# Run basic import test
echo -e "\n► Testing Python imports..."
if python -m compileall . -q; then
    echo "✅ All Python files compile successfully"
else
    echo "❌ Some Python files have syntax errors"
fi

# Test specific imports
echo -e "\n► Testing key imports..."
python -c "import genomevault; print('✅ genomevault imports successfully')" 2>/dev/null || echo "❌ genomevault import failed"
python -c "from genomevault.api.main import app; print('✅ API imports successfully')" 2>/dev/null || echo "❌ API import failed"

# Check if pytest is available and run a quick test
echo -e "\n► Running pytest (quick check)..."
if command -v pytest &> /dev/null; then
    pytest tests/smoke/test_api_startup.py -v 2>/dev/null && echo "✅ Smoke tests pass" || echo "⚠️  Smoke tests need attention"
else
    echo "⚠️  pytest not installed"
fi

# 3. Check commit history
echo -e "\n3. Recent commits:"
echo "------------------"
git log --oneline -10

# 4. Check remote status
echo -e "\n4. Remote repository status:"
echo "----------------------------"
git remote -v
echo ""
git branch -vv

echo -e "\n==================================="
echo "Ready to push? Review the above results."
echo "==================================="
