#!/bin/bash
cd /Users/rohanvinaik/genomevault

echo "==================================="
echo "Pre-Push Validation Check"
echo "==================================="

# Track validation results
VALIDATION_PASSED=true

# 1. Python compilation check
echo -e "\n1. Python Compilation Check:"
echo "----------------------------"
if python -m compileall . -q 2>/dev/null; then
    echo "✅ All Python files compile successfully"
else
    echo "❌ Python compilation errors found"
    VALIDATION_PASSED=false
    python -m compileall . 2>&1 | grep -E "SyntaxError|IndentationError" | head -10
fi

# 2. Ruff checks
echo -e "\n2. Ruff Linting:"
echo "----------------"
RUFF_OUTPUT=$(ruff check . 2>&1)
RUFF_EXIT=$?
if [ $RUFF_EXIT -eq 0 ]; then
    echo "✅ No linting issues"
else
    RUFF_COUNT=$(echo "$RUFF_OUTPUT" | grep -c "error:")
    echo "⚠️  Found $RUFF_COUNT linting issues (non-blocking)"
    echo "$RUFF_OUTPUT" | grep -E "F821|F401|E722" | head -5
fi

# 3. Key module imports
echo -e "\n3. Module Import Tests:"
echo "-----------------------"
MODULES=("genomevault" "genomevault.api.main" "genomevault.core.constants" "genomevault.blockchain.node")
for module in "${MODULES[@]}"; do
    if python -c "import $module" 2>/dev/null; then
        echo "✅ $module"
    else
        echo "❌ $module - import failed"
        VALIDATION_PASSED=false
    fi
done

# 4. File structure check
echo -e "\n4. Required Files Check:"
echo "------------------------"
REQUIRED_FILES=(
    ".ruff.toml"
    "mypy.ini"
    ".github/workflows/ci.yml"
    "tests/smoke/test_api_startup.py"
    "tests/unit/test_voting_power.py"
)
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        VALIDATION_PASSED=false
    fi
done

# 5. Git status
echo -e "\n5. Git Repository Status:"
echo "-------------------------"
UNCOMMITTED=$(git status --porcelain | wc -l)
if [ $UNCOMMITTED -eq 0 ]; then
    echo "✅ Working directory clean"
else
    echo "⚠️  $UNCOMMITTED uncommitted changes"
    git status --short | head -5
fi

# Show commits to be pushed
CURRENT_BRANCH=$(git branch --show-current)
UNPUSHED=$(git log origin/$CURRENT_BRANCH..HEAD --oneline | wc -l)
echo "📊 $UNPUSHED commits to push on branch: $CURRENT_BRANCH"

# 6. Summary
echo -e "\n==================================="
echo "Validation Summary:"
echo "==================================="
if [ "$VALIDATION_PASSED" = true ]; then
    echo "✅ All critical checks passed!"
    echo ""
    echo "Commits to push:"
    git log origin/$CURRENT_BRANCH..HEAD --oneline
    echo ""
    echo "Ready to push with: git push origin $CURRENT_BRANCH"
else
    echo "❌ Some critical checks failed. Please fix before pushing."
fi
