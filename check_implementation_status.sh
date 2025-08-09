#!/bin/bash
# GenomeVault Implementation Status Check

echo "================================================"
echo "GenomeVault Safe Fix Implementation Status"
echo "================================================"
echo ""

# Check git branch
echo "Current Branch:"
git branch --show-current
echo ""

# Check for syntax errors quickly
echo "Checking Python Syntax..."
python -m py_compile genomevault/**/*.py 2>&1 | head -5
if [ $? -eq 0 ]; then
    echo "✅ No major syntax errors detected"
else
    echo "⚠️ Syntax errors found - see above"
fi
echo ""

# Check if main module imports
echo "Testing Main Module Import..."
python -c "import genomevault" 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Main module imports successfully"
else
    echo "⚠️ Import error - module structure needs fixing"
fi
echo ""

# Count files with issues
echo "Quick Issue Count:"
echo -n "  Files with 'NotImplementedError': "
grep -r "NotImplementedError" genomevault --include="*.py" 2>/dev/null | wc -l

echo -n "  Files with 'print(' statements: "
grep -r "print(" genomevault --include="*.py" 2>/dev/null | grep -v "test" | wc -l

echo -n "  Files with 'pdb' references: "
grep -r "pdb" genomevault --include="*.py" 2>/dev/null | wc -l
echo ""

# Check for test files
echo "Test Infrastructure:"
if [ -d "tests" ]; then
    echo -n "  Test files found: "
    find tests -name "test_*.py" | wc -l

    # Try running a quick test
    echo "  Running quick test check..."
    python -m pytest tests -q --tb=no --maxfail=1 2>&1 | head -5
else
    echo "  ⚠️ No tests directory found"
fi
echo ""

# Show recent changes
echo "Recent Git Changes:"
git status --short | head -10
echo ""

echo "================================================"
echo "Next Steps:"
echo "1. Run: python genomevault_safe_fix_implementation.py"
echo "2. Review changes: git diff"
echo "3. Run tests: pytest tests/"
echo "4. Commit: git add -A && git commit -m 'Fix audit issues'"
echo "5. Push: git push origin clean-slate"
echo "================================================"
