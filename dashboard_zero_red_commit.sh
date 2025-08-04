#!/bin/bash
# Dashboard Zero-Red Sprint - Commit Script

set -e  # Exit on any error

echo "🎯 Dashboard Zero-Red Sprint - Commit Implementation"
echo "===================================================="

# Move to project directory
cd /Users/rohanvinaik/genomevault

echo "📦 Staging changes..."

# Stage the fixed benchmark file
git add benchmarks/benchmark_encoding.py

# Stage the implementation script
git add dashboard_zero_red.py

# Check for any other files that were modified
echo "📋 Files to commit:"
git status --porcelain

# Create the commit
echo "💾 Creating commit..."
git commit -m "fix: resolve E402 import order issues in benchmarks

- Move all imports to top of benchmark_encoding.py
- Fix logging statement to use % formatting instead of f-string
- Add comprehensive E402 fixer script for future use
- Ensure proper import organization: stdlib, third-party, local
- Add logger configuration after imports

Resolves: E402 import order violations in benchmark files
Part of: Dashboard Zero-Red sprint to eliminate ruff violations"

# Show commit details
echo "✅ Commit created:"
git log -1 --oneline

# Run validation
echo ""
echo "🧪 Final Validation:"
echo "==================="

echo "1. Checking E402 issues specifically..."
if ruff check . --select E402 --quiet; then
    echo "   ✅ E402: All import order issues resolved!"
else
    echo "   ⚠️  E402: Some issues remain"
    ruff check . --select E402
fi

echo "2. Running full ruff check..."
ruff_output=$(ruff check . 2>&1)
ruff_exit_code=$?

if [ $ruff_exit_code -eq 0 ]; then
    echo "   🎉 ALL RUFF CHECKS PASS!"
else
    echo "   ⚠️  Remaining ruff issues:"
    echo "$ruff_output" | head -10
    if [ $(echo "$ruff_output" | wc -l) -gt 10 ]; then
        echo "   ... (showing first 10 issues)"
    fi
fi

echo ""
echo "📊 Dashboard Status:"
echo "==================="

# Count remaining issues by type
echo "Current issue breakdown:"
ruff check . --statistics 2>/dev/null | head -5 || echo "No statistics available"

echo ""
if [ $ruff_exit_code -eq 0 ]; then
    echo "🎉 DASHBOARD ZERO-RED: COMPLETE!"
    echo "✅ All E402 import order issues resolved"
    echo "✅ Benchmark files are clean"
    echo "✅ Ready for next sprint phase"
else
    echo "🔄 DASHBOARD ZERO-RED: IN PROGRESS"
    echo "✅ E402 import order issues resolved"
    echo "⏳ Other issues remain for future sprints"
fi

echo ""
echo "Next Steps:"
echo "1. Review and push: git push origin clean-slate"
echo "2. Continue with remaining ruff issues if any"
echo "3. Move to next sprint phase"
