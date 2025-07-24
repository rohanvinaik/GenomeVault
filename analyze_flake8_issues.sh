#!/bin/bash
# analyze_flake8_issues.sh
# Script to analyze and categorize Flake8 issues

echo "🔍 Analyzing Flake8 Issues"
echo "========================="

cd /Users/rohanvinaik/genomevault

# Get detailed Flake8 report
echo "📊 Generating detailed Flake8 report..."

# Common Flake8 error codes:
# E101-E902: PEP8 style errors
# W191-W605: PEP8 style warnings  
# F401: imported but unused
# F841: local variable assigned but never used
# E501: line too long
# W293: blank line contains whitespace

echo -e "\n📋 Flake8 Issue Summary:"
flake8 . --statistics --count || true

echo -e "\n🔧 Auto-fixable issues:"

# Count specific issues
echo -e "\nF401 (unused imports):"
flake8 . --select=F401 --count || true

echo -e "\nW293 (blank line whitespace):"
flake8 . --select=W293 --count || true

echo -e "\nE501 (line too long):"
flake8 . --select=E501 --count || true

# For unused imports, we can use autoflake
if command -v autoflake &> /dev/null; then
    echo -e "\n🔧 Running autoflake to remove unused imports..."
    autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r .
else
    echo -e "\n⚠️  autoflake not installed. Install with: pip install autoflake"
fi

# Show what needs manual fixing
echo -e "\n⚠️  Issues requiring manual fixes:"
echo "The following issues typically need manual intervention:"
echo "- E501: Lines too long (consider breaking long lines)"
echo "- Complex F8xx errors: May indicate logic issues"
echo "- E7xx/E9xx: Syntax errors that need careful review"

echo -e "\n💡 Recommendations:"
echo "1. Install autoflake: pip install autoflake"
echo "2. Use '# noqa' comments for legitimate long lines"
echo "3. Configure Flake8 in setup.cfg or .flake8 for project-specific rules"
