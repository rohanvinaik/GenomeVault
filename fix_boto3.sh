#!/bin/bash
# Fix the boto3 dependency issue

echo "🔧 Fixing boto3 dependency"
echo "========================"

cd /Users/rohanvinaik/genomevault

# Option 1: Install boto3
echo "📦 Installing boto3..."
pip install boto3

# But wait - let's check if backup.py is even needed
echo ""
echo "🔍 Checking what's importing backup..."
grep -r "from . import backup" . --include="*.py" | head -5

# Let's also check what backup.py is for
echo ""
echo "📄 Checking backup.py purpose..."
head -20 utils/backup.py

# Run a quick test
echo ""
echo "🧪 Quick test after installing boto3..."
python -m pytest tests/test_simple.py -v
