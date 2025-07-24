#!/bin/bash
# fix_formatting_and_push.sh
# Script to fix Black formatting issues and push

echo "🔧 Fixing Black Formatting Issues"
echo "================================="

cd /Users/rohanvinaik/genomevault

# Run Black to fix formatting
echo "📝 Running Black formatter..."
black .

# Check if there were changes
if git diff --quiet; then
    echo "✅ No formatting changes needed!"
else
    echo "📋 Files formatted:"
    git diff --name-only
    
    # Add the formatted files
    echo -e "\n📦 Adding formatted files..."
    git add -A
    
    # Commit the formatting fixes
    echo -e "\n💾 Committing formatting fixes..."
    git commit -m "Fix Black formatting issues

- Format genomevault/version.py
- Format genomevault/zk_proofs/circuits/__init__.py
- Format genomevault/zk_proofs/service.py
- Format tests/adversarial/test_zk_adversarial.py
- Format tests/e2e/test_zk_e2e.py
- Format tests/property/test_zk_properties.py
- Format tests/unit/test_zk_basic.py

Applied Black formatter to ensure consistent code style."
    
    # Push the changes
    echo -e "\n🌐 Pushing formatting fixes to GitHub..."
    git push origin main
    
    echo -e "\n✨ Formatting fixes pushed successfully!"
fi

# Run a final check
echo -e "\n🔍 Running final Black check..."
black --check .

if [ $? -eq 0 ]; then
    echo "✅ All files are now properly formatted!"
else
    echo "⚠️  Some files may still need formatting. Please check the output above."
fi
