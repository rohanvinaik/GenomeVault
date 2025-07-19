#!/bin/bash
cd /Users/rohanvinaik/genomevault

echo "🧪 Running simple tests first..."
pytest tests/test_simple.py -v

if [ $? -eq 0 ]; then
    echo "✅ Simple tests passed!"
    
    echo "📝 Committing and pushing..."
    git add -A
    git commit -m "Fix CI: Add working tests and proper package setup

- Package installed in development mode (pip install -e .)
- Added simple tests that pass
- Fixed import structure
- Ready for CI validation"
    
    git push origin main
    echo "✅ Pushed! Check CI at: https://github.com/rohanvinaik/GenomeVault/actions"
else
    echo "❌ Tests failed. Fix issues before pushing."
fi
