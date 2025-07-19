#!/bin/bash
# Push import fixes

cd /Users/rohanvinaik/genomevault

echo "🔧 Pushing import fixes..."

git add -A
git commit -m "Fix import issues and Pydantic compatibility

- Fixed 'ModuleNotFoundError: No module named genomevault' by installing in dev mode
- Updated Pydantic imports for v2 compatibility (BaseSettings -> pydantic-settings)
- Fixed relative import issues in local_processing and hypervector_transform
- Updated conftest.py to properly set Python path
- Added pydantic-settings to requirements"

git push origin main

echo "✅ Pushed fixes! CI should now pass."
