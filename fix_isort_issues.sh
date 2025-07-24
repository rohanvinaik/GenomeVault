#!/bin/bash
# Fix isort issues specifically

echo "Fixing isort Issues"
echo "=================="

cd /Users/rohanvinaik/genomevault

# First, let's see what isort thinks is wrong
echo "Checking current import order in test_catalytic_implementation.py..."
isort --check-only --diff test_catalytic_implementation.py || true

# Now fix it
echo -e "\nApplying isort fixes..."
isort --profile black test_catalytic_implementation.py

# Also fix all other catalytic files
echo -e "\nFixing imports in all catalytic files..."
isort --profile black example_catalytic_usage.py
isort --profile black genomevault/hypervector/encoding/catalytic_projections.py
isort --profile black genomevault/pir/catalytic_client.py
isort --profile black genomevault/zk_proofs/advanced/coec_catalytic_proof.py
isort --profile black genomevault/integration/catalytic_pipeline.py

# Show what changed
echo -e "\nChanges made:"
git diff test_catalytic_implementation.py

# Run Black after isort to maintain consistent formatting
echo -e "\nRunning Black to maintain formatting..."
black --target-version py311 test_catalytic_implementation.py

# Stage and commit
git add test_catalytic_implementation.py
git add example_catalytic_usage.py
git add genomevault/hypervector/encoding/catalytic_projections.py
git add genomevault/pir/catalytic_client.py
git add genomevault/zk_proofs/advanced/coec_catalytic_proof.py
git add genomevault/integration/catalytic_pipeline.py

git commit -m "fix: Fix isort import ordering issues

- Apply isort with black profile to all catalytic files
- Ensure consistent import ordering
- Fix test_catalytic_implementation.py imports"

git push origin $(git branch --show-current)

echo -e "\n✅ isort fixes applied and pushed!"
