#!/bin/bash
# Quick status check for catalytic implementation

echo "Catalytic Implementation Status Check"
echo "===================================="

cd /Users/rohanvinaik/genomevault

echo -e "\n1. Checking if files exist:"
files=(
    "genomevault/hypervector/encoding/catalytic_projections.py"
    "genomevault/pir/catalytic_client.py" 
    "genomevault/zk_proofs/advanced/coec_catalytic_proof.py"
    "genomevault/integration/catalytic_pipeline.py"
    "test_catalytic_implementation.py"
    "example_catalytic_usage.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file NOT FOUND"
    fi
done

echo -e "\n2. Git status:"
git status --short

echo -e "\n3. Current branch:"
git branch --show-current

echo -e "\n4. Remote URL:"
git remote -v | grep origin | head -1

echo -e "\n5. Recent commits:"
git log --oneline -5

echo -e "\nStatus check complete!"
