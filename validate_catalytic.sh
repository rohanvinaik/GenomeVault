#!/bin/bash
# Quick validation of catalytic implementation files

echo "Validating Catalytic Implementation Files"
echo "========================================"

cd /Users/rohanvinaik/genomevault

# Check if files exist
echo -e "\nChecking file existence:"
files=(
    "genomevault/hypervector/encoding/catalytic_projections.py"
    "genomevault/pir/catalytic_client.py"
    "genomevault/zk_proofs/advanced/coec_catalytic_proof.py"
    "genomevault/integration/catalytic_pipeline.py"
    "test_catalytic_implementation.py"
    "example_catalytic_usage.py"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(wc -l < "$file")
        echo "✓ $file (${size} lines)"
    else
        echo "✗ $file NOT FOUND"
        all_exist=false
    fi
done

# Check imports
echo -e "\nChecking Python imports:"
python3 -c "
import sys
sys.path.insert(0, '.')

errors = []

try:
    from genomevault.hypervector.encoding.catalytic_projections import CatalyticProjectionPool
    print('✓ CatalyticProjectionPool imports correctly')
except Exception as e:
    print(f'✗ CatalyticProjectionPool import error: {e}')
    errors.append(str(e))

try:
    from genomevault.pir.catalytic_client import CatalyticPIRClient
    print('✓ CatalyticPIRClient imports correctly')
except Exception as e:
    print(f'✗ CatalyticPIRClient import error: {e}')
    errors.append(str(e))

try:
    from genomevault.zk_proofs.advanced.coec_catalytic_proof import COECCatalyticProofEngine
    print('✓ COECCatalyticProofEngine imports correctly')
except Exception as e:
    print(f'✗ COECCatalyticProofEngine import error: {e}')
    errors.append(str(e))

try:
    from genomevault.integration.catalytic_pipeline import CatalyticGenomeVaultPipeline
    print('✓ CatalyticGenomeVaultPipeline imports correctly')
except Exception as e:
    print(f'✗ CatalyticGenomeVaultPipeline import error: {e}')
    errors.append(str(e))

if errors:
    print(f'\n{len(errors)} import errors found')
    sys.exit(1)
else:
    print('\nAll imports successful!')
"

# Check git status
echo -e "\nGit status:"
git status --short

echo -e "\nValidation complete!"
