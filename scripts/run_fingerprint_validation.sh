#!/bin/bash
set -e

# Quick fingerprint validation for E2E pipeline
# Runs a minimal version of the production-grade evaluator

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

echo "🔬 Running HDC Fingerprint Validation..."
echo "========================================"

# Check if we should run quick or full validation
if [[ "${1:-quick}" == "quick" ]]; then
    echo "Mode: Quick validation (50 subjects, 2 folds)"
    
    # Run with reduced parameters for speed
    python3 -c "
import sys
import os
sys.path.insert(0, '${PROJECT_ROOT}')

from benchmarks.secure_fingerprint_evaluation import RigorousFingerprintEvaluator, ExperimentConfig

# Quick config for pipeline (minimum 2 folds for sklearn)
config = ExperimentConfig(
    dimension=4096,
    sparsity=0.5,
    n_subjects=50,
    n_families=12,
    samples_per_subject=3,
    n_batches=5,
    split_type='subject_disjoint',
    n_folds=2,
    seed=42,
    output_dir='benchmark_results/fingerprint_quick'
)

evaluator = RigorousFingerprintEvaluator(config)
results = evaluator.run_validation()

print('\n✅ Quick validation complete!')
print(f'AUC: {results[0].auc:.3f}')
print(f'EER: {results[0].eer:.3f}')
print(f'D-prime: {results[0].d_prime:.1f}')
"

elif [[ "${1}" == "full" ]]; then
    echo "Mode: Full production validation"
    python3 benchmarks/secure_fingerprint_evaluation.py
    
else
    echo "Usage: $0 [quick|full]"
    echo "  quick: Fast validation for CI/E2E (default)"
    echo "  full:  Complete production validation"
    exit 1
fi

echo ""
echo "✅ Fingerprint validation complete!"