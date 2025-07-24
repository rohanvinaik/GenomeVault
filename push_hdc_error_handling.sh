#!/bin/bash

# Push HDC error handling implementation to GitHub

echo "🚀 Pushing HDC Error Handling Implementation to GitHub"

# Add all new and modified files
echo "📄 Adding files..."
git add genomevault/hypervector/error_handling.py
git add genomevault/hypervector/__init__.py
git add genomevault/api/app.py
git add genomevault/core/constants.py
git add tests/test_hdc_error_handling.py
git add examples/hdc_error_tuning_example.py
git add HDC_INTEGRATION_SUMMARY.md

# Show status
echo -e "\n📊 Git status:"
git status

# Commit with descriptive message
echo -e "\n💾 Committing changes..."
git commit -m "feat: Implement HDC error handling with uncertainty tuning

- Add ErrorBudgetAllocator for dynamic dimension/repeat allocation
- Implement ECCEncoderMixin for self-healing hypervector codewords
- Create AdaptiveHDCEncoder extending GenomicEncoder with error budget
- Add API endpoints for budget estimation and tuned queries
- Include comprehensive test suite and examples

Mathematical foundation:
- Uses Johnson-Lindenstrauss lemma for dimension calculation
- Applies Hoeffding bounds for repeat count determination
- ECC provides quadratic variance reduction

This allows users to tune accuracy/performance tradeoff with:
- Epsilon: 0.1% to 5% relative error
- Delta: 1 in 32 to 1 in 1B failure probability
- Optional error correcting codes (XOR parity)

Closes #HDC-ERROR-HANDLING"

# Push to GitHub
echo -e "\n🌐 Pushing to GitHub..."
git push origin main

echo -e "\n✅ HDC Error Handling implementation pushed successfully!"
echo "📝 See HDC_INTEGRATION_SUMMARY.md for complete documentation"
