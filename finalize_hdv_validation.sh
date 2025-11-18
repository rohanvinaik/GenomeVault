#!/bin/bash
# Finalize HDV Validation - Generate Complete Package
# Run this after validate_whole_genome_hdv.py completes

set -e

echo "========================================================================"
echo "FINALIZING HDV VALIDATION PACKAGE"
echo "========================================================================"
echo ""

# Check if validation completed
if [ ! -f "WHOLE_GENOME_HDV_VALIDATION_REPORT.md" ]; then
    echo "ERROR: Validation report not found!"
    echo "Please ensure validate_whole_genome_hdv.py has completed successfully."
    exit 1
fi

echo "✓ Validation report found"
echo ""

# Generate validation package
echo "Generating validation package..."
python3 generate_hdv_validation_package.py

echo ""
echo "========================================================================"
echo "VALIDATION PACKAGE COMPLETE"
echo "========================================================================"
echo ""

# Show package location
if [ -d "HDV_VALIDATION_PACKAGE" ]; then
    echo "Package location: $(pwd)/HDV_VALIDATION_PACKAGE"
    echo ""
    echo "Package contents:"
    ls -lh HDV_VALIDATION_PACKAGE/
    echo ""
    echo "✅ Ready for distribution/review"
else
    echo "⚠ Package directory not created - check for errors above"
fi

echo ""
