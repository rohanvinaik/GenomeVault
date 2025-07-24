#!/bin/bash
# Setup script for Catalytic GenomeVault implementation

echo "Setting up Catalytic GenomeVault Implementation"
echo "=============================================="

# Ensure we're in the right directory
cd /Users/rohanvinaik/genomevault

# Create necessary directories
echo "Creating directory structure..."
mkdir -p genomevault/hypervector/gpu
mkdir -p ~/.genomevault/projections

# Check if cupy is available for GPU support
echo "Checking for GPU support..."
python -c "import cupy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ GPU support available (cupy installed)"
else
    echo "✗ GPU support not available (cupy not installed)"
    echo "  To enable GPU acceleration, install cupy:"
    echo "  pip install cupy-cuda11x  # or appropriate CUDA version"
fi

# Run the implementation script
echo ""
echo "Running implementation script..."
bash gv-catalytic-impl-script.sh

echo ""
echo "Setup complete! You can now test the implementation."
