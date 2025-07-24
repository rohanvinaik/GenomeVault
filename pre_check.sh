#!/bin/bash
# Pre-check script to ensure all tools are available

echo "Pre-flight Check for Catalytic Implementation"
echo "==========================================="

# Function to check if a command exists
check_command() {
    if command -v $1 &> /dev/null; then
        echo "✓ $1 is installed"
        return 0
    else
        echo "✗ $1 is NOT installed"
        return 1
    fi
}

# Check required tools
echo -e "\nChecking required tools:"
all_good=true

if ! check_command git; then
    all_good=false
fi

if ! check_command python; then
    all_good=false
fi

if ! check_command black; then
    echo "  Install with: pip install black"
    all_good=false
fi

if ! check_command isort; then
    echo "  Install with: pip install isort"
    all_good=false
fi

if ! check_command flake8; then
    echo "  Install with: pip install flake8"
    all_good=false
fi

# Optional tools
echo -e "\nChecking optional tools:"
check_command mypy || echo "  Install with: pip install mypy (optional)"
check_command pytest || echo "  Install with: pip install pytest (optional)"

# Check Python imports
echo -e "\nChecking Python dependencies:"
python -c "
import sys
try:
    import numpy
    print('✓ numpy is installed')
except ImportError:
    print('✗ numpy is NOT installed - pip install numpy')
    sys.exit(1)

try:
    import torch
    print('✓ torch is installed')
except ImportError:
    print('✗ torch is NOT installed - pip install torch')
    sys.exit(1)

try:
    import aiohttp
    print('✓ aiohttp is installed')
except ImportError:
    print('✗ aiohttp is NOT installed - pip install aiohttp')
    sys.exit(1)

# Optional GPU support
try:
    import cupy
    print('✓ cupy is installed (GPU support enabled)')
except ImportError:
    print('⚠ cupy is NOT installed (GPU support disabled)')
    print('  For GPU acceleration: pip install cupy-cuda11x')
"

# Check if we can import genomevault
echo -e "\nChecking GenomeVault installation:"
if python -c "import genomevault" 2>/dev/null; then
    echo "✓ GenomeVault package is importable"
else
    echo "⚠ GenomeVault package not in Python path"
    echo "  You may need to run: pip install -e ."
fi

echo ""
if [ "$all_good" = true ]; then
    echo "✅ All required tools are installed!"
    echo "You can proceed with merge_validate_push.sh"
else
    echo "❌ Some required tools are missing."
    echo "Please install them before proceeding."
    exit 1
fi
