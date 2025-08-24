#!/bin/bash
# Install Python dependencies with locked versions

set -e

echo "📦 Installing GenomeVault Python Dependencies"
echo "==========================================="

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Upgrade pip first
echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel

# Check if requirements files exist
if [ ! -f "requirements-minimal.txt" ]; then
    echo "Generating locked requirements..."
    $PYTHON_CMD -m pip install pip-tools
    $PYTHON_CMD -m piptools compile --resolver=backtracking -o requirements-minimal.txt requirements-minimal.in
fi

# Install production dependencies
echo ""
echo "Installing production dependencies..."
$PYTHON_CMD -m pip install -r requirements-minimal.txt

echo "✅ Production dependencies installed"

# Optional: Install dev dependencies
echo ""
read -p "Install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "requirements-dev.txt" ]; then
        $PYTHON_CMD -m pip install -r requirements-dev.txt
    else
        echo "Installing common dev tools..."
        $PYTHON_CMD -m pip install pytest pytest-cov black ruff mypy
    fi
    echo "✅ Development dependencies installed"
fi

# Optional: Install GPU acceleration
echo ""
read -p "Install GPU acceleration support? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Select GPU backend:"
    echo "  1) CUDA (NVIDIA)"
    echo "  2) Metal (Apple Silicon)"
    echo "  3) ROCm (AMD)"
    read -p "Choice (1-3): " -n 1 -r
    echo

    case $REPLY in
        1)
            echo "Installing CUDA support..."
            $PYTHON_CMD -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
            $PYTHON_CMD -m pip install cupy-cuda11x
            ;;
        2)
            echo "Installing Metal support..."
            $PYTHON_CMD -m pip install mlx
            ;;
        3)
            echo "Installing ROCm support..."
            $PYTHON_CMD -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
            ;;
    esac
    echo "✅ GPU acceleration installed"
fi

echo ""
echo "==========================================="
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Run tests: pytest tests/"
echo "  2. Start API: uvicorn genomevault.api.main:app --reload"
echo "  3. Run benchmark: python benchmark_harness.py"
