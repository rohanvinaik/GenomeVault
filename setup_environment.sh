#!/bin/bash
# Setup GenomeVault environment

echo "🧬 Setting up GenomeVault environment..."
echo "======================================"

cd ~/genomevault

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install GenomeVault in development mode
echo "Installing GenomeVault..."
if [ -f "setup.py" ]; then
    pip install -e .
else
    echo "⚠️  setup.py not found, installing requirements only..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
fi

# Install additional requirements if they exist
if [ -f "requirements-dev.txt" ]; then
    echo "Installing development requirements..."
    pip install -r requirements-dev.txt
fi

echo
echo "✅ Setup complete!"
echo
echo "To use GenomeVault, activate the environment with:"
echo "  source ~/genomevault/venv/bin/activate"
echo
echo "Then you can run experiments like:"
echo "  python examples/basic_usage.py"
