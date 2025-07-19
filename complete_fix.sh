#!/bin/bash
# Complete fix script for GenomeVault

echo "🚀 GenomeVault Complete Fix"
echo "============================"

# Change to the genomevault directory
cd /Users/rohanvinaik/genomevault

# Step 1: Install all missing dependencies
echo "📦 Installing missing dependencies..."
pip install scikit-learn biopython pysam pynacl pyyaml uvicorn web3 eth-account seaborn

# Step 2: Install pydantic-settings if not already installed
echo "📦 Ensuring pydantic-settings is installed..."
pip install pydantic-settings pydantic>=2.0.0

# Step 3: Create a proper Python package structure
echo "🔧 Setting up Python package structure..."

# Add __init__.py to utils if it doesn't exist
touch utils/__init__.py

# Create a setup.py for proper package installation
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="genomevault",
    version="3.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "biopython>=1.79",
        "pysam>=0.17.0",
        "cryptography>=36.0.0",
        "pynacl>=1.5.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pyyaml>=6.0",
        "click>=8.0",
        "fastapi>=0.85.0",
        "uvicorn>=0.18.0",
        "httpx>=0.23.0",
        "web3>=5.31.0",
        "eth-account>=0.5.9",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)
EOF

# Step 4: Install the package in development mode
echo "📦 Installing GenomeVault in development mode..."
pip install -e .

# Step 5: Run tests
echo "🧪 Running tests..."
python -m pytest tests/test_simple.py -v

echo ""
echo "✅ Fix complete!"
echo ""
echo "To verify everything is working:"
echo "  python -m pytest tests/ -v"
echo ""
echo "To use GenomeVault modules:"
echo "  from genomevault.core.config import Config"
echo "  from genomevault.local_processing import SequencingProcessor"
