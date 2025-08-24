#!/bin/bash
# Quick setup script for GenomeVault development

set -e

echo "🧬 GenomeVault Quick Setup"
echo "========================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Python version
if ! python3 --version | grep -E "3\.(1[0-9]|[2-9][0-9])" &> /dev/null; then
    echo "Error: Python 3.10+ is required"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

# Start Docker services
if command -v docker-compose &> /dev/null; then
    echo "Starting Docker services..."
    docker-compose up -d postgres redis
    
    # Wait for services
    echo "Waiting for services to start..."
    sleep 5
fi

# Run database migrations
echo "Setting up database..."
export DATABASE_URL="postgresql://genomevault:genomevault@localhost:5432/genomevault"
python -m alembic upgrade head 2>/dev/null || echo "Database migrations skipped"

# Run tests
echo "Running basic tests..."
python -m pytest tests/unit/test_hypervector.py -v || true

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  source venv/bin/activate        # Activate virtual environment"
echo "  python -m genomevault.cli --help  # Show CLI help"
echo "  uvicorn genomevault.api.main:app --reload  # Start API server"
echo ""
echo "Quick tests:"
echo "  python test_accelerator.py      # Test accelerator"
echo "  python test_module_fixes.py     # Test module imports"
echo ""