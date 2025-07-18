#!/bin/bash

# GenomeVault Node Initialization Script

set -e

echo "================================================="
echo "    GenomeVault 3.0 Node Initialization"
echo "================================================="

# Parse command line arguments
NODE_TYPE="light"
SIGNATORY="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            NODE_TYPE="$2"
            shift 2
            ;;
        --signatory)
            SIGNATORY="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --type [light|full|archive]  Set node type (default: light)"
            echo "  --signatory                  Enable trusted signatory status"
            echo "  --help                       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate node type
if [[ ! "$NODE_TYPE" =~ ^(light|full|archive)$ ]]; then
    echo "Error: Invalid node type. Must be light, full, or archive."
    exit 1
fi

# Set node class weight based on type
case $NODE_TYPE in
    light)
        NODE_CLASS_WEIGHT=1
        ;;
    full)
        NODE_CLASS_WEIGHT=4
        ;;
    archive)
        NODE_CLASS_WEIGHT=8
        ;;
esac

echo ""
echo "Configuration:"
echo "  Node Type: $NODE_TYPE"
echo "  Node Class Weight: $NODE_CLASS_WEIGHT"
echo "  Signatory Status: $SIGNATORY"

# Calculate voting power
if [ "$SIGNATORY" = "true" ]; then
    VOTING_POWER=$((NODE_CLASS_WEIGHT + 10))
    echo "  Total Voting Power: $VOTING_POWER (includes +10 signatory weight)"
else
    VOTING_POWER=$NODE_CLASS_WEIGHT
    echo "  Total Voting Power: $VOTING_POWER"
fi

echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{input,output,cache}
mkdir -p logs
mkdir -p config
mkdir -p keys

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    
    # Update .env with node configuration
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/NODE_TYPE=.*/NODE_TYPE=$NODE_TYPE/" .env
        sed -i '' "s/SIGNATORY_STATUS=.*/SIGNATORY_STATUS=$SIGNATORY/" .env
        sed -i '' "s/NODE_CLASS_WEIGHT=.*/NODE_CLASS_WEIGHT=$NODE_CLASS_WEIGHT/" .env
    else
        # Linux
        sed -i "s/NODE_TYPE=.*/NODE_TYPE=$NODE_TYPE/" .env
        sed -i "s/SIGNATORY_STATUS=.*/SIGNATORY_STATUS=$SIGNATORY/" .env
        sed -i "s/NODE_CLASS_WEIGHT=.*/NODE_CLASS_WEIGHT=$NODE_CLASS_WEIGHT/" .env
    fi
    
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env file to set:"
    echo "   - JWT_SECRET"
    echo "   - ENCRYPTION_KEY"
    echo "   - Database credentials"
    echo "   - Other security-sensitive values"
fi

# Generate keys if they don't exist
if [ ! -f keys/node_key.pem ]; then
    echo ""
    echo "Generating node keys..."
    openssl genpkey -algorithm RSA -out keys/node_key.pem -pkeyopt rsa_keygen_bits:4096
    openssl rsa -pubout -in keys/node_key.pem -out keys/node_key.pub
    chmod 600 keys/node_key.pem
    echo "✓ Node keys generated"
fi

# Check for Python environment
echo ""
echo "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python $PYTHON_VERSION found"

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo ""
echo "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"

# Check Docker
echo ""
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. Docker is required for local processing containers."
    echo "   Please install Docker: https://docs.docker.com/get-docker/"
else
    echo "✓ Docker found"
    
    # Check if Docker daemon is running
    if ! docker info > /dev/null 2>&1; then
        echo "⚠️  Docker daemon is not running. Please start Docker."
    else
        echo "✓ Docker daemon is running"
    fi
fi

# HIPAA compliance check for signatory nodes
if [ "$SIGNATORY" = "true" ]; then
    echo ""
    echo "================================================="
    echo "    HIPAA Compliance Check for Signatory Node"
    echo "================================================="
    echo ""
    echo "To enable trusted signatory status, you must provide:"
    echo "  1. NPI Number"
    echo "  2. Business Associate Agreement (BAA) hash"
    echo "  3. Risk Analysis hash"
    echo "  4. Hardware Security Module (HSM) serial number"
    echo ""
    echo "Please update these values in your .env file."
fi

# Summary
echo ""
echo "================================================="
echo "    Initialization Complete!"
echo "================================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your configuration"
echo "  2. Build Docker containers: docker-compose build"
echo "  3. Start the node: docker-compose up -d"
echo "  4. Check API health: curl http://localhost:8000/health"
echo ""
echo "For development API server:"
echo "  source venv/bin/activate"
echo "  python -m api.app"
echo ""
echo "Documentation: docs/deployment/node-setup.md"
echo ""
