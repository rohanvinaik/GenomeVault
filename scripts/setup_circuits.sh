#!/bin/bash
# Setup script for GenomeVault ZK circuits

set -e

echo "🔧 Setting up GenomeVault ZK Circuits"
echo "======================================"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    echo "Please install Node.js 16+ from: https://nodejs.org/"
    exit 1
fi

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed."
    echo "Please install npm with Node.js"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo "❌ Node.js 16+ is required (found: $(node -v))"
    exit 1
fi

echo "✅ Node.js $(node -v) detected"
echo "✅ npm $(npm -v) detected"

# Install dependencies
echo ""
echo "Installing circuit dependencies..."
cd zk_circuits
npm ci --production

# Check Circom
if ! command -v circom &> /dev/null; then
    echo ""
    echo "⚠️  Circom compiler not found"
    echo "To compile circuits, install Circom:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://docs.circom.io/getting-started/installation/ | sh"
else
    echo "✅ Circom $(circom --version) detected"
fi

echo ""
echo "✅ ZK Circuits setup complete!"
echo ""
echo "Next steps:"
echo "  1. Compile circuits: cd zk_circuits && npm run compile"
echo "  2. Run tests: cd zk_circuits && npm test"
echo "  3. Run benchmarks: cd zk_circuits && npm run benchmark"
