#!/bin/bash
# Build script for GenomeVault Rust accelerator

set -e

echo "🦀 Building GenomeVault Rust Accelerator..."

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

cd rust_accelerator

# Build in release mode
echo "Building Rust extension..."
maturin develop --release

# Run tests if they exist
if [ -f "Cargo.toml" ]; then
    echo "Running Rust tests..."
    cargo test --release
fi

cd ..

echo "✅ Rust accelerator built successfully!"
echo ""
echo "To use the accelerator in Python:"
echo "  import genomevault_accel"
echo "  result = genomevault_accel.fast_hypervector_similarity(vec1, vec2)"
