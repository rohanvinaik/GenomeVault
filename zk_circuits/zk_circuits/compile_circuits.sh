#!/bin/bash
set -e

echo "Compiling complete GenomeVault ZK circuits..."

# Create build directory
mkdir -p build keys

# Compile main variant presence circuit
echo "Compiling variant presence circuit..."
circom circuits/variant_presence.circom --r1cs --wasm --sym -o build/ --prime bn128

# Compile test circuit
echo "Compiling test circuit..."  
circom circuits/test/variant_presence_test.circom --r1cs --wasm --sym -o build/ --prime bn128

# Generate trusted setup (for testing - use ceremony for production)
echo "Generating trusted setup..."
snarkjs powersoftau new bn128 12 keys/pot12_0000.ptau -v
snarkjs powersoftau contribute keys/pot12_0000.ptau keys/pot12_0001.ptau --name="First contribution" -v -e="random entropy"
snarkjs powersoftau prepare phase2 keys/pot12_0001.ptau keys/pot12_final.ptau -v

# Generate proving and verifying keys
snarkjs groth16 setup build/variant_presence.r1cs keys/pot12_final.ptau keys/variant_presence_0000.zkey
snarkjs zkey contribute keys/variant_presence_0000.zkey keys/variant_presence_final.zkey --name="1st Contributor Name" -v -e="Additional entropy"

# Export verification key
snarkjs zkey export verificationkey keys/variant_presence_final.zkey keys/verification_key.json

echo "✅ Circuit compilation and setup complete!"
echo "Files generated:"
echo "  - build/variant_presence.r1cs"
echo "  - build/variant_presence_js/variant_presence.wasm"
echo "  - keys/variant_presence_final.zkey"
echo "  - keys/verification_key.json"
