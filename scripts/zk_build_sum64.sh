#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CIRCUIT_DIR="$ROOT/genomevault/zk/circuits/sum64"
BUILD_DIR="$CIRCUIT_DIR/build"

echo "[*] Checking tools..."
command -v circom >/dev/null 2>&1 || { echo "circom not found"; exit 1; }
command -v snarkjs >/dev/null 2>&1 || { echo "snarkjs not found"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "node not found"; exit 1; }

mkdir -p "$BUILD_DIR"

echo "[*] Compile circom -> r1cs/wasm"
circom "$CIRCUIT_DIR/sum64.circom" --r1cs --wasm --output "$BUILD_DIR"

echo "[*] Powers of tau"
snarkjs powersoftau new bn128 12 "$BUILD_DIR/pot12_0000.ptau" -v
snarkjs powersoftau contribute "$BUILD_DIR/pot12_0000.ptau" "$BUILD_DIR/pot12_final.ptau" --name "genesis" -v

echo "[*] Groth16 setup"
snarkjs groth16 setup "$BUILD_DIR/sum64.r1cs" "$BUILD_DIR/pot12_final.ptau" "$BUILD_DIR/sum64_0000.zkey"
snarkjs zkey export verificationkey "$BUILD_DIR/sum64_0000.zkey" "$BUILD_DIR/verification_key.json"
cp "$BUILD_DIR/sum64_0000.zkey" "$BUILD_DIR/sum64_final.zkey"

echo "[✓] Build finished: $BUILD_DIR"
