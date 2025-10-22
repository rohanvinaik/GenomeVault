#!/bin/bash
# Real ZK Proof Benchmark - GenomeVault
# Compiles Circom circuits and generates actual Groth16/PLONK proofs with timing

set -e

CIRCUIT_DIR="/Users/rohanvinaik/genomevault/genomevault/zk/circuits"
RESULTS_DIR="/Users/rohanvinaik/genomevault/benchmark_results/zk_proofs_real"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "GenomeVault Real ZK Proof Benchmark"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Circuit Dir: $CIRCUIT_DIR"
echo "Results Dir: $RESULTS_DIR"
echo ""

# Check if powers of tau file exists, if not download/generate
PTAU_FILE="/tmp/powersOfTau28_hez_final_12.ptau"
if [ ! -f "$PTAU_FILE" ]; then
    echo "Powers of Tau file not found. Downloading..."
    wget https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_12.ptau -O "$PTAU_FILE" || {
        echo "Failed to download, generating with snarkjs..."
        snarkjs powersoftau new bn128 12 pot12_0000.ptau
        snarkjs powersoftau prepare phase2 pot12_0000.ptau "$PTAU_FILE"
        rm pot12_0000.ptau
    }
fi

# Benchmark each circuit
CIRCUITS=("variant_presence" "diabetes_risk" "variant_simple")

for CIRCUIT_NAME in "${CIRCUITS[@]}"; do
    CIRCUIT_PATH="$CIRCUIT_DIR/$CIRCUIT_NAME/$CIRCUIT_NAME.circom"

    if [ ! -f "$CIRCUIT_PATH" ]; then
        echo "⚠️  Circuit not found: $CIRCUIT_PATH"
        continue
    fi

    echo "=========================================="
    echo "Circuit: $CIRCUIT_NAME"
    echo "=========================================="

    BUILD_DIR="$CIRCUIT_DIR/$CIRCUIT_NAME/build_$TIMESTAMP"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Step 1: Compile circuit
    echo ""
    echo "[1/7] Compiling circuit..."
    START_COMPILE=$(date +%s.%N)
    circom "$CIRCUIT_PATH" --r1cs --wasm --sym 2>&1 | tail -10
    END_COMPILE=$(date +%s.%N)
    COMPILE_TIME=$(echo "$END_COMPILE - $START_COMPILE" | bc)
    echo "    ✓ Compiled in ${COMPILE_TIME}s"

    # Get circuit info
    echo ""
    echo "[2/7] Circuit Statistics:"
    snarkjs r1cs info "${CIRCUIT_NAME}.r1cs" > circuit_info.txt 2>&1
    CONSTRAINTS=$(grep "# of Constraints" circuit_info.txt | awk '{print $NF}')
    WIRES=$(grep "# of Wires" circuit_info.txt | awk '{print $NF}')
    echo "    Constraints: $CONSTRAINTS"
    echo "    Wires: $WIRES"

    # Create test input
    echo ""
    echo "[3/7] Creating test input..."
    cat > input.json << EOF
{
  "variant_hash": "12345678901234567890123456789012",
  "reference_hash": "98765432109876543210987654321098",
  "commitment_root": "11111111111111111111111111111111",
  "chr": "1",
  "position": "100000",
  "ref_allele": "65",
  "alt_allele": "67",
  "merkle_proof": ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"],
  "merkle_indices": ["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],
  "witness_randomness": "42"
}
EOF

    # Calculate witness
    echo ""
    echo "[4/7] Calculating witness..."
    START_WITNESS=$(date +%s.%N)
    node "${CIRCUIT_NAME}_js/generate_witness.js" "${CIRCUIT_NAME}_js/${CIRCUIT_NAME}.wasm" input.json witness.wtns 2>&1 | tail -5
    END_WITNESS=$(date +%s.%N)
    WITNESS_TIME=$(echo "$END_WITNESS - $START_WITNESS" | bc)
    echo "    ✓ Witness calculated in ${WITNESS_TIME}s"

    # ==== GROTH16 ====
    echo ""
    echo "[5/7] GROTH16 Setup..."
    START_GROTH16_SETUP=$(date +%s.%N)
    snarkjs groth16 setup "${CIRCUIT_NAME}.r1cs" "$PTAU_FILE" "circuit_groth16.zkey" 2>&1 | tail -5
    END_GROTH16_SETUP=$(date +%s.%N)
    GROTH16_SETUP_TIME=$(echo "$END_GROTH16_SETUP - $START_GROTH16_SETUP" | bc)
    echo "    ✓ Setup in ${GROTH16_SETUP_TIME}s"

    echo ""
    echo "[6/7] GROTH16 Proving..."
    START_GROTH16_PROVE=$(date +%s.%N)
    snarkjs groth16 prove "circuit_groth16.zkey" witness.wtns proof_groth16.json public_groth16.json 2>&1 | tail -5
    END_GROTH16_PROVE=$(date +%s.%N)
    GROTH16_PROVE_TIME=$(echo "$END_GROTH16_PROVE - $START_GROTH16_PROVE" | bc)
    GROTH16_PROVE_MS=$(echo "$GROTH16_PROVE_TIME * 1000" | bc)
    echo "    ✓ Proof generated in ${GROTH16_PROVE_MS}ms"

    # Get proof size
    GROTH16_PROOF_SIZE=$(wc -c < proof_groth16.json)

    # Extract verification key
    snarkjs zkey export verificationkey "circuit_groth16.zkey" vkey_groth16.json

    # Verify proof
    echo ""
    echo "[7/7] GROTH16 Verification..."
    START_GROTH16_VERIFY=$(date +%s.%N)
    snarkjs groth16 verify vkey_groth16.json public_groth16.json proof_groth16.json 2>&1 | tail -5
    END_GROTH16_VERIFY=$(date +%s.%N)
    GROTH16_VERIFY_TIME=$(echo "$END_GROTH16_VERIFY - $START_GROTH16_VERIFY" | bc)
    GROTH16_VERIFY_MS=$(echo "$GROTH16_VERIFY_TIME * 1000" | bc)
    echo "    ✓ Verified in ${GROTH16_VERIFY_MS}ms"

    # ==== PLONK ====
    echo ""
    echo "[8/10] PLONK Setup..."
    START_PLONK_SETUP=$(date +%s.%N)
    snarkjs plonk setup "${CIRCUIT_NAME}.r1cs" "$PTAU_FILE" "circuit_plonk.zkey" 2>&1 | tail -5
    END_PLONK_SETUP=$(date +%s.%N)
    PLONK_SETUP_TIME=$(echo "$END_PLONK_SETUP - $START_PLONK_SETUP" | bc)
    echo "    ✓ Setup in ${PLONK_SETUP_TIME}s"

    echo ""
    echo "[9/10] PLONK Proving..."
    START_PLONK_PROVE=$(date +%s.%N)
    snarkjs plonk prove "circuit_plonk.zkey" witness.wtns proof_plonk.json public_plonk.json 2>&1 | tail -5
    END_PLONK_PROVE=$(date +%s.%N)
    PLONK_PROVE_TIME=$(echo "$END_PLONK_PROVE - $START_PLONK_PROVE" | bc)
    PLONK_PROVE_MS=$(echo "$PLONK_PROVE_TIME * 1000" | bc)
    echo "    ✓ Proof generated in ${PLONK_PROVE_MS}ms"

    PLONK_PROOF_SIZE=$(wc -c < proof_plonk.json)

    # Verify PLONK
    snarkjs zkey export verificationkey "circuit_plonk.zkey" vkey_plonk.json
    echo ""
    echo "[10/10] PLONK Verification..."
    START_PLONK_VERIFY=$(date +%s.%N)
    snarkjs plonk verify vkey_plonk.json public_plonk.json proof_plonk.json 2>&1 | tail -5
    END_PLONK_VERIFY=$(date +%s.%N)
    PLONK_VERIFY_TIME=$(echo "$END_PLONK_VERIFY - $START_PLONK_VERIFY" | bc)
    PLONK_VERIFY_MS=$(echo "$PLONK_VERIFY_TIME * 1000" | bc)
    echo "    ✓ Verified in ${PLONK_VERIFY_MS}ms"

    # Generate JSON results
    cat > "results_$CIRCUIT_NAME.json" << EOF
{
  "circuit": "$CIRCUIT_NAME",
  "timestamp": "$TIMESTAMP",
  "constraints": $CONSTRAINTS,
  "wires": $WIRES,
  "compile_time_s": $COMPILE_TIME,
  "witness_time_s": $WITNESS_TIME,
  "groth16": {
    "setup_time_s": $GROTH16_SETUP_TIME,
    "prove_time_ms": $GROTH16_PROVE_MS,
    "verify_time_ms": $GROTH16_VERIFY_MS,
    "proof_size_bytes": $GROTH16_PROOF_SIZE
  },
  "plonk": {
    "setup_time_s": $PLONK_SETUP_TIME,
    "prove_time_ms": $PLONK_PROVE_MS,
    "verify_time_ms": $PLONK_VERIFY_MS,
    "proof_size_bytes": $PLONK_PROOF_SIZE
  }
}
EOF

    # Copy results to benchmark directory
    cp "results_$CIRCUIT_NAME.json" "$RESULTS_DIR/"

    echo ""
    echo "=========================================="
    echo "Results for $CIRCUIT_NAME:"
    echo "=========================================="
    echo "Constraints: $CONSTRAINTS"
    echo "Groth16 Proving: ${GROTH16_PROVE_MS}ms"
    echo "Groth16 Verification: ${GROTH16_VERIFY_MS}ms"
    echo "Groth16 Proof Size: ${GROTH16_PROOF_SIZE} bytes"
    echo "PLONK Proving: ${PLONK_PROVE_MS}ms"
    echo "PLONK Verification: ${PLONK_VERIFY_MS}ms"
    echo "PLONK Proof Size: ${PLONK_PROOF_SIZE} bytes"
    echo ""

    cd - > /dev/null
done

echo "=========================================="
echo "All benchmarks complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
