#!/bin/bash
# Setup Groth16 for Enhanced Variant Presence Circuit
# Circuit has 117,143 constraints, requires pot20 (2^20 = 1M constraints)

set -e

CIRCUIT_NAME="variant_presence_enhanced"
CIRCUITS_DIR="/Users/rohanvinaik/genomevault/genomevault/zk/circuits/variant_presence"
BUILD_DIR="${CIRCUITS_DIR}/build"
POT_FILE="${BUILD_DIR}/pot20_final.ptau"

echo "========================================"
echo "Enhanced Circuit Setup for GenomeVault"
echo "========================================"
echo "Circuit: $CIRCUIT_NAME"
echo "Constraints: 117,143"
echo "Build Dir: $BUILD_DIR"
echo ""

# Create build directory if needed
mkdir -p "$BUILD_DIR"

# Step 1: Download Powers of Tau if not exists
if [ ! -f "$POT_FILE" ]; then
    echo "📥 [1/6] Downloading Powers of Tau (pot20 ~600MB)..."
    echo "This will take a few minutes..."
    curl --progress-bar -o "$POT_FILE" \
        https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_20.ptau || {
        echo "❌ Download failed. Trying alternative URL..."
        wget -O "$POT_FILE" \
            https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_20.ptau || {
            echo "❌ Both download methods failed."
            echo "Please manually download pot20_final.ptau to: $POT_FILE"
            exit 1
        }
    }
    echo "✅ Downloaded $(du -h $POT_FILE | cut -f1)"
else
    echo "✅ [1/6] Powers of Tau already exists ($(du -h $POT_FILE | cut -f1))"
fi

# Step 2: Verify circuit files exist
echo ""
echo "🔍 [2/6] Verifying circuit files..."
if [ ! -f "${BUILD_DIR}/${CIRCUIT_NAME}.r1cs" ]; then
    echo "❌ Circuit not compiled yet!"
    echo "Expected: ${BUILD_DIR}/${CIRCUIT_NAME}.r1cs"
    echo ""
    echo "Run this first:"
    echo "  cd /tmp/zk_test_enhanced"
    echo "  circom ${CIRCUITS_DIR}/${CIRCUIT_NAME}.circom --r1cs --wasm --sym \\"
    echo "    -l /Users/rohanvinaik/genomevault/node_modules/circomlib/circuits"
    echo "  cp variant_presence_enhanced.* ${BUILD_DIR}/"
    echo "  cp -r variant_presence_enhanced_js ${BUILD_DIR}/"
    exit 1
fi
echo "✅ Circuit files found"

# Step 3: Display circuit info
echo ""
echo "📊 [3/6] Circuit Statistics:"
snarkjs r1cs info "${BUILD_DIR}/${CIRCUIT_NAME}.r1cs" | grep -E "(Curve|Wires|Constraints|Inputs|Outputs)" || true

# Step 4: Setup witness generator (if needed)
echo ""
echo "🔧 [4/6] Setting up witness generator..."
if [ -d "${BUILD_DIR}/${CIRCUIT_NAME}_js" ]; then
    cd "${BUILD_DIR}/${CIRCUIT_NAME}_js"
    if [ ! -d "node_modules" ]; then
        echo "Installing dependencies..."
        npm install
    else
        echo "✅ Dependencies already installed"
    fi
    cd - > /dev/null
else
    echo "❌ Witness generator not found: ${BUILD_DIR}/${CIRCUIT_NAME}_js"
    exit 1
fi

# Step 5: Groth16 setup (Phase 2 ceremony)
echo ""
echo "⚙️  [5/6] Running Groth16 setup..."
echo "This will take 5-10 minutes for 117K constraints..."

ZKEY_0="${BUILD_DIR}/${CIRCUIT_NAME}_0000.zkey"
ZKEY_FINAL="${BUILD_DIR}/${CIRCUIT_NAME}_final.zkey"

if [ ! -f "$ZKEY_0" ]; then
    echo "Running groth16 setup..."
    snarkjs groth16 setup \
        "${BUILD_DIR}/${CIRCUIT_NAME}.r1cs" \
        "$POT_FILE" \
        "$ZKEY_0" 2>&1 | tail -10
    echo "✅ Initial zkey created"
else
    echo "✅ Initial zkey already exists"
fi

# Step 6: Contribute randomness
echo ""
echo "🎲 [6/6] Contributing randomness..."
if [ ! -f "$ZKEY_FINAL" ]; then
    RANDOM_ENTROPY=$(openssl rand -hex 64)
    echo "Contributing with entropy: ${RANDOM_ENTROPY:0:16}..."

    snarkjs zkey contribute \
        "$ZKEY_0" \
        "$ZKEY_FINAL" \
        --name="GenomeVault Enhanced Circuit" \
        -e="$RANDOM_ENTROPY" \
        -v 2>&1 | tail -10

    echo "✅ Final zkey created"
else
    echo "✅ Final zkey already exists"
fi

# Step 7: Export verification key
VKEY="${BUILD_DIR}/verification_key_enhanced.json"
if [ ! -f "$VKEY" ]; then
    echo ""
    echo "📤 Exporting verification key..."
    snarkjs zkey export verificationkey \
        "$ZKEY_FINAL" \
        "$VKEY"
    echo "✅ Verification key exported"
else
    echo "✅ Verification key already exists"
fi

# Summary
echo ""
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "Files created:"
echo "  🔑 Powers of Tau:     $(basename $POT_FILE) ($(du -h $POT_FILE | cut -f1))"
echo "  🔐 Proving Key:       ${CIRCUIT_NAME}_final.zkey ($(du -h $ZKEY_FINAL | cut -f1))"
echo "  ✓  Verification Key:  verification_key_enhanced.json ($(du -h $VKEY | cut -f1))"
echo ""
echo "Circuit Statistics:"
snarkjs r1cs info "${BUILD_DIR}/${CIRCUIT_NAME}.r1cs" | grep -E "Constraints"
echo ""
echo "Next Steps:"
echo "  1. Test witness generation:"
echo "     cd ${BUILD_DIR}"
echo "     node ${CIRCUIT_NAME}_js/generate_witness.js \\"
echo "       ${CIRCUIT_NAME}_js/${CIRCUIT_NAME}.wasm \\"
echo "       input.json witness.wtns"
echo ""
echo "  2. Generate a proof:"
echo "     snarkjs groth16 prove \\"
echo "       ${CIRCUIT_NAME}_final.zkey witness.wtns \\"
echo "       proof.json public.json"
echo ""
echo "  3. Run full benchmark:"
echo "     python benchmarks/zk_groth16_benchmark.py \\"
echo "       --circuit ${CIRCUIT_NAME} \\"
echo "       --iterations 10"
echo ""
echo "========================================"
