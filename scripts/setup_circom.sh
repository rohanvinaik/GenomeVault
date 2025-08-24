#!/bin/bash
set -e

echo "🔧 Setting up Circom/snarkjs for native ZK compilation"
echo "====================================================="

# Configuration
CIRCOM_VERSION="2.1.6"
CIRCUITS_DIR="zk_circuits"
BUILD_DIR="zk_circuits/build"
PTAU_SIZE="15"  # Powers of Tau size (2^15 constraints)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check Node.js
check_nodejs() {
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Node.js is required but not installed.${NC}"
        echo "Install from: https://nodejs.org/"
        exit 1
    fi
    echo -e "${GREEN}✓ Node.js found: $(node --version)${NC}"
}

# Install Circom
install_circom() {
    echo -e "${BLUE}Installing Circom compiler...${NC}"

    if command -v circom &> /dev/null; then
        echo -e "${GREEN}✓ Circom already installed: $(circom --version)${NC}"
    else
        # Check for Rust/Cargo
        if ! command -v cargo &> /dev/null; then
            echo -e "${RED}Rust/Cargo is required to build Circom.${NC}"
            echo "Install from: https://rustup.rs/"
            exit 1
        fi

        # Install from source for latest version
        echo "Building Circom from source..."
        git clone https://github.com/iden3/circom.git /tmp/circom
        cd /tmp/circom
        cargo build --release
        cargo install --path circom
        cd -
        rm -rf /tmp/circom

        echo -e "${GREEN}✓ Circom installed${NC}"
    fi
}

# Install snarkjs
install_snarkjs() {
    echo -e "${BLUE}Installing snarkjs...${NC}"

    if npm list -g snarkjs &> /dev/null; then
        echo -e "${GREEN}✓ snarkjs already installed${NC}"
    else
        npm install -g snarkjs
        echo -e "${GREEN}✓ snarkjs installed${NC}"
    fi
}

# Download trusted setup
download_ptau() {
    echo -e "${BLUE}Downloading Powers of Tau trusted setup...${NC}"

    mkdir -p ${BUILD_DIR}/ptau
    cd ${BUILD_DIR}/ptau

    if [ ! -f "powersOfTau28_hez_final_${PTAU_SIZE}.ptau" ]; then
        echo "Downloading Powers of Tau file (this may take a while)..."
        wget https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_${PTAU_SIZE}.ptau
        echo -e "${GREEN}✓ Downloaded Powers of Tau (2^${PTAU_SIZE})${NC}"
    else
        echo -e "${GREEN}✓ Powers of Tau already present${NC}"
    fi

    cd - > /dev/null
}

# Create optimized circuits
create_circuits() {
    echo -e "${BLUE}Creating optimized circuits...${NC}"

    mkdir -p ${CIRCUITS_DIR}/src

    # Variant Presence Circuit
    cat > ${CIRCUITS_DIR}/src/variant_presence.circom << 'EOF'
pragma circom 2.0.0;

include "../../node_modules/circomlib/circuits/comparators.circom";
include "../../node_modules/circomlib/circuits/poseidon.circom";

template VariantPresence(maxVariants) {
    signal input variants[maxVariants][3]; // [chr, pos, alt]
    signal input query[3]; // [chr, pos, alt]
    signal output found;

    component isEqual[maxVariants][3];
    component allEqual[maxVariants];
    signal matches[maxVariants];

    // Check each variant
    for (var i = 0; i < maxVariants; i++) {
        for (var j = 0; j < 3; j++) {
            isEqual[i][j] = IsEqual();
            isEqual[i][j].in[0] <== variants[i][j];
            isEqual[i][j].in[1] <== query[j];
        }

        // All components must match
        allEqual[i] = IsEqual();
        allEqual[i].in[0] <== isEqual[i][0].out + isEqual[i][1].out + isEqual[i][2].out;
        allEqual[i].in[1] <== 3;

        matches[i] <== allEqual[i].out;
    }

    // OR all matches
    signal accumulated[maxVariants];
    accumulated[0] <== matches[0];

    for (var i = 1; i < maxVariants; i++) {
        accumulated[i] <== accumulated[i-1] + matches[i] - accumulated[i-1] * matches[i];
    }

    found <== accumulated[maxVariants - 1];
}

component main = VariantPresence(100);
EOF

    # PRS Calculation Circuit
    cat > ${CIRCUITS_DIR}/src/prs_calculation.circom << 'EOF'
pragma circom 2.0.0;

include "../../node_modules/circomlib/circuits/comparators.circom";

template PRSCalculation(numVariants) {
    signal input genotypes[numVariants]; // 0, 1, or 2
    signal input weights[numVariants];    // Effect sizes
    signal output score;

    signal products[numVariants];
    signal accumulated[numVariants];

    // Calculate weighted sum
    for (var i = 0; i < numVariants; i++) {
        products[i] <== genotypes[i] * weights[i];
    }

    // Accumulate
    accumulated[0] <== products[0];
    for (var i = 1; i < numVariants; i++) {
        accumulated[i] <== accumulated[i-1] + products[i];
    }

    score <== accumulated[numVariants - 1];
}

component main = PRSCalculation(500);
EOF

    # Diabetes Risk Alert Circuit
    cat > ${CIRCUITS_DIR}/src/diabetes_risk.circom << 'EOF'
pragma circom 2.0.0;

include "../../node_modules/circomlib/circuits/comparators.circom";

template DiabetesRisk(numFactors) {
    signal input risk_factors[numFactors];
    signal input thresholds[3]; // low, medium, high
    signal output risk_level; // 0=low, 1=medium, 2=high

    component gt_high[numFactors];
    component gt_medium[numFactors];

    signal high_count;
    signal medium_count;

    var high_sum = 0;
    var medium_sum = 0;

    for (var i = 0; i < numFactors; i++) {
        gt_high[i] = GreaterThan(10);
        gt_high[i].in[0] <== risk_factors[i];
        gt_high[i].in[1] <== thresholds[2];
        high_sum += gt_high[i].out;

        gt_medium[i] = GreaterThan(10);
        gt_medium[i].in[0] <== risk_factors[i];
        gt_medium[i].in[1] <== thresholds[1];
        medium_sum += gt_medium[i].out;
    }

    high_count <== high_sum;
    medium_count <== medium_sum;

    // Determine risk level
    component is_high = GreaterThan(10);
    is_high.in[0] <== high_count;
    is_high.in[1] <== numFactors \ 3; // More than 1/3 high

    component is_medium = GreaterThan(10);
    is_medium.in[0] <== medium_count;
    is_medium.in[1] <== numFactors \ 2; // More than 1/2 medium

    risk_level <== is_high.out * 2 + (1 - is_high.out) * is_medium.out;
}

component main = DiabetesRisk(10);
EOF

    echo -e "${GREEN}✓ Circuits created${NC}"
}

# Create sample inputs
create_sample_inputs() {
    echo -e "${BLUE}Creating sample input files...${NC}"

    # Variant presence input
    cat > ${CIRCUITS_DIR}/input_variant_presence.json << 'EOF'
{
    "variants": [
        [1, 1000, 1],
        [1, 2000, 2],
        [2, 3000, 1]
    ],
    "query": [1, 1000, 1]
}
EOF

    # PRS calculation input
    cat > ${CIRCUITS_DIR}/input_prs_calculation.json << 'EOF'
{
    "genotypes": [1, 0, 2, 1, 1],
    "weights": [0.3, 0.1, 0.5, 0.2, 0.4]
}
EOF

    # Diabetes risk input
    cat > ${CIRCUITS_DIR}/input_diabetes_risk.json << 'EOF'
{
    "risk_factors": [45, 65, 80, 55, 70, 40, 85, 60, 75, 50],
    "thresholds": [30, 60, 80]
}
EOF

    echo -e "${GREEN}✓ Sample inputs created${NC}"
}

# Compile circuits
compile_circuits() {
    echo -e "${BLUE}Compiling circuits...${NC}"

    cd ${CIRCUITS_DIR}

    # Install circomlib if not present
    if [ ! -d "node_modules/circomlib" ]; then
        npm init -y > /dev/null 2>&1
        npm install circomlib
    fi

    # Compile each circuit
    for circuit in src/*.circom; do
        name=$(basename $circuit .circom)
        echo "Compiling $name..."

        circom $circuit \
            --r1cs \
            --wasm \
            --sym \
            -o build/ \
            -l node_modules

        echo -e "${GREEN}✓ Compiled $name${NC}"
    done

    cd - > /dev/null
}

# Generate verification keys
generate_keys() {
    echo -e "${BLUE}Generating verification keys...${NC}"

    cd ${BUILD_DIR}

    for r1cs in *.r1cs; do
        name=$(basename $r1cs .r1cs)

        if [ ! -f "${name}_final.zkey" ]; then
            echo "Generating keys for $name..."

            # Setup
            snarkjs groth16 setup $r1cs ptau/powersOfTau28_hez_final_${PTAU_SIZE}.ptau ${name}_0000.zkey

            # Contribute to ceremony
            echo "random_contribution_${name}" | snarkjs zkey contribute ${name}_0000.zkey ${name}_0001.zkey --name="GenomeVault" -v

            # Verify and finalize
            snarkjs zkey verify $r1cs ptau/powersOfTau28_hez_final_${PTAU_SIZE}.ptau ${name}_0001.zkey
            snarkjs zkey beacon ${name}_0001.zkey ${name}_final.zkey 0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f 10 -n="Final Beacon"

            # Export verification key
            snarkjs zkey export verificationkey ${name}_final.zkey ${name}_verification_key.json

            # Clean up intermediate files
            rm ${name}_0000.zkey ${name}_0001.zkey

            echo -e "${GREEN}✓ Generated keys for $name${NC}"
        else
            echo -e "${GREEN}✓ Keys already exist for $name${NC}"
        fi
    done

    cd - > /dev/null
}

# Create benchmark script
create_benchmark() {
    echo -e "${BLUE}Creating benchmark script...${NC}"

    cat > ${CIRCUITS_DIR}/benchmark.js << 'EOF'
const snarkjs = require("snarkjs");
const fs = require("fs");

async function benchmark(circuitName, input) {
    console.log(`\nBenchmarking ${circuitName}...`);

    const wasmPath = `build/${circuitName}_js/${circuitName}.wasm`;
    const zkeyPath = `build/${circuitName}_final.zkey`;
    const vKeyPath = `build/${circuitName}_verification_key.json`;

    // Check files exist
    if (!fs.existsSync(wasmPath)) {
        console.error(`  ❌ WASM file not found: ${wasmPath}`);
        return null;
    }
    if (!fs.existsSync(zkeyPath)) {
        console.error(`  ❌ zkey file not found: ${zkeyPath}`);
        return null;
    }

    try {
        // Witness generation and proof
        const start_witness = Date.now();
        const { proof, publicSignals } = await snarkjs.groth16.fullProve(
            input,
            wasmPath,
            zkeyPath
        );
        const witness_time = Date.now() - start_witness;

        // Verification
        const vKey = JSON.parse(fs.readFileSync(vKeyPath));
        const start_verify = Date.now();
        const res = await snarkjs.groth16.verify(vKey, publicSignals, proof);
        const verify_time = Date.now() - start_verify;

        console.log(`  Witness + Proof: ${witness_time}ms`);
        console.log(`  Verification: ${verify_time}ms`);
        console.log(`  Valid: ${res}`);
        console.log(`  Proof size: ${JSON.stringify(proof).length} bytes`);

        return { witness_time, verify_time, valid: res };
    } catch (error) {
        console.error(`  ❌ Error: ${error.message}`);
        return null;
    }
}

async function main() {
    console.log("=".repeat(60));
    console.log("ZK Circuit Benchmarks (Native Circom)");
    console.log("=".repeat(60));

    // Read input files
    const inputs = {};

    if (fs.existsSync("input_variant_presence.json")) {
        const raw = JSON.parse(fs.readFileSync("input_variant_presence.json"));
        // Pad arrays to match circuit size
        inputs.variant_presence = {
            variants: Array(100).fill([0, 0, 0]).map((v, i) =>
                i < raw.variants.length ? raw.variants[i] : v
            ),
            query: raw.query
        };
    }

    if (fs.existsSync("input_prs_calculation.json")) {
        const raw = JSON.parse(fs.readFileSync("input_prs_calculation.json"));
        inputs.prs_calculation = {
            genotypes: Array(500).fill(0).map((v, i) =>
                i < raw.genotypes.length ? raw.genotypes[i] : v
            ),
            weights: Array(500).fill(0).map((v, i) =>
                i < raw.weights.length ? raw.weights[i] : v
            )
        };
    }

    if (fs.existsSync("input_diabetes_risk.json")) {
        inputs.diabetes_risk = JSON.parse(fs.readFileSync("input_diabetes_risk.json"));
    }

    const results = {};
    for (const [circuit, input] of Object.entries(inputs)) {
        const result = await benchmark(circuit, input);
        if (result) {
            results[circuit] = result;
        }
    }

    console.log("\n" + "=".repeat(60));
    console.log("Summary");
    console.log("=".repeat(60));

    let totalTime = 0;
    let validCount = 0;

    for (const [circuit, result] of Object.entries(results)) {
        if (result) {
            totalTime += result.witness_time + result.verify_time;
            if (result.valid) validCount++;

            console.log(`${circuit}:`);
            console.log(`  Total time: ${result.witness_time + result.verify_time}ms`);
            console.log(`  Valid: ${result.valid ? "✓" : "✗"}`);
        }
    }

    console.log(`\nAll proofs valid: ${validCount === Object.keys(results).length ? "✓" : "✗"}`);
    console.log(`Total benchmark time: ${totalTime}ms`);
}

main().then(() => process.exit(0)).catch(err => {
    console.error(err);
    process.exit(1);
});
EOF

    echo -e "${GREEN}✓ Benchmark script created${NC}"
}

# Create Python integration
create_python_integration() {
    echo -e "${BLUE}Creating Python integration...${NC}"

    cat > ${CIRCUITS_DIR}/circom_prover.py << 'EOF'
#!/usr/bin/env python3
"""Python integration for native Circom circuits."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple

class CircomProver:
    """Native Circom proof generation."""

    def __init__(self, circuits_dir: str = "zk_circuits"):
        self.circuits_dir = Path(circuits_dir)
        self.build_dir = self.circuits_dir / "build"

    def generate_proof(
        self,
        circuit_name: str,
        inputs: Dict[str, Any]
    ) -> Tuple[Dict, list, bool]:
        """Generate proof using native Circom.

        Returns:
            Tuple of (proof, public_signals, valid)
        """
        # Write input to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            json.dump(inputs, f)
            input_file = f.name

        try:
            # Run snarkjs CLI
            wasm_path = self.build_dir / f"{circuit_name}_js" / f"{circuit_name}.wasm"
            zkey_path = self.build_dir / f"{circuit_name}_final.zkey"

            # Generate proof
            result = subprocess.run(
                [
                    "snarkjs", "groth16", "fullprove",
                    input_file,
                    str(wasm_path),
                    str(zkey_path),
                    "proof.json",
                    "public.json"
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Proof generation failed: {result.stderr}")

            # Read proof and public signals
            with open("proof.json") as f:
                proof = json.load(f)
            with open("public.json") as f:
                public_signals = json.load(f)

            # Verify
            vkey_path = self.build_dir / f"{circuit_name}_verification_key.json"

            result = subprocess.run(
                [
                    "snarkjs", "groth16", "verify",
                    str(vkey_path),
                    "public.json",
                    "proof.json"
                ],
                capture_output=True,
                text=True
            )

            valid = "OK" in result.stdout

            return proof, public_signals, valid

        finally:
            # Cleanup
            Path(input_file).unlink(missing_ok=True)
            Path("proof.json").unlink(missing_ok=True)
            Path("public.json").unlink(missing_ok=True)

if __name__ == "__main__":
    prover = CircomProver()

    # Test variant presence
    inputs = {
        "variants": [[1, 1000, 1]] + [[0, 0, 0]] * 99,
        "query": [1, 1000, 1]
    }

    proof, signals, valid = prover.generate_proof("variant_presence", inputs)
    print(f"Proof valid: {valid}")
    print(f"Public signals: {signals}")
EOF

    chmod +x ${CIRCUITS_DIR}/circom_prover.py
    echo -e "${GREEN}✓ Python integration created${NC}"
}

# Check installation
check_installation() {
    echo -e "${BLUE}Checking installation...${NC}"

    errors=0

    # Check circom
    if command -v circom &> /dev/null; then
        echo -e "${GREEN}✓ circom installed${NC}"
    else
        echo -e "${RED}✗ circom not found${NC}"
        errors=$((errors + 1))
    fi

    # Check snarkjs
    if npm list -g snarkjs &> /dev/null; then
        echo -e "${GREEN}✓ snarkjs installed${NC}"
    else
        echo -e "${RED}✗ snarkjs not found${NC}"
        errors=$((errors + 1))
    fi

    # Check Powers of Tau
    if [ -f "${BUILD_DIR}/ptau/powersOfTau28_hez_final_${PTAU_SIZE}.ptau" ]; then
        echo -e "${GREEN}✓ Powers of Tau present${NC}"
    else
        echo -e "${RED}✗ Powers of Tau missing${NC}"
        errors=$((errors + 1))
    fi

    # Check compiled circuits
    for circuit in variant_presence prs_calculation diabetes_risk; do
        if [ -f "${BUILD_DIR}/${circuit}.r1cs" ]; then
            echo -e "${GREEN}✓ ${circuit} compiled${NC}"
        else
            echo -e "${RED}✗ ${circuit} not compiled${NC}"
            errors=$((errors + 1))
        fi
    done

    return $errors
}

# Main execution
main() {
    echo "Starting Circom setup..."
    echo ""

    # Parse arguments
    if [ "$1" == "--check" ]; then
        check_installation
        exit $?
    fi

    # Full installation
    check_nodejs
    install_circom
    install_snarkjs
    download_ptau
    create_circuits
    create_sample_inputs
    compile_circuits
    generate_keys
    create_benchmark
    create_python_integration

    echo ""
    echo -e "${GREEN}✅ Circom setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run benchmark: cd ${CIRCUITS_DIR} && node benchmark.js"
    echo "  2. Python integration: python3 ${CIRCUITS_DIR}/circom_prover.py"
    echo "  3. Check installation: $0 --check"
    echo ""
    echo "Circuit locations:"
    echo "  Sources: ${CIRCUITS_DIR}/src/"
    echo "  Compiled: ${BUILD_DIR}/"
    echo "  Keys: ${BUILD_DIR}/*_final.zkey"
}

# Run main
main "$@"
