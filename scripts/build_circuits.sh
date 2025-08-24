#!/usr/bin/env bash

################################################################################
# Build Script for Zero-Knowledge Circuits
#
# This script compiles Circom circuits, generates proving/verification keys,
# and creates Python wrappers for the GenomeVault project.
#
# Requirements:
#   - circom (v2.0+)
#   - snarkjs
#   - Node.js (v14+)
#
# Usage:
#   ./scripts/build_circuits.sh [options]
#
# Options:
#   --clean    Clean build artifacts before building
#   --force    Force rebuild even if artifacts exist
#   --verbose  Enable verbose output
#   --help     Show this help message
################################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CIRCUITS_DIR="$PROJECT_ROOT/genomevault/zk/circuits"
SUM64_DIR="$CIRCUITS_DIR/sum64"
BUILD_DIR="$SUM64_DIR/build"
PTAU_FILE="$BUILD_DIR/pot15_final.ptau"

# Command line options
CLEAN=false
FORCE=false
VERBOSE=false

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${MAGENTA}[STEP]${NC} $1"
}

# Print usage information
print_usage() {
    cat << EOF
Usage: $0 [options]

Options:
    --clean    Clean build artifacts before building
    --force    Force rebuild even if artifacts exist
    --verbose  Enable verbose output
    --help     Show this help message

This script builds Zero-Knowledge circuits for the GenomeVault project.
It compiles Circom circuits, generates proving/verification keys, and
creates Python wrappers for easy integration.

Requirements:
    - circom (v2.0+)
    - snarkjs
    - Node.js (v14+)
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                CLEAN=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

# Check if a command exists
check_command() {
    local cmd=$1
    if ! command -v "$cmd" &> /dev/null; then
        return 1
    fi
    return 0
}

# Check required dependencies
check_dependencies() {
    log_step "Checking dependencies..."

    local missing_deps=()

    # Check for Node.js
    if ! check_command "node"; then
        missing_deps+=("node")
    else
        local node_version=$(node --version | cut -d'v' -f2)
        log_info "Found Node.js version: $node_version"
    fi

    # Check for npm
    if ! check_command "npm"; then
        missing_deps+=("npm")
    fi

    # Check for circom
    if ! check_command "circom"; then
        log_warning "circom not found in PATH"
        # Try to find circom in common locations
        if [[ -f "$HOME/.cargo/bin/circom" ]]; then
            export PATH="$HOME/.cargo/bin:$PATH"
            log_info "Found circom in ~/.cargo/bin"
        else
            missing_deps+=("circom")
        fi
    else
        local circom_version=$(circom --version 2>&1 | head -n1)
        log_info "Found circom: $circom_version"
    fi

    # Check for snarkjs
    if ! check_command "snarkjs"; then
        log_warning "snarkjs not found globally, checking local installation..."
        if [[ -f "$PROJECT_ROOT/node_modules/.bin/snarkjs" ]]; then
            export PATH="$PROJECT_ROOT/node_modules/.bin:$PATH"
            log_info "Found snarkjs in node_modules"
        else
            missing_deps+=("snarkjs")
        fi
    else
        log_info "Found snarkjs"
    fi

    # Report missing dependencies
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        echo
        log_info "Installation instructions:"

        for dep in "${missing_deps[@]}"; do
            case $dep in
                node|npm)
                    echo "  - Install Node.js from https://nodejs.org/"
                    ;;
                circom)
                    echo "  - Install circom:"
                    echo "      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
                    echo "      git clone https://github.com/iden3/circom.git"
                    echo "      cd circom && cargo build --release"
                    echo "      cargo install --path circom"
                    ;;
                snarkjs)
                    echo "  - Install snarkjs:"
                    echo "      npm install -g snarkjs"
                    ;;
            esac
        done

        exit 1
    fi

    log_success "All dependencies found"
}

# Create necessary directories
setup_directories() {
    log_step "Setting up directories..."

    # Create build directory if it doesn't exist
    if [[ ! -d "$BUILD_DIR" ]]; then
        mkdir -p "$BUILD_DIR"
        log_info "Created build directory: $BUILD_DIR"
    fi

    # Clean if requested
    if [[ "$CLEAN" == true ]]; then
        log_info "Cleaning build artifacts..."
        rm -rf "$BUILD_DIR"/*
        log_success "Build directory cleaned"
    fi
}

# Create the sum64 circuit if it doesn't exist
create_sum64_circuit() {
    local circuit_file="$SUM64_DIR/sum64.circom"

    if [[ ! -f "$circuit_file" ]]; then
        log_step "Creating sum64.circom circuit..."

        mkdir -p "$SUM64_DIR"

        cat > "$circuit_file" << 'EOF'
pragma circom 2.0.0;

/*
 * Sum64 Circuit
 *
 * This circuit proves knowledge of two private inputs (a, b)
 * that sum to a public output (c).
 *
 * Constraints:
 *   a + b = c
 *
 * This is a simple demonstration circuit for the ZK proof system.
 */

template Sum64() {
    // Private inputs
    signal private input a;
    signal private input b;

    // Public output
    signal output c;

    // Constraint: a + b = c
    c <== a + b;
}

// Main component
component main = Sum64();
EOF

        log_success "Created sum64.circom"
    else
        log_info "sum64.circom already exists"
    fi
}

# Compile the circuit
compile_circuit() {
    log_step "Compiling circuit..."

    local circuit_file="$SUM64_DIR/sum64.circom"
    local r1cs_file="$BUILD_DIR/sum64.r1cs"
    local wasm_file="$BUILD_DIR/sum64_js/sum64.wasm"
    local sym_file="$BUILD_DIR/sum64.sym"

    # Check if compilation is needed
    if [[ -f "$r1cs_file" ]] && [[ -f "$wasm_file" ]] && [[ "$FORCE" != true ]]; then
        log_info "Circuit already compiled (use --force to rebuild)"
        return 0
    fi

    # Compile circuit
    log_info "Compiling sum64.circom..."

    if [[ "$VERBOSE" == true ]]; then
        circom "$circuit_file" \
            --r1cs \
            --wasm \
            --sym \
            --c \
            -o "$BUILD_DIR" \
            -v
    else
        circom "$circuit_file" \
            --r1cs \
            --wasm \
            --sym \
            --c \
            -o "$BUILD_DIR" \
            2>&1 | grep -E "^(template|error)" || true
    fi

    # Check compilation results
    if [[ ! -f "$r1cs_file" ]]; then
        log_error "Failed to generate R1CS file"
        exit 1
    fi

    if [[ ! -f "$wasm_file" ]]; then
        log_error "Failed to generate WASM file"
        exit 1
    fi

    # Print circuit info
    log_info "Circuit compilation complete"
    log_info "  R1CS: $r1cs_file"
    log_info "  WASM: $wasm_file"
    log_info "  Symbols: $sym_file"

    # Get circuit info
    if check_command "snarkjs"; then
        local info=$(snarkjs r1cs info "$r1cs_file" 2>/dev/null | grep -E "(Constraints|Private|Public)")
        if [[ -n "$info" ]]; then
            log_info "Circuit statistics:"
            echo "$info" | while read line; do
                echo "    $line"
            done
        fi
    fi

    log_success "Circuit compiled successfully"
}

# Download or generate Powers of Tau
setup_powers_of_tau() {
    log_step "Setting up Powers of Tau..."

    if [[ -f "$PTAU_FILE" ]] && [[ "$FORCE" != true ]]; then
        log_info "Powers of Tau file already exists"
        return 0
    fi

    # For small circuits, we can generate a small Powers of Tau
    log_info "Generating Powers of Tau (this may take a moment)..."

    # Start new ceremony
    snarkjs powersoftau new bn128 12 "$BUILD_DIR/pot12_0000.ptau" -v

    # Contribute to ceremony
    snarkjs powersoftau contribute "$BUILD_DIR/pot12_0000.ptau" \
        "$BUILD_DIR/pot12_0001.ptau" \
        --name="First contribution" -v <<< "$(date +%s)random_entropy_$(uuidgen || echo $$)"

    # Phase 2
    snarkjs powersoftau prepare phase2 "$BUILD_DIR/pot12_0001.ptau" \
        "$PTAU_FILE" -v

    # Clean up intermediate files
    rm -f "$BUILD_DIR/pot12_0000.ptau" "$BUILD_DIR/pot12_0001.ptau"

    log_success "Powers of Tau generated"
}

# Generate proving and verification keys
generate_keys() {
    log_step "Generating proving and verification keys..."

    local r1cs_file="$BUILD_DIR/sum64.r1cs"
    local zkey_file="$BUILD_DIR/sum64_final.zkey"
    local vkey_file="$BUILD_DIR/verification_key.json"

    # Check if keys exist
    if [[ -f "$zkey_file" ]] && [[ -f "$vkey_file" ]] && [[ "$FORCE" != true ]]; then
        log_info "Keys already generated (use --force to regenerate)"
        return 0
    fi

    # Setup Groth16
    log_info "Setting up Groth16 proving system..."
    snarkjs groth16 setup "$r1cs_file" "$PTAU_FILE" "$BUILD_DIR/sum64_0000.zkey"

    # Contribute to phase 2 ceremony
    log_info "Contributing to phase 2..."
    snarkjs zkey contribute "$BUILD_DIR/sum64_0000.zkey" \
        "$BUILD_DIR/sum64_0001.zkey" \
        --name="Phase 2 contribution" -v <<< "$(date +%s)more_entropy_$(uuidgen || echo $$)"

    # Export verification key
    log_info "Exporting verification key..."
    snarkjs zkey export verificationkey "$BUILD_DIR/sum64_0001.zkey" "$vkey_file"

    # Rename final zkey
    mv "$BUILD_DIR/sum64_0001.zkey" "$zkey_file"

    # Clean up
    rm -f "$BUILD_DIR/sum64_0000.zkey"

    log_success "Keys generated successfully"
    log_info "  Proving key: $zkey_file"
    log_info "  Verification key: $vkey_file"
}

# Create Python wrapper
create_python_wrapper() {
    log_step "Creating Python wrapper..."

    local wrapper_file="$SUM64_DIR/circuit_loader.py"

    cat > "$wrapper_file" << 'EOF'
"""
Circuit Loader for sum64 ZK Circuit

This module provides Python utilities to load and use the compiled
sum64 circuit artifacts.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Sum64Circuit:
    """Loader for the sum64 zero-knowledge circuit."""

    def __init__(self, build_dir: Optional[str] = None):
        """
        Initialize the circuit loader.

        Args:
            build_dir: Path to build directory. If None, uses default location.
        """
        if build_dir is None:
            self.build_dir = Path(__file__).parent / "build"
        else:
            self.build_dir = Path(build_dir)

        # Check that build directory exists
        if not self.build_dir.exists():
            raise FileNotFoundError(
                f"Build directory not found: {self.build_dir}\n"
                "Please run scripts/build_circuits.sh first."
            )

        # Define artifact paths
        self.r1cs_path = self.build_dir / "sum64.r1cs"
        self.wasm_path = self.build_dir / "sum64_js" / "sum64.wasm"
        self.zkey_path = self.build_dir / "sum64_final.zkey"
        self.vkey_path = self.build_dir / "verification_key.json"

        # Load verification key
        self._vkey = None

    @property
    def verification_key(self) -> Dict[str, Any]:
        """Get the verification key."""
        if self._vkey is None:
            if not self.vkey_path.exists():
                raise FileNotFoundError(
                    f"Verification key not found: {self.vkey_path}"
                )
            with open(self.vkey_path, 'r') as f:
                self._vkey = json.load(f)
        return self._vkey

    def get_paths(self) -> Dict[str, Path]:
        """
        Get paths to all circuit artifacts.

        Returns:
            Dictionary mapping artifact names to paths.
        """
        return {
            "r1cs": self.r1cs_path,
            "wasm": self.wasm_path,
            "zkey": self.zkey_path,
            "vkey": self.vkey_path,
            "witness_calculator": self.build_dir / "sum64_js" / "witness_calculator.js",
            "circuit_wasm": self.build_dir / "sum64_js" / "sum64.wasm",
        }

    def validate_artifacts(self) -> bool:
        """
        Validate that all required artifacts exist.

        Returns:
            True if all artifacts exist, False otherwise.
        """
        paths = self.get_paths()
        missing = []

        for name, path in paths.items():
            if not path.exists():
                missing.append(f"{name}: {path}")

        if missing:
            print("Missing artifacts:")
            for item in missing:
                print(f"  - {item}")
            return False

        return True

    def get_circuit_info(self) -> Dict[str, Any]:
        """
        Get information about the compiled circuit.

        Returns:
            Dictionary with circuit information.
        """
        info = {
            "name": "sum64",
            "description": "Proves knowledge of two numbers that sum to a public value",
            "build_dir": str(self.build_dir),
            "artifacts_valid": self.validate_artifacts(),
        }

        # Add file sizes
        if self.r1cs_path.exists():
            info["r1cs_size"] = self.r1cs_path.stat().st_size
        if self.zkey_path.exists():
            info["zkey_size"] = self.zkey_path.stat().st_size

        return info

    def create_witness_input(self, a: int, b: int) -> Dict[str, str]:
        """
        Create witness input for the circuit.

        Args:
            a: First private input
            b: Second private input

        Returns:
            Dictionary with witness input format.
        """
        c = a + b
        return {
            "a": str(a),
            "b": str(b),
            "c": str(c)
        }

    @staticmethod
    def example_usage():
        """Show example usage of the circuit."""
        print("Sum64 Circuit Example Usage:")
        print("-" * 40)
        print()
        print("from genomevault.zk.circuits.sum64.circuit_loader import Sum64Circuit")
        print()
        print("# Load circuit")
        print("circuit = Sum64Circuit()")
        print()
        print("# Check artifacts")
        print("if circuit.validate_artifacts():")
        print("    print('All artifacts present')")
        print()
        print("# Create witness input")
        print("witness = circuit.create_witness_input(a=15, b=27)")
        print("print(f'Witness: {witness}')")
        print()
        print("# Get verification key")
        print("vkey = circuit.verification_key")
        print("print(f'Verification key loaded: {len(vkey)} fields')")
        print()
        print("# Get paths for snarkjs")
        print("paths = circuit.get_paths()")
        print("print(f'WASM path: {paths[\"wasm\"]}')")
        print("print(f'ZKey path: {paths[\"zkey\"]}')")


if __name__ == "__main__":
    # Test the loader
    try:
        circuit = Sum64Circuit()
        info = circuit.get_circuit_info()

        print("Sum64 Circuit Loader Test")
        print("=" * 40)
        print(json.dumps(info, indent=2))

        if info["artifacts_valid"]:
            print("\n✓ All artifacts present and valid")

            # Test witness creation
            witness = circuit.create_witness_input(10, 32)
            print(f"\nTest witness (10 + 32): {witness}")

            # Show verification key info
            vkey = circuit.verification_key
            print(f"\nVerification key fields: {list(vkey.keys())}")
        else:
            print("\n✗ Some artifacts are missing")
            print("  Run scripts/build_circuits.sh to build the circuit")

        print("\n" + "=" * 40)
        print("\nExample usage:\n")
        Sum64Circuit.example_usage()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
EOF

    log_success "Python wrapper created: $wrapper_file"
}

# Create test script
create_test_script() {
    log_step "Creating test script..."

    local test_file="$SUM64_DIR/test_circuit.js"

    cat > "$test_file" << 'EOF'
#!/usr/bin/env node

/**
 * Test script for sum64 circuit
 *
 * This script tests the complete ZK proof generation and verification flow.
 */

const snarkjs = require("snarkjs");
const fs = require("fs");
const path = require("path");

const BUILD_DIR = path.join(__dirname, "build");

async function testCircuit() {
    console.log("Testing sum64 circuit...\n");

    try {
        // Test inputs
        const input = {
            a: "15",
            b: "27"
        };

        console.log("Input values:");
        console.log(`  a = ${input.a}`);
        console.log(`  b = ${input.b}`);
        console.log(`  Expected c = ${parseInt(input.a) + parseInt(input.b)}\n`);

        // Generate witness
        console.log("Generating witness...");
        const wasmPath = path.join(BUILD_DIR, "sum64_js", "sum64.wasm");
        const wtnsPath = path.join(BUILD_DIR, "witness.wtns");

        await snarkjs.wtns.calculate(
            {type: "file", data: wasmPath},
            input,
            {type: "file", data: wtnsPath}
        );
        console.log("✓ Witness generated\n");

        // Generate proof
        console.log("Generating proof...");
        const zkeyPath = path.join(BUILD_DIR, "sum64_final.zkey");

        const { proof, publicSignals } = await snarkjs.groth16.prove(
            zkeyPath,
            wtnsPath
        );

        console.log("✓ Proof generated");
        console.log(`  Public signal (c): ${publicSignals[0]}\n`);

        // Verify proof
        console.log("Verifying proof...");
        const vKeyPath = path.join(BUILD_DIR, "verification_key.json");
        const vKey = JSON.parse(fs.readFileSync(vKeyPath, "utf8"));

        const res = await snarkjs.groth16.verify(vKey, publicSignals, proof);

        if (res === true) {
            console.log("✓ Proof verified successfully!");
            console.log("\nThe prover knows two numbers that sum to", publicSignals[0]);
            return true;
        } else {
            console.log("✗ Proof verification failed");
            return false;
        }

    } catch (error) {
        console.error("Error:", error.message);
        return false;
    }
}

// Run test if called directly
if (require.main === module) {
    testCircuit().then(success => {
        process.exit(success ? 0 : 1);
    });
}

module.exports = { testCircuit };
EOF

    chmod +x "$test_file"
    log_success "Test script created: $test_file"
}

# Install npm dependencies if needed
install_npm_deps() {
    log_step "Checking npm dependencies..."

    local package_json="$SUM64_DIR/package.json"

    # Create package.json if it doesn't exist
    if [[ ! -f "$package_json" ]]; then
        cat > "$package_json" << 'EOF'
{
  "name": "sum64-circuit",
  "version": "1.0.0",
  "description": "Sum64 ZK circuit for GenomeVault",
  "scripts": {
    "test": "node test_circuit.js"
  },
  "dependencies": {
    "snarkjs": "^0.7.0"
  }
}
EOF
        log_info "Created package.json"
    fi

    # Install dependencies if node_modules doesn't exist
    if [[ ! -d "$SUM64_DIR/node_modules" ]]; then
        log_info "Installing npm dependencies..."
        cd "$SUM64_DIR"
        npm install --silent
        cd - > /dev/null
        log_success "npm dependencies installed"
    else
        log_info "npm dependencies already installed"
    fi
}

# Run tests
run_tests() {
    log_step "Running circuit tests..."

    # Test Python wrapper
    log_info "Testing Python wrapper..."
    cd "$SUM64_DIR"
    if python3 circuit_loader.py > /dev/null 2>&1; then
        log_success "Python wrapper test passed"
    else
        log_warning "Python wrapper test failed (non-critical)"
    fi
    cd - > /dev/null

    # Test with snarkjs
    if [[ -f "$SUM64_DIR/test_circuit.js" ]]; then
        log_info "Testing circuit with snarkjs..."
        cd "$SUM64_DIR"
        if node test_circuit.js; then
            log_success "Circuit test passed"
        else
            log_warning "Circuit test failed"
        fi
        cd - > /dev/null
    fi
}

# Print summary
print_summary() {
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}Build Complete!${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    echo "Circuit artifacts generated in:"
    echo "  $BUILD_DIR"
    echo
    echo "Key files:"
    echo "  • R1CS:            $(basename $BUILD_DIR)/sum64.r1cs"
    echo "  • WASM:            $(basename $BUILD_DIR)/sum64_js/sum64.wasm"
    echo "  • Proving key:     $(basename $BUILD_DIR)/sum64_final.zkey"
    echo "  • Verification key: $(basename $BUILD_DIR)/verification_key.json"
    echo
    echo "Python wrapper:"
    echo "  $SUM64_DIR/circuit_loader.py"
    echo
    echo "To use in Python:"
    echo "  from genomevault.zk.circuits.sum64.circuit_loader import Sum64Circuit"
    echo "  circuit = Sum64Circuit()"
    echo
    echo "To test the circuit:"
    echo "  cd $SUM64_DIR && npm test"
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Main execution
main() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${CYAN}Zero-Knowledge Circuit Build Script${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo

    # Parse arguments
    parse_args "$@"

    # Check dependencies
    check_dependencies

    # Setup directories
    setup_directories

    # Create circuit if needed
    create_sum64_circuit

    # Compile circuit
    compile_circuit

    # Setup Powers of Tau
    setup_powers_of_tau

    # Generate keys
    generate_keys

    # Create Python wrapper
    create_python_wrapper

    # Create test script
    create_test_script

    # Install npm dependencies
    install_npm_deps

    # Run tests
    if [[ "$VERBOSE" == true ]]; then
        run_tests
    fi

    # Print summary
    print_summary
}

# Run main function
main "$@"
