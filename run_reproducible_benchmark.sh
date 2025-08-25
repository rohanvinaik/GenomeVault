#!/bin/bash
# One-command reproducible benchmark runner for GenomeVault
# Usage: ./run_reproducible_benchmark.sh [docker|local]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MODE=${1:-docker}
SEED=${GENOMEVAULT_SEED:-42}
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")

echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     GenomeVault Reproducible Benchmark Runner           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Mode:${NC} $MODE"
echo -e "${YELLOW}Seed:${NC} $SEED"
echo -e "${YELLOW}Time:${NC} $TIMESTAMP"
echo ""

# Get git SHA
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
echo -e "${YELLOW}Git SHA:${NC} $GIT_SHA"

if [ "$MODE" == "docker" ]; then
    echo -e "\n${GREEN}🐳 Running in Docker (fully reproducible)${NC}\n"
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        echo "Please install Docker from https://docker.com"
        exit 1
    fi
    
    # Build Docker image
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -f Dockerfile.benchmark -t genomevault-benchmark:latest . || {
        echo -e "${RED}Failed to build Docker image${NC}"
        exit 1
    }
    
    # Create results directory
    mkdir -p results
    
    # Run benchmark in Docker
    echo -e "${YELLOW}Running benchmarks...${NC}"
    docker run --rm \
        -v "$(pwd)/results:/genomevault/results" \
        -e GENOMEVAULT_SEED=$SEED \
        --name genomevault-benchmark-$TIMESTAMP \
        genomevault-benchmark:latest
    
    # Find the latest results
    LATEST_RESULTS=$(find results -name "verify.py" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_RESULTS" ]; then
        RESULTS_DIR=$(dirname "$LATEST_RESULTS")
        echo -e "\n${GREEN}✅ Benchmark complete!${NC}"
        echo -e "${YELLOW}Results:${NC} $RESULTS_DIR"
        
        # Verify results
        echo -e "\n${YELLOW}Verifying signature...${NC}"
        python "$RESULTS_DIR/verify.py" "$RESULTS_DIR"
    fi
    
elif [ "$MODE" == "local" ]; then
    echo -e "\n${GREEN}💻 Running locally${NC}\n"
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${YELLOW}Python:${NC} $PYTHON_VERSION"
    
    # Install dependencies if needed
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv venv
        source venv/bin/activate
        pip install -q --upgrade pip
        pip install -q -e ".[dev]"
        pip install -q cryptography
    else
        source venv/bin/activate
    fi
    
    # Run benchmark
    echo -e "${YELLOW}Running benchmarks...${NC}"
    GENOMEVAULT_SEED=$SEED python reproducible_harness.py
    
    # Find the latest results
    LATEST_RESULTS=$(find results -name "verify.py" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_RESULTS" ]; then
        RESULTS_DIR=$(dirname "$LATEST_RESULTS")
        echo -e "\n${GREEN}✅ Benchmark complete!${NC}"
        echo -e "${YELLOW}Results:${NC} $RESULTS_DIR"
        
        # Verify results
        echo -e "\n${YELLOW}Verifying signature...${NC}"
        python "$RESULTS_DIR/verify.py" "$RESULTS_DIR"
    fi
    
else
    echo -e "${RED}Invalid mode: $MODE${NC}"
    echo "Usage: $0 [docker|local]"
    exit 1
fi

# Generate summary report
if [ -n "${RESULTS_DIR:-}" ]; then
    echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
    echo -e "${GREEN}  REPRODUCIBILITY REPORT${NC}"
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    
    # Check if results are identical to previous runs
    if [ -f "results/.last_run_hash" ]; then
        CURRENT_HASH=$(sha256sum "$RESULTS_DIR/results.json" | cut -d' ' -f1)
        LAST_HASH=$(cat results/.last_run_hash)
        
        if [ "$CURRENT_HASH" == "$LAST_HASH" ]; then
            echo -e "${GREEN}✅ Results are IDENTICAL to previous run${NC}"
            echo -e "   (Deterministic execution confirmed)"
        else
            echo -e "${YELLOW}⚠️  Results differ from previous run${NC}"
            echo -e "   Current: $CURRENT_HASH"
            echo -e "   Previous: $LAST_HASH"
        fi
    else
        # Save hash for next comparison
        sha256sum "$RESULTS_DIR/results.json" | cut -d' ' -f1 > results/.last_run_hash
        echo -e "${YELLOW}📝 First run - hash saved for comparison${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}📁 Output files:${NC}"
    echo "   • results.json    - Benchmark results"
    echo "   • environment.json - Environment snapshot"
    echo "   • sbom.json       - Software Bill of Materials"
    echo "   • signature.sig   - Digital signature"
    echo "   • verify.py       - Verification script"
    echo ""
    echo -e "${YELLOW}🔐 To verify independently:${NC}"
    echo "   python $RESULTS_DIR/verify.py $RESULTS_DIR"
fi

echo -e "\n${GREEN}Done!${NC}"