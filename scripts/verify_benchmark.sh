#!/bin/bash
# Verify benchmark signatures using public key
# Usage: ./verify_benchmark.sh <results_directory>

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <results_directory>"
    echo "Example: $0 results/abc123/2025-08-24T12-00-00"
    exit 1
fi

RESULTS_DIR="$1"

# Check directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}Error: Directory $RESULTS_DIR not found${NC}"
    exit 1
fi

# Check required files
REQUIRED_FILES=(
    "results.json"
    "signature.sig"
    "public_key.pem"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$RESULTS_DIR/$file" ]; then
        echo -e "${RED}Error: Missing required file: $file${NC}"
        exit 1
    fi
done

echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     GenomeVault Benchmark Verification      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# Method 1: Using OpenSSL (standard tool)
if command -v openssl &> /dev/null; then
    echo -e "${YELLOW}Using OpenSSL for verification...${NC}"
    
    # Create hash of results file
    openssl dgst -sha256 -binary "$RESULTS_DIR/results.json" > /tmp/results.hash
    
    # Verify signature
    if openssl dgst -sha256 \
        -verify "$RESULTS_DIR/public_key.pem" \
        -signature "$RESULTS_DIR/signature.sig" \
        "$RESULTS_DIR/results.json" &> /dev/null; then
        
        echo -e "${GREEN}✅ Signature verified successfully!${NC}"
        VERIFIED=true
    else
        echo -e "${RED}❌ Signature verification failed!${NC}"
        VERIFIED=false
    fi
    
# Method 2: Using Python cryptography
elif command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Using Python cryptography for verification...${NC}"
    
    python3 - <<EOF
import json
import sys
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

results_dir = Path("$RESULTS_DIR")

# Load files
with open(results_dir / "results.json", 'r') as f:
    results = json.load(f)

with open(results_dir / "signature.sig", 'rb') as f:
    signature = f.read()

with open(results_dir / "public_key.pem", 'rb') as f:
    public_key = serialization.load_pem_public_key(
        f.read(),
        backend=default_backend()
    )

# Serialize results deterministically
results_json = json.dumps(results, sort_keys=True, default=str)
results_bytes = results_json.encode('utf-8')

# Verify signature
try:
    public_key.verify(
        signature,
        results_bytes,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    print("✅ Signature verified successfully!")
    sys.exit(0)
except InvalidSignature:
    print("❌ Signature verification failed!")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        VERIFIED=true
    else
        VERIFIED=false
    fi
    
else
    echo -e "${RED}Error: Neither OpenSSL nor Python3 found${NC}"
    echo "Please install one of these tools to verify signatures"
    exit 1
fi

# Extract and display metadata if verified
if [ "$VERIFIED" = true ]; then
    echo ""
    echo -e "${GREEN}📊 Benchmark Metadata:${NC}"
    
    # Extract key fields using jq if available, otherwise use python
    if command -v jq &> /dev/null; then
        echo -e "  Git SHA: $(jq -r '.git_sha' "$RESULTS_DIR/results.json" | head -c 8)"
        echo -e "  Timestamp: $(jq -r '.timestamp' "$RESULTS_DIR/results.json")"
        echo -e "  Seed: $(jq -r '.seed' "$RESULTS_DIR/results.json")"
        echo -e "  Benchmarks: $(jq -r '.benchmarks | length' "$RESULTS_DIR/results.json")"
    else
        python3 - <<EOF
import json
with open("$RESULTS_DIR/results.json", 'r') as f:
    data = json.load(f)
    print(f"  Git SHA: {data.get('git_sha', 'unknown')[:8]}")
    print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
    print(f"  Seed: {data.get('seed', 'unknown')}")
    print(f"  Benchmarks: {len(data.get('benchmarks', []))}")
EOF
    fi
    
    echo ""
    echo -e "${GREEN}✅ Results are authentic and have not been tampered with${NC}"
else
    echo ""
    echo -e "${RED}⚠️  WARNING: Results may have been modified${NC}"
    echo "The signature does not match the results file."
    echo "This could mean:"
    echo "  1. The results have been modified after signing"
    echo "  2. The wrong public key was used"
    echo "  3. The signature file is corrupted"
fi

exit $([ "$VERIFIED" = true ] && echo 0 || echo 1)