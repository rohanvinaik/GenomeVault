#!/bin/bash
#
# Compress FASTQ files to save disk space
# Uses pigz (parallel gzip) if available, otherwise regular gzip
#
# Usage: ./scripts/compress_fastq.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}GenomeVault FASTQ Compression${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

FASTQ_DIR="data/downloaded/fastq"

# Check for pigz (parallel gzip)
if command -v pigz &> /dev/null; then
    COMPRESSOR="pigz -p 4"
    echo -e "${GREEN}✓${NC} Using pigz (parallel compression, 4 threads)"
else
    COMPRESSOR="gzip"
    echo -e "${YELLOW}⚠${NC} Using gzip (single-threaded). Install pigz for faster compression:"
    echo -e "   ${YELLOW}conda install -c conda-forge pigz${NC}"
fi
echo ""

# Find all uncompressed FASTQ files
FASTQ_FILES=$(find "$FASTQ_DIR" -name "*.fastq" -type f 2>/dev/null || true)

if [ -z "$FASTQ_FILES" ]; then
    echo -e "${GREEN}✓${NC} No uncompressed FASTQ files found. All files are already compressed!"
    exit 0
fi

COUNT=$(echo "$FASTQ_FILES" | wc -l | tr -d ' ')
echo -e "${YELLOW}Found $COUNT uncompressed FASTQ files${NC}"
echo ""

# Compress each file
INDEX=1
for FILE in $FASTQ_FILES; do
    BASENAME=$(basename "$FILE")
    SIZE_BEFORE=$(stat -f%z "$FILE" 2>/dev/null || stat -c%s "$FILE" 2>/dev/null)
    SIZE_BEFORE_GB=$(echo "scale=2; $SIZE_BEFORE / 1073741824" | bc)

    echo -e "${BLUE}[$INDEX/$COUNT]${NC} Compressing: $BASENAME (${SIZE_BEFORE_GB}GB)"

    # Compress
    $COMPRESSOR "$FILE"

    # Check result
    if [ -f "${FILE}.gz" ]; then
        SIZE_AFTER=$(stat -f%z "${FILE}.gz" 2>/dev/null || stat -c%s "${FILE}.gz" 2>/dev/null)
        SIZE_AFTER_GB=$(echo "scale=2; $SIZE_AFTER / 1073741824" | bc)
        RATIO=$(echo "scale=1; ($SIZE_BEFORE - $SIZE_AFTER) * 100 / $SIZE_BEFORE" | bc)

        echo -e "  ${GREEN}✓${NC} Compressed: ${SIZE_BEFORE_GB}GB → ${SIZE_AFTER_GB}GB (${RATIO}% reduction)"
    else
        echo -e "  ${YELLOW}⚠${NC} Compression failed"
    fi

    INDEX=$((INDEX + 1))
    echo ""
done

# Summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Compression complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Show total space savings
TOTAL_COMPRESSED=$(find "$FASTQ_DIR" -name "*.fastq.gz" -type f -exec stat -f%z {} + 2>/dev/null | awk '{s+=$1} END {print s}' || echo 0)
TOTAL_GB=$(echo "scale=2; $TOTAL_COMPRESSED / 1073741824" | bc)

echo -e "Total compressed FASTQ size: ${GREEN}${TOTAL_GB}GB${NC}"
echo -e "Typical compression ratio: ${GREEN}~70-80%${NC} space saved"
echo ""
