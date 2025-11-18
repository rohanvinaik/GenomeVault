#!/bin/bash
#
# Recompress guide FASTAs with bgzip for pysam compatibility
#
# This script:
# 1. Decompresses gzip-compressed FASTAs
# 2. Recompresses with bgzip (block-gzip)
# 3. Creates .fai index files with samtools
#
# Memory-safe: Processes one file at a time
# Time estimate: ~3-5 minutes per guide (11 guides = 30-55 minutes)

set -e

GUIDE_DIR="/Volumes/1TBStorage/guide_strands"
LOG_FILE="bgzip_recompression_$(date +%Y%m%d_%H%M%S).log"

echo "=============================================="
echo "Guide FASTA Recompression (gzip → bgzip)"
echo "=============================================="
echo ""
echo "Guide directory: $GUIDE_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Check if bgzip is available
if ! command -v bgzip &> /dev/null; then
    echo "ERROR: bgzip not found. Installing via conda..."
    conda install -y -c bioconda htslib
fi

# Check if samtools is available
if ! command -v samtools &> /dev/null; then
    echo "ERROR: samtools not found. Installing via conda..."
    conda install -y -c bioconda samtools
fi

echo "✓ Tools available: bgzip, samtools"
echo ""

# Process each guide FASTA
for i in {1..11}; do
    FASTA_GZ="$GUIDE_DIR/ref${i}.fa.gz"
    FASTA_UNCOMPRESSED="$GUIDE_DIR/ref${i}.fa"
    FASTA_INDEX="$GUIDE_DIR/ref${i}.fa.gz.fai"

    echo "================================================"
    echo "Processing ref${i}.fa.gz"
    echo "================================================"

    # Check if file exists
    if [ ! -f "$FASTA_GZ" ]; then
        echo "⚠️  File not found: $FASTA_GZ (skipping)"
        echo ""
        continue
    fi

    # Check current size
    SIZE_BEFORE=$(du -h "$FASTA_GZ" | awk '{print $1}')
    echo "Current size: $SIZE_BEFORE (gzip-compressed)"

    # Step 1: Decompress gzip
    echo "Step 1/3: Decompressing with gunzip..."
    START_TIME=$(date +%s)

    gunzip "$FASTA_GZ"

    if [ ! -f "$FASTA_UNCOMPRESSED" ]; then
        echo "ERROR: Decompression failed for ref${i}"
        exit 1
    fi

    SIZE_UNCOMPRESSED=$(du -h "$FASTA_UNCOMPRESSED" | awk '{print $1}')
    DECOMPRESS_TIME=$(($(date +%s) - START_TIME))
    echo "  ✓ Decompressed: $SIZE_UNCOMPRESSED (took ${DECOMPRESS_TIME}s)"

    # Step 2: Recompress with bgzip
    echo "Step 2/3: Recompressing with bgzip..."
    START_TIME=$(date +%s)

    # Use 8 threads for faster compression
    bgzip -@ 8 "$FASTA_UNCOMPRESSED"

    if [ ! -f "$FASTA_GZ" ]; then
        echo "ERROR: bgzip compression failed for ref${i}"
        exit 1
    fi

    SIZE_AFTER=$(du -h "$FASTA_GZ" | awk '{print $1}')
    COMPRESS_TIME=$(($(date +%s) - START_TIME))
    echo "  ✓ Recompressed: $SIZE_AFTER (took ${COMPRESS_TIME}s)"

    # Step 3: Create FASTA index
    echo "Step 3/3: Creating FASTA index..."
    START_TIME=$(date +%s)

    samtools faidx "$FASTA_GZ"

    if [ ! -f "$FASTA_INDEX" ]; then
        echo "ERROR: Index creation failed for ref${i}"
        exit 1
    fi

    INDEX_SIZE=$(du -h "$FASTA_INDEX" | awk '{print $1}')
    INDEX_TIME=$(($(date +%s) - START_TIME))
    echo "  ✓ Index created: $INDEX_SIZE (took ${INDEX_TIME}s)"

    echo ""
    echo "Summary for ref${i}:"
    echo "  Before: $SIZE_BEFORE (gzip)"
    echo "  After:  $SIZE_AFTER (bgzip)"
    echo "  Index:  $INDEX_SIZE (.fai)"
    echo ""

done | tee -a "$LOG_FILE"

echo "=============================================="
echo "RECOMPRESSION COMPLETE"
echo "=============================================="
echo ""
echo "All guide FASTAs have been recompressed with bgzip."
echo "You can now use pysam to access these files."
echo ""
echo "Log saved to: $LOG_FILE"
