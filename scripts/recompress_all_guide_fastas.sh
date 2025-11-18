#!/bin/bash
#
# Recompress ALL 11 guide FASTAs with bgzip (memory-safe, one at a time)
#
# Handles both locations:
# 1. /Volumes/1TBStorage/guide_strands/ (ref10.fa.gz, ref11.fa.gz)
# 2. benchmark_results/k3_whole_genome_benchmark/guide_sequences/ (guide1-3.fa.gz)
#
# Memory usage: ~3 GB per FASTA during decompression (processes one at a time)
# Total time: ~30-55 minutes for all 11 guides

set -e

echo "============================================================"
echo "Complete Guide FASTA Recompression (gzip → bgzip)"
echo "============================================================"
echo ""
echo "This will recompress ALL 11 guide FASTAs for full nucleotide"
echo "resolution encoding."
echo ""
echo "Estimated time: 3-5 minutes per guide (30-55 min total)"
echo "Memory usage: ~3 GB peak (one file at a time)"
echo ""

# Check tools
if ! command -v bgzip &> /dev/null; then
    echo "Installing bgzip..."
    conda install -y -c bioconda htslib
fi

if ! command -v samtools &> /dev/null; then
    echo "Installing samtools..."
    conda install -y -c bioconda samtools
fi

echo "✓ Tools ready: bgzip, samtools"
echo ""

# Function to recompress a single FASTA
recompress_fasta() {
    local FASTA_PATH="$1"
    local FILE_NAME=$(basename "$FASTA_PATH")

    echo "========================================"
    echo "Processing: $FILE_NAME"
    echo "========================================"

    if [ ! -f "$FASTA_PATH" ]; then
        echo "⚠️  File not found: $FASTA_PATH"
        return 1
    fi

    # Check if already bgzip-compressed
    if file "$FASTA_PATH" | grep -q "gzip compressed"; then
        # Additional check: try to open with pysam (will fail if not bgzip)
        if python3 -c "import pysam; pysam.FastaFile('$FASTA_PATH')" 2>/dev/null; then
            echo "✓ Already bgzip-compressed (skipping)"
            return 0
        fi
    fi

    local DIR=$(dirname "$FASTA_PATH")
    local BASE=$(basename "$FASTA_PATH" .gz)
    local UNCOMPRESSED="$DIR/$BASE"

    SIZE_BEFORE=$(du -h "$FASTA_PATH" | awk '{print $1}')
    echo "Current: $SIZE_BEFORE (gzip)"

    echo "Step 1/3: Decompressing..."
    START=$(date +%s)
    gunzip "$FASTA_PATH"
    ELAPSED=$(($(date +%s) - START))
    SIZE_UNCOMP=$(du -h "$UNCOMPRESSED" | awk '{print $1}')
    echo "  ✓ Decompressed: $SIZE_UNCOMP (${ELAPSED}s)"

    echo "Step 2/3: Recompressing with bgzip..."
    START=$(date +%s)
    bgzip -@ 8 "$UNCOMPRESSED"  # 8 threads
    ELAPSED=$(($(date +%s) - START))
    SIZE_AFTER=$(du -h "$FASTA_PATH" | awk '{print $1}')
    echo "  ✓ Recompressed: $SIZE_AFTER (${ELAPSED}s)"

    echo "Step 3/3: Creating index..."
    START=$(date +%s)
    samtools faidx "$FASTA_PATH"
    ELAPSED=$(($(date +%s) - START))
    INDEX_SIZE=$(du -h "$FASTA_PATH.fai" | awk '{print $1}')
    echo "  ✓ Index: $INDEX_SIZE (${ELAPSED}s)"

    echo "Summary: $SIZE_BEFORE → $SIZE_AFTER + $INDEX_SIZE index"
    echo ""
}

# Track overall progress
TOTAL_FILES=11
COMPLETED=0

echo "============================================================"
echo "PHASE 1: K=3 Guide Sequences"
echo "============================================================"

GUIDE_DIR="benchmark_results/k3_whole_genome_benchmark/guide_sequences"

for i in 1 2 3; do
    FASTA="$GUIDE_DIR/guide${i}.fa.gz"
    if recompress_fasta "$FASTA"; then
        COMPLETED=$((COMPLETED + 1))
        echo "Progress: $COMPLETED/$TOTAL_FILES complete"
        echo ""
    fi
done

echo "============================================================"
echo "PHASE 2: 1TBStorage Guide Strands"
echo "============================================================"

GUIDE_DIR="/Volumes/1TBStorage/guide_strands"

for i in 10 11; do
    FASTA="$GUIDE_DIR/ref${i}.fa.gz"
    if recompress_fasta "$FASTA"; then
        COMPLETED=$((COMPLETED + 1))
        echo "Progress: $COMPLETED/$TOTAL_FILES complete"
        echo ""
    fi
done

# Also handle ref4-ref9 if they exist as actual files (not symlinks)
for i in 4 5 6 7 8 9; do
    FASTA="$GUIDE_DIR/ref${i}.fa.gz"
    if [ -f "$FASTA" ] && [ ! -L "$FASTA" ]; then
        if recompress_fasta "$FASTA"; then
            COMPLETED=$((COMPLETED + 1))
            echo "Progress: $COMPLETED/$TOTAL_FILES complete"
            echo ""
        fi
    fi
done

echo "============================================================"
echo "RECOMPRESSION COMPLETE"
echo "============================================================"
echo ""
echo "Recompressed: $COMPLETED/$TOTAL_FILES guide FASTAs"
echo ""
echo "All guide FASTAs are now bgzip-compressed and indexed."
echo "You can now run full nucleotide-resolution HDV encoding!"
echo ""
