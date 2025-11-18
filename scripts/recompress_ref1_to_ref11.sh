#!/bin/bash
#
# Recompress ALL reference FASTAs (ref1-ref11) with bgzip
#
# Memory-safe: One file at a time (~3 GB peak)
# Total time: ~6-8 minutes for 11 files

set -e

echo "============================================================"
echo "Recompressing ref1-ref11 FASTAs (gzip → bgzip)"
echo "============================================================"
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

echo "✓ Tools ready"
echo ""

# Function to recompress a single FASTA
recompress_fasta() {
    local FASTA_PATH="$1"
    local FILE_NAME=$(basename "$FASTA_PATH")

    echo "========================================"
    echo "Processing: $FILE_NAME"
    echo "========================================"

    # Resolve symlinks
    if [ -L "$FASTA_PATH" ]; then
        REAL_PATH=$(readlink -f "$FASTA_PATH" || readlink "$FASTA_PATH")
        echo "Symlink → $REAL_PATH"
        FASTA_PATH="$REAL_PATH"
    fi

    if [ ! -f "$FASTA_PATH" ]; then
        echo "⚠️  File not found: $FASTA_PATH"
        return 1
    fi

    # Check if already bgzip-compressed
    if python3 -c "import pysam; pysam.FastaFile('$FASTA_PATH')" 2>/dev/null; then
        echo "✓ Already bgzip-compressed (skipping)"
        return 0
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
    bgzip -@ 8 "$UNCOMPRESSED"
    ELAPSED=$(($(date +%s) - START))
    SIZE_AFTER=$(du -h "$FASTA_PATH" | awk '{print $1}')
    echo "  ✓ Recompressed: $SIZE_AFTER (${ELAPSED}s)"

    echo "Step 3/3: Creating index..."
    START=$(date +%s)
    samtools faidx "$FASTA_PATH"
    ELAPSED=$(($(date +%s) - START))
    INDEX_SIZE=$(du -h "$FASTA_PATH.fai" | awk '{print $1}')
    echo "  ✓ Index: $INDEX_SIZE (${ELAPSED}s)"

    echo "Done: $SIZE_BEFORE → $SIZE_AFTER + $INDEX_SIZE index"
    echo ""
}

# Process all 11 references
TOTAL=11
COMPLETED=0

for i in {1..11}; do
    # Check both locations
    if [ -f "/Volumes/1TBStorage/guide_strands/ref${i}.fa.gz" ]; then
        FASTA="/Volumes/1TBStorage/guide_strands/ref${i}.fa.gz"
    elif [ -L "/Volumes/1TBStorage/guide_strands/ref${i}.fa.gz" ]; then
        FASTA="/Volumes/1TBStorage/guide_strands/ref${i}.fa.gz"
    else
        echo "⚠️  ref${i}.fa.gz not found (skipping)"
        continue
    fi

    if recompress_fasta "$FASTA"; then
        COMPLETED=$((COMPLETED + 1))
        echo "Progress: $COMPLETED/$TOTAL complete"
        echo ""
    fi
done

echo "============================================================"
echo "RECOMPRESSION COMPLETE"
echo "============================================================"
echo ""
echo "Completed: $COMPLETED/$TOTAL reference FASTAs"
echo ""
echo "All files are now bgzip-compressed and indexed!"
echo "Ready for full nucleotide-resolution HDV encoding."
echo ""
