#!/bin/bash
# Build Pre-Indexed Minimap2 References
# Optimized for short-read alignment (paired-end sequencing)
#
# Usage:
#   ./scripts/build_minimap2_index.sh <reference.fa.gz> [output_name]
#
# Benefits:
#   - Save 2-5 min per alignment run
#   - Consistent indexing parameters
#   - ARM64 NEON optimization (automatic on M2 Pro)

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <reference.fa.gz> [output_name]"
    echo ""
    echo "Example:"
    echo "  $0 data/reference_genomes/hg38.fa.gz hg38_sr"
    echo ""
    echo "Creates: data/reference_genomes/hg38_sr.mmi"
    exit 1
fi

REFERENCE="$1"
OUTPUT_NAME="${2:-$(basename "$REFERENCE" .fa.gz)_sr}"
OUTPUT_DIR="$(dirname "$REFERENCE")"
OUTPUT_MMI="${OUTPUT_DIR}/${OUTPUT_NAME}.mmi"

echo "================================================================================"
echo "MINIMAP2 INDEX BUILDER (SHORT-READ OPTIMIZED)"
echo "================================================================================"
echo "Input:  $REFERENCE"
echo "Output: $OUTPUT_MMI"
echo "Preset: sr (short-read paired-end)"
echo "================================================================================"
echo ""

if [ ! -f "$REFERENCE" ]; then
    echo "ERROR: Reference file not found: $REFERENCE"
    exit 1
fi

if [ -f "$OUTPUT_MMI" ]; then
    echo "⚠️  Index already exists: $OUTPUT_MMI"
    echo ""
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo "Building minimap2 index..."
echo "Command: minimap2 -x sr -d $OUTPUT_MMI $REFERENCE"
echo ""

start_time=$(date +%s)

minimap2 -x sr -d "$OUTPUT_MMI" "$REFERENCE"

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "================================================================================"
echo "✓ INDEX BUILD COMPLETE"
echo "================================================================================"
echo "Output: $OUTPUT_MMI"
echo "Size:   $(du -h "$OUTPUT_MMI" | cut -f1)"
echo "Time:   ${duration}s ($(($duration / 60))m $(($duration % 60))s)"
echo "================================================================================"
echo ""
echo "Usage in pipeline:"
echo "  minimap2 -ax sr -t 10 -K 250M -2 $OUTPUT_MMI reads_R1.fq.gz reads_R2.fq.gz"
echo ""
echo "Speedup: ~2-5 min saved per alignment (no re-indexing)"
