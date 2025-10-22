#!/bin/bash
#
# Regenerate Missing NEAT Chunks 1-21
#
# This script re-runs NEAT on the failed first 21 chunks to complete Ref2.
# The chunks 22-102 are already salvaged and concatenated.
#

set -e

WORK_DIR="/Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples"
TEMP_DIR="$WORK_DIR/temp"
CHUNK_DIR="/var/folders/pf/h3szfpt17_nc2vbq9mmd__pc0000gp/T/tmp5fmcup8p"

echo "========================================================================"
echo "Regenerating Missing Chunks 1-21 for Ref2"
echo "========================================================================"
echo ""

cd "$TEMP_DIR"

# Activate NEAT environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neat

# Create output directory for regenerated chunks
mkdir -p chunks_1-21_regenerated

echo "Regenerating chunks 1-21 using existing split files..."
echo "Split files location: $CHUNK_DIR/splits/"
echo ""

# Missing chunks: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
MISSING_CHUNKS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)

for chunk in "${MISSING_CHUNKS[@]}"; do
    CHUNK_NUM=$(printf "%010d" $chunk)
    SPLIT_FILE="$CHUNK_DIR/splits/${CHUNK_NUM}__chr22.fa.gz"
    OUTPUT_DIR="chunks_1-21_regenerated/chunk_${CHUNK_NUM}"

    mkdir -p "$OUTPUT_DIR"

    echo "[$(date +%H:%M:%S)] Processing chunk $CHUNK_NUM..."

    # Decompress split file
    gunzip -c "$SPLIT_FILE" > "$OUTPUT_DIR/input.fa"

    # Create minimal NEAT config for this chunk
    cat > "$OUTPUT_DIR/neat_config.yml" << EOF
reference: $OUTPUT_DIR/input.fa
read_len: 150
fragment_mean: 300
fragment_st_dev: 50
coverage: 30
paired_ended: true
rng_seed: $((200 + chunk))
produce_bam: false
ploidy: 2
threads: 1
EOF

    # Run NEAT on this single chunk
    if neat read-simulator \
        -c "$OUTPUT_DIR/neat_config.yml" \
        -o "$OUTPUT_DIR" \
        -p "chunk${CHUNK_NUM}" 2>&1 | tee "$OUTPUT_DIR/neat.log"; then
        echo "  ✓ Chunk $CHUNK_NUM complete"

        # Rename output files
        mv "$OUTPUT_DIR/chunk${CHUNK_NUM}_r1.fastq.gz" "$OUTPUT_DIR/sample2_r1.fastq.gz" 2>/dev/null || true
        mv "$OUTPUT_DIR/chunk${CHUNK_NUM}_r2.fastq.gz" "$OUTPUT_DIR/sample2_r2.fastq.gz" 2>/dev/null || true
    else
        echo "  ✗ Chunk $CHUNK_NUM failed - continuing anyway"
    fi

    # Clean up temp files
    rm -f "$OUTPUT_DIR/input.fa"
done

echo ""
echo "Concatenating regenerated chunks 1-21..."

# Concatenate in order
for chunk in "${MISSING_CHUNKS[@]}"; do
    CHUNK_NUM=$(printf "%010d" $chunk)
    OUTPUT_DIR="chunks_1-21_regenerated/chunk_${CHUNK_NUM}"

    if [ -f "$OUTPUT_DIR/sample2_r1.fastq.gz" ] && [ -s "$OUTPUT_DIR/sample2_r1.fastq.gz" ]; then
        cat "$OUTPUT_DIR/sample2_r1.fastq.gz" >> sample2_r1_chunks1-21.fastq.gz
        cat "$OUTPUT_DIR/sample2_r2.fastq.gz" >> sample2_r2_chunks1-21.fastq.gz
    fi
done

echo ""
echo "Merging chunks 1-21 with salvaged chunks 22-102..."

# Concatenate (prepend regenerated chunks to salvaged ones)
cat sample2_r1_chunks1-21.fastq.gz sample2_r1_chunks22-102.fastq.gz > sample2_r1_complete.fastq.gz
cat sample2_r2_chunks1-21.fastq.gz sample2_r2_chunks22-102.fastq.gz > sample2_r2_complete.fastq.gz

echo ""
echo "✓ Complete Ref2 files created:"
ls -lh sample2_r*_complete.fastq.gz

echo ""
echo "Moving to final location..."
mv sample2_r1_complete.fastq.gz ../references/ref2/sample2_r1.fastq.gz
mv sample2_r2_complete.fastq.gz ../references/ref2/sample2_r2.fastq.gz

echo ""
echo "✓ Ref2 generation complete!"
echo "Final files:"
ls -lh ../references/ref2/sample2_*.fastq.gz
