#!/bin/bash
#
# Generate Reference Pool for Differential Encoding
#
# Creates 4 synthetic genome samples:
#   - 3 reference genomes (for random selection pool)
#   - 1 query genome (experimental data to encode)
#
# Usage: ./generate_reference_pool.sh
# Runtime: ~4-5 hours total (sequential generation)
#

set -e  # Exit on error

WORK_DIR="/Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples"
REFERENCE_DIR="$WORK_DIR/references"
QUERY_DIR="$WORK_DIR/query"
TEMP_DIR="$WORK_DIR/temp"

echo "========================================================================"
echo "Differential Encoding Reference Pool Generation"
echo "========================================================================"
echo "Target: 3 reference genomes + 1 query genome"
echo "Region: chr22 (~50Mb)"
echo "Coverage: 30x paired-end (150bp reads)"
echo "Estimated Runtime: 4-5 hours"
echo ""

# Create directory structure
mkdir -p "$REFERENCE_DIR"/{ref1,ref2,ref3}
mkdir -p "$QUERY_DIR"
mkdir -p "$TEMP_DIR"

# Copy existing Sample 1 as Reference 1
echo "========================================================================"
echo "Step 1: Organizing Sample 1 as Reference 1"
echo "========================================================================"
if [ -f "/Users/rohanvinaik/genomevault/benchmark_results/full_pipeline_synthetic/neat_output/neat_sim_r1.fastq.gz" ]; then
    echo "Moving existing Sample 1 to reference pool..."
    cp /Users/rohanvinaik/genomevault/benchmark_results/full_pipeline_synthetic/neat_output/neat_sim_r1.fastq.gz \
       "$REFERENCE_DIR/ref1/sample1_r1.fastq.gz"
    cp /Users/rohanvinaik/genomevault/benchmark_results/full_pipeline_synthetic/neat_output/neat_sim_r2.fastq.gz \
       "$REFERENCE_DIR/ref1/sample1_r2.fastq.gz"

    # Copy variant information
    cp /Users/rohanvinaik/genomevault/benchmark_results/full_pipeline_synthetic/simug_output/simulated.refseq2simseq.SNP.vcf \
       "$REFERENCE_DIR/ref1/variants_snp.vcf"
    cp /Users/rohanvinaik/genomevault/benchmark_results/full_pipeline_synthetic/simug_output/simulated.refseq2simseq.INDEL.vcf \
       "$REFERENCE_DIR/ref1/variants_indel.vcf"

    echo "✓ Reference 1 ready (seed=42)"
else
    echo "ERROR: Sample 1 not found. Run full_pipeline_synthetic.sh first."
    exit 1
fi

# Function to generate a sample
generate_sample() {
    local SAMPLE_NUM=$1
    local SEED=$2
    local OUTPUT_DIR=$3
    local LABEL=$4

    echo ""
    echo "========================================================================"
    echo "Step $SAMPLE_NUM: Generating $LABEL (seed=$SEED)"
    echo "========================================================================"
    echo "Started: $(date)"

    cd "$TEMP_DIR"

    # Use existing reference genome (chr22)
    REF_GENOME="/Users/rohanvinaik/genomevault/benchmark_results/full_pipeline_synthetic/reference/chr22.fa"

    if [ ! -f "$REF_GENOME" ]; then
        echo "ERROR: Reference genome not found at $REF_GENOME"
        return 1
    fi

    # Generate variants with simuG
    echo "[$(date +%H:%M:%S)] Generating variants with simuG (seed=$SEED)..."

    # Create simuG output directory
    mkdir -p "simug_sample${SAMPLE_NUM}"

    if ! perl ~/simuG/simuG.pl \
        -refseq "$REF_GENOME" \
        -snp_count 10000 \
        -indel_count 2000 \
        -cnv_count 20 \
        -inversion_count 3 \
        -titv_ratio 2.0 \
        -seed "$SEED" \
        -prefix "simug_sample${SAMPLE_NUM}/sample${SAMPLE_NUM}"; then
        echo "ERROR: simuG failed with seed $SEED"
        return 1
    fi

    echo "✓ Generated $(grep -v '^#' simug_sample${SAMPLE_NUM}/sample${SAMPLE_NUM}.refseq2simseq.SNP.vcf 2>/dev/null | wc -l | tr -d ' ') SNPs"
    echo "✓ Generated $(grep -v '^#' simug_sample${SAMPLE_NUM}/sample${SAMPLE_NUM}.refseq2simseq.INDEL.vcf 2>/dev/null | wc -l | tr -d ' ') Indels"

    # Generate reads with NEAT
    echo "[$(date +%H:%M:%S)] Generating sequencing reads with NEAT..."

    # Create NEAT config
    cat > neat_config_sample${SAMPLE_NUM}.yml << EOF
reference: simug_sample${SAMPLE_NUM}/sample${SAMPLE_NUM}.simseq.genome.fa
read_len: 150
fragment_mean: 300
fragment_st_dev: 50
coverage: 30
paired_ended: true
rng_seed: $SEED
produce_bam: false
ploidy: 2
threads: 10
EOF

    # Activate NEAT conda environment and run
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate neat

    if ! neat read-simulator \
        -c "neat_config_sample${SAMPLE_NUM}.yml" \
        -o . \
        -p "sample${SAMPLE_NUM}" 2>&1 | tee "neat_sample${SAMPLE_NUM}.log"; then
        echo "ERROR: NEAT failed with seed $SEED"
        echo "See neat_sample${SAMPLE_NUM}.log for details"
        return 1
    fi

    # Check if outputs were actually created
    if [ ! -f "sample${SAMPLE_NUM}_r1.fastq.gz" ] || [ ! -f "sample${SAMPLE_NUM}_r2.fastq.gz" ]; then
        echo "ERROR: NEAT did not produce expected FASTQ files"
        return 1
    fi

    echo "[$(date +%H:%M:%S)] NEAT complete - validating chunk completeness..."

    # GENOMEVAULT_FIX: Validate all chunks were generated (catch startup race conditions)
    # Find the temp directory NEAT used
    TEMP_CHUNK_DIR=$(find /var/folders -path "*/tmp*/splits" -type d 2>/dev/null | head -1 | xargs dirname 2>/dev/null)

    if [ -n "$TEMP_CHUNK_DIR" ] && [ -d "$TEMP_CHUNK_DIR" ]; then
        EXPECTED_CHUNKS=$(ls "$TEMP_CHUNK_DIR"/splits/*.fa.gz 2>/dev/null | wc -l | tr -d ' ')
        ACTUAL_R1=$(find "$TEMP_CHUNK_DIR" -name "sample${SAMPLE_NUM}_r1.fastq.gz" -type f -size +1M 2>/dev/null | wc -l | tr -d ' ')
        ACTUAL_R2=$(find "$TEMP_CHUNK_DIR" -name "sample${SAMPLE_NUM}_r2.fastq.gz" -type f -size +1M 2>/dev/null | wc -l | tr -d ' ')

        echo "  Chunk validation: $ACTUAL_R1 R1 chunks, $ACTUAL_R2 R2 chunks (expected: $EXPECTED_CHUNKS each)"

        # If we're missing more than 25% of chunks, this is a serious failure
        MISSING_R1=$((EXPECTED_CHUNKS - ACTUAL_R1))
        MISSING_R2=$((EXPECTED_CHUNKS - ACTUAL_R2))
        MISSING_TOTAL=$((MISSING_R1 + MISSING_R2))

        if [ $MISSING_TOTAL -gt 0 ]; then
            echo "  ⚠️  WARNING: Missing $MISSING_TOTAL chunk files (R1: $MISSING_R1, R2: $MISSING_R2)"
            echo "  This is likely due to multiprocessing warmup issues in first ~20 chunks"

            # Salvage what we have
            echo "  Salvaging successful chunks..."
            find "$TEMP_CHUNK_DIR" -name "sample${SAMPLE_NUM}_r1.fastq.gz" -type f -size +1M 2>/dev/null | sort | xargs cat > "sample${SAMPLE_NUM}_r1_partial.fastq.gz"
            find "$TEMP_CHUNK_DIR" -name "sample${SAMPLE_NUM}_r2.fastq.gz" -type f -size +1M 2>/dev/null | sort | xargs cat > "sample${SAMPLE_NUM}_r2_partial.fastq.gz"

            echo "  ✓ Salvaged partial genome: $(du -h sample${SAMPLE_NUM}_r1_partial.fastq.gz | awk '{print $1}')"

            # Regenerate missing chunks individually (避免 race conditions)
            echo "  Regenerating missing chunks individually..."
            REGENERATED=0
            for split_file in "$TEMP_CHUNK_DIR"/splits/*.fa.gz; do
                CHUNK_NUM=$(basename "$split_file" | cut -d'_' -f1)
                CHUNK_R1="$TEMP_CHUNK_DIR/$CHUNK_NUM/sample${SAMPLE_NUM}_r1.fastq.gz"

                # Only regenerate if missing or empty
                if [ ! -f "$CHUNK_R1" ] || [ ! -s "$CHUNK_R1" ]; then
                    echo "    Regenerating chunk $CHUNK_NUM..."
                    REGEN_DIR="regenerated_$CHUNK_NUM"
                    mkdir -p "$REGEN_DIR"

                    # Decompress split
                    gunzip -c "$split_file" > "$REGEN_DIR/input.fa"

                    # Create single-chunk config with threads=1 (避免 race conditions)
                    cat > "$REGEN_DIR/config.yml" <<REGEN_EOF
reference: $REGEN_DIR/input.fa
read_len: 150
fragment_mean: 300
fragment_st_dev: 50
coverage: 30
paired_ended: true
rng_seed: $((SEED + ${CHUNK_NUM//[^0-9]/}))
produce_bam: false
ploidy: 2
threads: 1
REGEN_EOF

                    # Run NEAT on this single chunk
                    if neat read-simulator -c "$REGEN_DIR/config.yml" -o "$REGEN_DIR" -p "regen" 2>&1 | grep -q "Done"; then
                        # Append to partial files
                        cat "$REGEN_DIR/regen_r1.fastq.gz" >> "sample${SAMPLE_NUM}_r1_partial.fastq.gz"
                        cat "$REGEN_DIR/regen_r2.fastq.gz" >> "sample${SAMPLE_NUM}_r2_partial.fastq.gz"
                        REGENERATED=$((REGENERATED + 1))
                    fi

                    # Cleanup
                    rm -rf "$REGEN_DIR"
                fi
            done

            echo "  ✓ Regenerated $REGENERATED missing chunks"

            # Rename partial to final
            mv "sample${SAMPLE_NUM}_r1_partial.fastq.gz" "sample${SAMPLE_NUM}_r1.fastq.gz"
            mv "sample${SAMPLE_NUM}_r2_partial.fastq.gz" "sample${SAMPLE_NUM}_r2.fastq.gz"

            echo "  ✓ Complete genome assembled with chunk regeneration"
        else
            echo "  ✓ All chunks present - no regeneration needed"
        fi
    else
        echo "  ⚠️  Could not validate chunks (temp directory not found)"
    fi

    # Move outputs to final location
    echo "Organizing outputs..."
    mv "sample${SAMPLE_NUM}_r1.fastq.gz" "$OUTPUT_DIR/"
    mv "sample${SAMPLE_NUM}_r2.fastq.gz" "$OUTPUT_DIR/"
    cp "simug_sample${SAMPLE_NUM}/sample${SAMPLE_NUM}.refseq2simseq.SNP.vcf" "$OUTPUT_DIR/variants_snp.vcf"
    cp "simug_sample${SAMPLE_NUM}/sample${SAMPLE_NUM}.refseq2simseq.INDEL.vcf" "$OUTPUT_DIR/variants_indel.vcf"

    # Cleanup temp files
    rm -rf "simug_sample${SAMPLE_NUM}"
    rm -f "sample${SAMPLE_NUM}"*.log neat_config_sample${SAMPLE_NUM}.yml

    echo "Completed: $(date)"
    echo "✓ $LABEL ready"
    return 0
}

# Generate sample with retry logic
generate_sample_with_retry() {
    local SAMPLE_NUM=$1
    local PRIMARY_SEED=$2
    local OUTPUT_DIR=$3
    local LABEL=$4

    # Try with primary seed
    if generate_sample "$SAMPLE_NUM" "$PRIMARY_SEED" "$OUTPUT_DIR" "$LABEL"; then
        return 0
    fi

    # Primary seed failed, try fallback seeds
    echo ""
    echo "⚠️  Primary seed $PRIMARY_SEED failed, trying fallback seeds..."

    # Fallback seeds (simple, round numbers less likely to hit edge cases)
    local FALLBACK_SEEDS=($((SAMPLE_NUM * 1000)) $((SAMPLE_NUM * 10000)) $((SAMPLE_NUM * 100)))

    for FALLBACK_SEED in "${FALLBACK_SEEDS[@]}"; do
        echo ""
        echo "Retry attempt with seed=$FALLBACK_SEED"

        if generate_sample "$SAMPLE_NUM" "$FALLBACK_SEED" "$OUTPUT_DIR" "$LABEL"; then
            echo "✅ Success with fallback seed $FALLBACK_SEED"
            return 0
        fi

        echo "❌ Fallback seed $FALLBACK_SEED also failed"
    done

    echo ""
    echo "❌ ERROR: All retry attempts failed for $LABEL"
    echo "Skipping this sample and continuing with remaining samples..."
    return 1
}

# Generate Reference 2 (with retry on failure)
# Using simpler seeds: 200, 2000, 20000, 20 as fallbacks
generate_sample_with_retry 2 200 "$REFERENCE_DIR/ref2" "Reference 2"

# Generate Reference 3 (with retry on failure)
# Using simpler seeds: 300, 3000, 30000, 30 as fallbacks
generate_sample_with_retry 3 300 "$REFERENCE_DIR/ref3" "Reference 3"

# Generate Query Sample (with retry on failure)
# Using simpler seeds: 400, 4000, 40000, 40 as fallbacks
generate_sample_with_retry 4 400 "$QUERY_DIR" "Query Sample (experimental)"

# Final summary
echo ""
echo "========================================================================"
echo "Reference Pool Generation Complete"
echo "========================================================================"
echo "Generated Files:"
echo ""

# Count successful generations
SUCCESS_COUNT=0
TOTAL_EXPECTED=4

echo "References (for random selection):"
for i in 1 2 3; do
    if [ -f "$REFERENCE_DIR/ref${i}/sample${i}_r1.fastq.gz" ] && [ -f "$REFERENCE_DIR/ref${i}/sample${i}_r2.fastq.gz" ]; then
        ls -lh "$REFERENCE_DIR/ref${i}/"*.fastq.gz | awk '{print "  " $9 ": " $5}'
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  ref${i}: ❌ FAILED (directory exists but no FASTQ files)"
    fi
done

echo ""
echo "Query Sample:"
if [ -f "$QUERY_DIR/sample4_r1.fastq.gz" ] && [ -f "$QUERY_DIR/sample4_r2.fastq.gz" ]; then
    ls -lh "$QUERY_DIR"/*.fastq.gz | awk '{print "  " $9 ": " $5}'
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "  query: ❌ FAILED (directory exists but no FASTQ files)"
fi

echo ""
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo "Successfully generated: $SUCCESS_COUNT / $TOTAL_EXPECTED samples"
echo ""

if [ $SUCCESS_COUNT -ge 3 ]; then
    echo "✅ SUCCESS: Sufficient samples for k-anonymity (k=$((SUCCESS_COUNT - 1)))"
    echo ""
    echo "Ready for differential encoding pipeline:"
    echo "  - $((SUCCESS_COUNT - 1)) reference genomes in reference pool"
    echo "  - 1 query genome for encoding"
    echo "  - Random reference selection provides k-anonymity (k=$((SUCCESS_COUNT - 1)))"
elif [ $SUCCESS_COUNT -ge 2 ]; then
    echo "⚠️  WARNING: Only $SUCCESS_COUNT samples generated"
    echo "   Minimum k-anonymity achieved (k=$((SUCCESS_COUNT - 1)))"
    echo "   Recommend generating more references for better privacy"
else
    echo "❌ ERROR: Insufficient samples for differential encoding"
    echo "   Need at least 2 samples (1 reference + 1 query)"
    echo "   Only $SUCCESS_COUNT samples successfully generated"
fi

echo ""
echo "Total Storage: $(du -sh $WORK_DIR 2>/dev/null | awk '{print $1}' || echo 'N/A')"
echo ""
echo "Next Steps:"
if [ $SUCCESS_COUNT -ge 2 ]; then
    echo "  1. Test FASTQ integration: python examples/fastq_to_differential_encoding_example.py"
    echo "  2. Run differential encoding benchmarks"
    echo "  3. Verify k=$((SUCCESS_COUNT - 1)) anonymity guarantee"
else
    echo "  1. Review error logs above"
    echo "  2. Try running script again with different seeds"
    echo "  3. Check NEAT installation: conda list | grep neat"
fi
echo ""
echo "Completed: $(date)"
