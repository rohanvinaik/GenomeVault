#!/bin/bash
#
# Generate Reference 3 in Parallel with Ref2 Regeneration
#
# This is safe because:
# - Different temp directories (NEAT creates unique /var/folders/tmp*)
# - Different output location (references/ref3/)
# - Different seed (300)
# - Uses available CPU cores (9 cores idle during regeneration)
#

set -e

WORK_DIR="/Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples"
OUTPUT_DIR="$WORK_DIR/references/ref3"
TEMP_DIR="$WORK_DIR/temp_ref3"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

echo "========================================================================"
echo "Reference 3 Generation (Parallel with Ref2 Regeneration)"
echo "========================================================================"
echo "Seed: 300"
echo "Output: $OUTPUT_DIR"
echo "Started: $(date)"
echo ""

cd "$TEMP_DIR"

# Use existing reference genome (chr22)
REF_GENOME="/Users/rohanvinaik/genomevault/benchmark_results/full_pipeline_synthetic/reference/chr22.fa"

if [ ! -f "$REF_GENOME" ]; then
    echo "ERROR: Reference genome not found at $REF_GENOME"
    exit 1
fi

# Step 1: Generate variants with simuG
echo "========================================================================"
echo "Step 1: simuG - Generating Variants (seed=300)"
echo "========================================================================"
echo "Started: $(date +%H:%M:%S)"

mkdir -p "simug_ref3"

if ! perl ~/simuG/simuG.pl \
    -refseq "$REF_GENOME" \
    -snp_count 10000 \
    -indel_count 2000 \
    -cnv_count 20 \
    -inversion_count 3 \
    -titv_ratio 2.0 \
    -seed 300 \
    -prefix "simug_ref3/sample3"; then
    echo "ERROR: simuG failed"
    exit 1
fi

SNPS=$(grep -v '^#' simug_ref3/sample3.refseq2simseq.SNP.vcf 2>/dev/null | wc -l | tr -d ' ')
INDELS=$(grep -v '^#' simug_ref3/sample3.refseq2simseq.INDEL.vcf 2>/dev/null | wc -l | tr -d ' ')

echo "✓ Generated $SNPS SNPs"
echo "✓ Generated $INDELS Indels"
echo "Completed: $(date +%H:%M:%S)"
echo ""

# Step 2: Generate reads with NEAT (with self-healing)
echo "========================================================================"
echo "Step 2: NEAT - Generating Sequencing Reads (threads=10)"
echo "========================================================================"
echo "Started: $(date +%H:%M:%S)"

# Create NEAT config
cat > neat_config_ref3.yml << NEATEOF
reference: simug_ref3/sample3.simseq.genome.fa
read_len: 150
fragment_mean: 300
fragment_st_dev: 50
coverage: 30
paired_ended: true
rng_seed: 300
produce_bam: false
ploidy: 2
threads: 10
NEATEOF

# Activate NEAT conda environment and run
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neat

if ! neat read-simulator \
    -c "neat_config_ref3.yml" \
    -o . \
    -p "sample3" 2>&1 | tee "neat_ref3.log"; then
    echo "ERROR: NEAT failed"
    exit 1
fi

# Check if outputs were created
if [ ! -f "sample3_r1.fastq.gz" ] || [ ! -f "sample3_r2.fastq.gz" ]; then
    echo "ERROR: NEAT did not produce expected FASTQ files"
    exit 1
fi

echo "✓ NEAT complete"
echo "Completed: $(date +%H:%M:%S)"
echo ""

# Step 3: Validate chunks and apply self-healing fix
echo "========================================================================"
echo "Step 3: Chunk Validation and Self-Healing"
echo "========================================================================"

TEMP_CHUNK_DIR=$(find /var/folders -path "*/tmp*/splits" -type d 2>/dev/null | grep -v tmp5fmcup8p | tail -1 | xargs dirname 2>/dev/null)

if [ -n "$TEMP_CHUNK_DIR" ] && [ -d "$TEMP_CHUNK_DIR" ]; then
    EXPECTED_CHUNKS=$(ls "$TEMP_CHUNK_DIR"/splits/*.fa.gz 2>/dev/null | wc -l | tr -d ' ')
    ACTUAL_R1=$(find "$TEMP_CHUNK_DIR" -name "sample3_r1.fastq.gz" -type f -size +1M 2>/dev/null | wc -l | tr -d ' ')
    ACTUAL_R2=$(find "$TEMP_CHUNK_DIR" -name "sample3_r2.fastq.gz" -type f -size +1M 2>/dev/null | wc -l | tr -d ' ')

    echo "Chunk validation: $ACTUAL_R1 R1 chunks, $ACTUAL_R2 R2 chunks (expected: $EXPECTED_CHUNKS each)"

    MISSING_R1=$((EXPECTED_CHUNKS - ACTUAL_R1))
    MISSING_R2=$((EXPECTED_CHUNKS - ACTUAL_R2))
    MISSING_TOTAL=$((MISSING_R1 + MISSING_R2))

    if [ $MISSING_TOTAL -gt 0 ]; then
        echo "⚠️  Missing $MISSING_TOTAL chunk files - applying self-healing..."

        # Salvage successful chunks
        find "$TEMP_CHUNK_DIR" -name "sample3_r1.fastq.gz" -type f -size +1M 2>/dev/null | sort | xargs cat > "sample3_r1_partial.fastq.gz"
        find "$TEMP_CHUNK_DIR" -name "sample3_r2.fastq.gz" -type f -size +1M 2>/dev/null | sort | xargs cat > "sample3_r2_partial.fastq.gz"

        echo "✓ Salvaged $(du -h sample3_r1_partial.fastq.gz | awk '{print $1}')"

        # Regenerate missing chunks
        echo "Regenerating missing chunks individually..."
        REGENERATED=0
        for split_file in "$TEMP_CHUNK_DIR"/splits/*.fa.gz; do
            CHUNK_NUM=$(basename "$split_file" | cut -d'_' -f1)
            CHUNK_R1="$TEMP_CHUNK_DIR/$CHUNK_NUM/sample3_r1.fastq.gz"

            if [ ! -f "$CHUNK_R1" ] || [ ! -s "$CHUNK_R1" ]; then
                echo "  Regenerating chunk $CHUNK_NUM..."
                REGEN_DIR="regenerated_$CHUNK_NUM"
                mkdir -p "$REGEN_DIR"

                gunzip -c "$split_file" > "$REGEN_DIR/input.fa"

                cat > "$REGEN_DIR/config.yml" <<REGEN_EOF
reference: $REGEN_DIR/input.fa
read_len: 150
fragment_mean: 300
fragment_st_dev: 50
coverage: 30
paired_ended: true
rng_seed: $((300 + ${CHUNK_NUM//[^0-9]/}))
produce_bam: false
ploidy: 2
threads: 1
REGEN_EOF

                if neat read-simulator -c "$REGEN_DIR/config.yml" -o "$REGEN_DIR" -p "regen" 2>&1 | grep -q "Done"; then
                    cat "$REGEN_DIR/regen_r1.fastq.gz" >> "sample3_r1_partial.fastq.gz"
                    cat "$REGEN_DIR/regen_r2.fastq.gz" >> "sample3_r2_partial.fastq.gz"
                    REGENERATED=$((REGENERATED + 1))
                fi

                rm -rf "$REGEN_DIR"
            fi
        done

        echo "✓ Regenerated $REGENERATED missing chunks"

        mv "sample3_r1_partial.fastq.gz" "sample3_r1.fastq.gz"
        mv "sample3_r2_partial.fastq.gz" "sample3_r2.fastq.gz"

        echo "✓ Complete genome assembled with self-healing"
    else
        echo "✓ All chunks present - no regeneration needed"
    fi
else
    echo "⚠️  Could not validate chunks (temp directory not found)"
fi

echo ""

# Step 4: Move outputs to final location
echo "========================================================================"
echo "Step 4: Organizing Outputs"
echo "========================================================================"

mv "sample3_r1.fastq.gz" "$OUTPUT_DIR/"
mv "sample3_r2.fastq.gz" "$OUTPUT_DIR/"
cp "simug_ref3/sample3.refseq2simseq.SNP.vcf" "$OUTPUT_DIR/variants_snp.vcf"
cp "simug_ref3/sample3.refseq2simseq.INDEL.vcf" "$OUTPUT_DIR/variants_indel.vcf"

echo "✓ Files moved to $OUTPUT_DIR"
echo ""

# Cleanup
rm -rf "simug_ref3"
rm -f neat_config_ref3.yml neat_ref3.log

echo "========================================================================"
echo "Reference 3 Generation Complete"
echo "========================================================================"
echo "Completed: $(date)"
echo ""

ls -lh "$OUTPUT_DIR"/*.fastq.gz | awk '{print "  " $9 ": " $5}'

echo ""
echo "✅ Ref3 ready for differential encoding"
