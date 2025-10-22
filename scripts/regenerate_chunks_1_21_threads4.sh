#!/bin/bash
#
# Regenerate Missing Chunks 1-21 with threads=4 (devs' original value)
#
# Hypothesis: NEAT code may have internal dependencies on threads=4
# This could explain why chunks 1-21 fail with other thread counts.
#

set -e

WORK_DIR="/Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples"
REF_GENOME="/Users/rohanvinaik/genomevault/benchmark_results/full_pipeline_synthetic/reference/chr22.fa"
REGEN_DIR="$WORK_DIR/temp/chunks_1-21_threads4"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================================================"
echo "Regenerating Chunks 1-21 with threads=4 (Devs' Original Configuration)"
echo "========================================================================"
echo "Started: $(date)"
echo ""
echo "Hypothesis: Internal NEAT logic may depend on threads=4 specifically"
echo ""

mkdir -p "$REGEN_DIR"

# Activate NEAT environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neat

#
# Function: Generate sample with limited coverage targeting chunks 1-21
#
generate_sample_chunks_1_21() {
    local sample_name=$1
    local seed=$2
    local sample_dir="$REGEN_DIR/${sample_name}"

    echo ""
    echo "========================================================================"
    echo "Generating $sample_name (seed=$seed) - chunks 1-21 with threads=4"
    echo "========================================================================"

    mkdir -p "$sample_dir"
    cd "$sample_dir"

    # Create simuG genome
    echo "[$(date +%H:%M:%S)] Running simuG with seed=$seed..."
    perl ~/simuG/simuG.pl \
        -refseq "$REF_GENOME" \
        -snp_count 10000 \
        -indel_count 2000 \
        -cnv_count 20 \
        -inversion_count 3 \
        -titv_ratio 2.0 \
        -seed $seed \
        -prefix "${sample_name}"

    echo "✓ simuG complete"

    # NEAT config with threads=4 (EXACTLY as devs intended)
    # Using lower coverage to target just first chunks
    cat > neat_config_${sample_name}.yml << EOF
reference: ${sample_name}.simseq.genome.fa
read_len: 150
fragment_mean: 300
fragment_st_dev: 50
coverage: 15
paired_ended: true
rng_seed: $seed
produce_bam: false
produce_vcf: false
ploidy: 2
threads: 4
EOF

    echo "[$(date +%H:%M:%S)] Running NEAT with threads=4..."
    echo "Config:"
    cat neat_config_${sample_name}.yml
    echo ""

    # Run NEAT with diagnostic logging enabled
    neat read-simulator \
        -c neat_config_${sample_name}.yml \
        -o . \
        -p ${sample_name} 2>&1 | tee neat_${sample_name}_${TIMESTAMP}.log

    NEAT_EXIT=$?

    echo ""
    echo "NEAT exit code: $NEAT_EXIT"
    echo ""

    # Check what was generated
    echo "Checking generated chunks..."
    CHUNK_COUNT=$(find /var/folders -name "${sample_name}_r1.fastq.gz" -type f -size +1M 2>/dev/null | wc -l | tr -d ' ')
    echo "Chunks generated: $CHUNK_COUNT"

    if [ $CHUNK_COUNT -gt 0 ]; then
        echo "✓ Successfully generated chunks for $sample_name"

        # Extract chunk numbers
        echo ""
        echo "Chunk numbers generated:"
        find /var/folders -name "${sample_name}_r1.fastq.gz" -type f -size +1M 2>/dev/null \
            | sed 's/.*\/\([0-9]*\)__.*/\1/' \
            | sort -n \
            | uniq \
            | head -30
        echo ""
    else
        echo "⚠️  No chunks generated for $sample_name"
    fi

    cd "$REGEN_DIR"
}

#
# Generate all three samples
#
echo "Will generate 3 samples with threads=4:"
echo "  - Ref2 (seed=200)"
echo "  - Ref3 (seed=300)"
echo "  - Query (seed=1)"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to proceed..."
sleep 5

# Ref2
generate_sample_chunks_1_21 "sample2" 200

# Ref3
generate_sample_chunks_1_21 "sample3" 300

# Query
generate_sample_chunks_1_21 "sample4" 1

echo ""
echo "========================================================================"
echo "Regeneration Complete"
echo "========================================================================"
echo ""
echo "Final Status:"
echo "-------------"

for sample in sample2 sample3 sample4; do
    count=$(find /var/folders -name "${sample}_r1.fastq.gz" -type f -size +1M 2>/dev/null | wc -l | tr -d ' ')
    echo "$sample: $count chunks generated"
done

echo ""
echo "Next Steps:"
echo "1. Verify chunk ranges include 1-21"
echo "2. Merge with existing chunks 22-102"
echo "3. Create final reference pool with 100% coverage"
echo ""
echo "Working directory: $REGEN_DIR"
echo ""
