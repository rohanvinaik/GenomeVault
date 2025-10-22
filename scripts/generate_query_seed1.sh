#!/bin/bash
#
# Generate Query Sample with seed=1
# Testing hypothesis: NEAT devs only tested with low seeds like 1, 42
#

set -e

WORK_DIR="/Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples"
OUTPUT_DIR="$WORK_DIR/query"
TEMP_DIR="$WORK_DIR/temp_query"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

echo "========================================================================"
echo "Query Sample Generation (seed=1 - likely well-tested)"
echo "========================================================================"
echo "Started: $(date)"

cd "$TEMP_DIR"

REF_GENOME="/Users/rohanvinaik/genomevault/benchmark_results/full_pipeline_synthetic/reference/chr22.fa"

# Step 1: simuG
echo ""
echo "Step 1: simuG - Generating Variants (seed=1)"
echo "----------------------------------------------------------------------"
mkdir -p "simug_query"

perl ~/simuG/simuG.pl \
    -refseq "$REF_GENOME" \
    -snp_count 10000 \
    -indel_count 2000 \
    -cnv_count 20 \
    -inversion_count 3 \
    -titv_ratio 2.0 \
    -seed 1 \
    -prefix "simug_query/sample4"

echo "✓ simuG complete"

# Step 2: NEAT
echo ""
echo "Step 2: NEAT - Generating Sequencing Reads (threads=10, seed=1)"
echo "----------------------------------------------------------------------"

cat > neat_config_query.yml << NEATEOF
reference: simug_query/sample4.simseq.genome.fa
read_len: 150
fragment_mean: 300
fragment_st_dev: 50
coverage: 30
paired_ended: true
rng_seed: 1
produce_bam: false
ploidy: 2
threads: 10
NEATEOF

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neat

neat read-simulator \
    -c "neat_config_query.yml" \
    -o . \
    -p "sample4" 2>&1 | tee "neat_query.log"

echo ""
echo "NEAT complete - checking for chunks..."

# Step 3: Salvage chunks (NEAT will likely hang again)
TEMP_CHUNK_DIR=$(find /var/folders -path "*/tmp*/splits" -type d 2>/dev/null | grep -v tmp5fmcup8p | grep -v tmpm3g7m5u9 | tail -1 | xargs dirname 2>/dev/null)

if [ -n "$TEMP_CHUNK_DIR" ] && [ -d "$TEMP_CHUNK_DIR" ]; then
    CHUNKS=$(find "$TEMP_CHUNK_DIR" -name "sample4_r1.fastq.gz" -type f -size +1M 2>/dev/null | wc -l | tr -d ' ')
    echo "Generated $CHUNKS chunks"
    
    if [ $CHUNKS -gt 0 ]; then
        echo "Salvaging chunks..."
        find "$TEMP_CHUNK_DIR" -name "sample4_r1.fastq.gz" -type f -size +1M 2>/dev/null | sort | xargs cat > sample4_r1.fastq.gz
        find "$TEMP_CHUNK_DIR" -name "sample4_r2.fastq.gz" -type f -size +1M 2>/dev/null | sort | xargs cat > sample4_r2.fastq.gz
        echo "✓ Salvaged $CHUNKS chunks"
    fi
fi

# Move to final location
if [ -f "sample4_r1.fastq.gz" ] && [ -s "sample4_r1.fastq.gz" ]; then
    mv sample4_r1.fastq.gz "$OUTPUT_DIR/"
    mv sample4_r2.fastq.gz "$OUTPUT_DIR/"
    cp simug_query/sample4.refseq2simseq.SNP.vcf "$OUTPUT_DIR/variants_snp.vcf"
    cp simug_query/sample4.refseq2simseq.INDEL.vcf "$OUTPUT_DIR/variants_indel.vcf"
    
    echo ""
    echo "========================================================================"
    echo "Query Sample Complete"
    echo "========================================================================"
    ls -lh "$OUTPUT_DIR"/*.fastq.gz | awk '{print $9 ": " $5}'
else
    echo "ERROR: No output files generated"
    exit 1
fi
