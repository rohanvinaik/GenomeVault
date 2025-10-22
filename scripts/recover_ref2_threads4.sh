#!/bin/bash
#
# Ref2 Recovery with threads=4 (devs' tested value)
# Regenerates chunks 1-21 using existing simuG genome
#

set -e

WORK_DIR="/Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples"
RECOVERY_DIR="$WORK_DIR/temp/ref2_recovery_t4"
REF_GENOME="/Users/rohanvinaik/genomevault/benchmark_results/full_pipeline_synthetic/reference/chr22.fa"
SEED=200

echo "========================================================================"
echo "Ref2 Recovery - threads=4 (Missing Chunks 1-21)"
echo "========================================================================"
echo "Started: $(date)"

mkdir -p "$RECOVERY_DIR"
cd "$RECOVERY_DIR"

# Step 1: Regenerate simuG genome with seed=200
echo "[$(date +%H:%M:%S)] Regenerating simuG genome (seed=$SEED)..."
perl ~/simuG/simuG.pl \
    -refseq "$REF_GENOME" \
    -snp_count 10000 \
    -indel_count 2000 \
    -cnv_count 20 \
    -inversion_count 3 \
    -titv_ratio 2.0 \
    -seed $SEED \
    -prefix "sample2"

echo "✓ simuG complete"

# Step 2: NEAT with threads=4 (devs' tested value)
echo "[$(date +%H:%M:%S)] Running NEAT with threads=4..."

cat > neat_config_ref2_t4.yml << EOF
reference: sample2.simseq.genome.fa
read_len: 150
fragment_mean: 300
fragment_st_dev: 50
coverage: 30
paired_ended: true
rng_seed: $SEED
produce_bam: false
ploidy: 2
threads: 4
EOF

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neat

nohup neat read-simulator \
    -c neat_config_ref2_t4.yml \
    -o . \
    -p sample2 > neat_ref2_t4.log 2>&1 &

NEAT_PID=$!
echo "NEAT started, PID: $NEAT_PID (threads=4)"
echo "Monitor: tail -f $RECOVERY_DIR/neat_ref2_t4.log"
echo ""
echo "Will salvage chunks after NEAT completes or hangs"
