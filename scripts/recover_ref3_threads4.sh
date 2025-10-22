#!/bin/bash
#
# Ref3 Recovery with threads=4 (devs' tested value)
# Uses existing simuG genome from temp/simug_sample3
#

set -e

WORK_DIR="/Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples"
RECOVERY_DIR="$WORK_DIR/temp/ref3_recovery_t4"
SIMUG_GENOME="$WORK_DIR/temp/simug_sample3/sample3.simseq.genome.fa"
SEED=300

echo "========================================================================"
echo "Ref3 Recovery - threads=4 (Missing Chunks 1-21)"
echo "========================================================================"
echo "Started: $(date)"

mkdir -p "$RECOVERY_DIR"
cd "$RECOVERY_DIR"

# Step 1: Verify simuG genome exists
if [ ! -f "$SIMUG_GENOME" ]; then
    echo "ERROR: simuG genome not found at $SIMUG_GENOME"
    exit 1
fi

echo "[$(date +%H:%M:%S)] Using existing simuG genome: $SIMUG_GENOME"

# Step 2: NEAT with threads=4 (devs' tested value)
echo "[$(date +%H:%M:%S)] Running NEAT with threads=4..."

cat > neat_config_ref3_t4.yml << EOF
reference: $SIMUG_GENOME
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
    -c neat_config_ref3_t4.yml \
    -o . \
    -p sample3 > neat_ref3_t4.log 2>&1 &

NEAT_PID=$!
echo "NEAT started, PID: $NEAT_PID (threads=4)"
echo "Monitor: tail -f $RECOVERY_DIR/neat_ref3_t4.log"
echo ""
echo "Will salvage chunks after NEAT completes or hangs"
