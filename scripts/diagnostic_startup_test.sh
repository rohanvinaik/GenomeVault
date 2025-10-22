#!/bin/bash
#
# Diagnostic Test - NEAT Startup Phase Analysis
#
# This script runs NEAT with comprehensive diagnostic logging
# to identify the root cause of the startup race condition
# affecting chunks 1-21.
#

set -e

WORK_DIR="/Users/rohanvinaik/genomevault/benchmark_results/diagnostic_test"
TEST_GENOME="/Users/rohanvinaik/genomevault/benchmark_results/full_pipeline_synthetic/reference/chr22.fa"
SEED=999
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================================================"
echo "NEAT Startup Diagnostic Test"
echo "========================================================================"
echo "Started: $(date)"
echo "Goal: Identify why first ~21 chunks systematically fail"
echo ""

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Create minimal NEAT config focused on startup
cat > neat_diagnostic_config.yml << EOF
reference: $TEST_GENOME
read_len: 150
fragment_mean: 300
fragment_st_dev: 50
coverage: 10
paired_ended: true
rng_seed: $SEED
produce_bam: false
produce_vcf: true
ploidy: 2
threads: 4
EOF

echo "[$(date +%H:%M:%S)] Running NEAT with diagnostic logging..."
echo "Configuration:"
cat neat_diagnostic_config.yml
echo ""
echo "Log file: $WORK_DIR/diagnostic_${TIMESTAMP}.log"
echo ""

# Activate NEAT environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neat

# Run NEAT with comprehensive logging
neat read-simulator \
    -c neat_diagnostic_config.yml \
    -o . \
    -p diagnostic_test 2>&1 | tee "diagnostic_${TIMESTAMP}.log"

NEAT_EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Test Complete - Exit Code: $NEAT_EXIT_CODE"
echo "========================================================================"
echo ""

# Analyze results
echo "Chunk Analysis:"
echo "---------------"

# Count generated chunks
TOTAL_CHUNKS=$(find . -name "diagnostic_test_r1.fastq.gz" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "Total chunks generated: $TOTAL_CHUNKS"

# Check for diagnostic log messages
echo ""
echo "Diagnostic Log Summary:"
echo "-----------------------"
grep -E "\[GENOMEVAULT_DIAG\]" "diagnostic_${TIMESTAMP}.log" | tail -20

# Check for errors
echo ""
echo "Errors/Warnings:"
echo "----------------"
grep -iE "(error|warning|exception|traceback|failed|timeout)" "diagnostic_${TIMESTAMP}.log" | grep -v "GENOMEVAULT_DIAG" | tail -10 || echo "No errors found"

# List generated chunk files
echo ""
echo "Generated Files:"
echo "----------------"
ls -lh diagnostic_test*.fastq.gz 2>/dev/null || echo "No FASTQ files generated"

echo ""
echo "Full diagnostic log: $WORK_DIR/diagnostic_${TIMESTAMP}.log"
echo "Analyze with: grep '\[GENOMEVAULT_DIAG\]' diagnostic_${TIMESTAMP}.log"
echo ""
