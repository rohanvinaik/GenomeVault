#!/bin/bash
# Wrapper to run pipeline with comprehensive error logging

LOG_FILE="pipeline_error_$(date +%Y%m%d_%H%M%S).log"

echo "Starting pipeline with error logging..." | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run with stderr and stdout captured
python benchmarks/run_enhanced_privacy_pipeline.py \
  --user-id demo@genomevault.com \
  --consensus-references \
    data/reference_genomes/hg38.fa.gz \
    data/reference_genomes/hg19.fa.gz \
    data/reference_genomes/chm13v2.0.fa.gz \
  --reference-pool-fastq \
    data/downloaded/fastq/ERR3239276_1.fastq.gz \
    data/downloaded/fastq/ERR3239276_2.fastq.gz \
    data/downloaded/fastq/ERR3239454_1.fastq.gz \
    data/downloaded/fastq/ERR3239454_2.fastq.gz \
    data/downloaded/fastq/ERR3239475_1.fastq.gz \
    data/downloaded/fastq/ERR3239475_2.fastq.gz \
  --query-fastq \
    data/downloaded/fastq/ERR3239334_1.fastq.gz \
    data/downloaded/fastq/ERR3239334_2.fastq.gz \
  --output benchmark_results/enhanced_privacy_pipeline \
  --enable-superposition \
  --enable-user-randomization \
  --enable-rolling-pool \
  --enable-challenge-detection \
  --threads 8 \
  --preset production \
  2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

echo "" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Pipeline failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check $LOG_FILE for details"
else
    echo "SUCCESS: Pipeline completed successfully" | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE
