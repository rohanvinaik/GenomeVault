#!/bin/bash
# Run Enhanced Privacy Pipeline with k=13 GUIDE samples
# 12 samples in reference pool + 1 experimental query (ERR3239454)

set -e

OUTPUT_DIR="benchmark_results/enhanced_privacy_k13_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "ENHANCED PRIVACY PIPELINE - k=13 GUIDE SAMPLES"
echo "================================================================================"
echo "Superposition reference: benchmark_results/superposition_7refs_sequential/superposition_consensus.fa.gz"
echo "Reference pool: 12 FASTQ samples (6 European, 2 East Asian, 2 African, 2 South Asian)"
echo "Query sample: ERR3239454 (European, experimental strand)"
echo "Output directory: $OUTPUT_DIR"
echo "Performance optimizations: ACTIVE (minimap2, pigz, sambamba, BCF streaming)"
echo "================================================================================"
echo ""

python benchmarks/run_enhanced_privacy_pipeline.py \
  --user-id "genomevault_k13_full_test" \
  --output "$OUTPUT_DIR" \
  --consensus-references benchmark_results/superposition_7refs_sequential/superposition_consensus.fa.gz \
  --reference-pool-fastq \
    data/downloaded/fastq/ERR3239276_1.fastq.gz data/downloaded/fastq/ERR3239276_2.fastq.gz \
    data/downloaded/fastq/ERR3239334_1.fastq.gz data/downloaded/fastq/ERR3239334_2.fastq.gz \
    data/downloaded/fastq/ERR3239475_1.fastq.gz data/downloaded/fastq/ERR3239475_2.fastq.gz \
    data/downloaded/fastq/european/ERR3239548/ERR3239548_1.fastq.gz data/downloaded/fastq/european/ERR3239548/ERR3239548_2.fastq.gz \
    data/downloaded/fastq/european/ERR3239590/ERR3239590_1.fastq.gz data/downloaded/fastq/european/ERR3239590/ERR3239590_2.fastq.gz \
    data/downloaded/fastq/european/ERR3239920/ERR3239920_1.fastq.gz data/downloaded/fastq/european/ERR3239920/ERR3239920_2.fastq.gz \
    data/downloaded/fastq/east_asian/ERR3239578/ERR3239578_1.fastq.gz data/downloaded/fastq/east_asian/ERR3239578/ERR3239578_2.fastq.gz \
    data/downloaded/fastq/east_asian/ERR3239612/ERR3239612_1.fastq.gz data/downloaded/fastq/east_asian/ERR3239612/ERR3239612_2.fastq.gz \
    data/downloaded/fastq/african/european/ERR3239756/ERR3239756_1.fastq.gz data/downloaded/fastq/african/european/ERR3239756/ERR3239756_2.fastq.gz \
    data/downloaded/fastq/african/european/ERR3239778/ERR3239778_1.fastq.gz data/downloaded/fastq/african/european/ERR3239778/ERR3239778_2.fastq.gz \
    data/downloaded/fastq/south_asian/european/ERR3239912/ERR3239912_1.fastq.gz data/downloaded/fastq/south_asian/european/ERR3239912/ERR3239912_2.fastq.gz \
    data/downloaded/fastq/south_asian/european/ERR3239934/ERR3239934_1.fastq.gz data/downloaded/fastq/south_asian/european/ERR3239934/ERR3239934_2.fastq.gz \
  --query-fastq data/downloaded/fastq/ERR3239454_1.fastq.gz data/downloaded/fastq/ERR3239454_2.fastq.gz \
  --threads 10 \
  2>&1 | tee "$OUTPUT_DIR/pipeline_run.log"

echo ""
echo "================================================================================"
echo "Pipeline complete! Results in: $OUTPUT_DIR"
echo "================================================================================"
