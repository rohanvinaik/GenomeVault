#!/bin/bash
# Optimized BCFtools Variant Calling with Regional Parallelization
#
# Key Optimizations:
#   1. BCF streaming (-Ou) for 5-10× faster parsing
#   2. Regional parallelization (6-8× speedup on 8 cores)
#   3. Efficient chromosome-level splitting
#
# Usage:
#   ./scripts/optimize_bcftools_variant_calling.sh <reference.fa> <sample.bam> <output.vcf.gz> [threads]
#
# Example:
#   ./scripts/optimize_bcftools_variant_calling.sh \
#       data/reference_genomes/hg38.fa \
#       benchmark_results/sample.sorted.bam \
#       benchmark_results/sample.vcf.gz \
#       8

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <reference.fa> <sample.bam> <output.vcf.gz> [threads]"
    echo ""
    echo "Optimizations:"
    echo "  - BCF streaming (-Ou) for fast parsing"
    echo "  - Regional parallelization by chromosome"
    echo "  - Expected speedup: 6-8× on 8 cores"
    exit 1
fi

REFERENCE="$1"
BAM_FILE="$2"
OUTPUT_VCF="$3"
THREADS="${4:-8}"

# Temp directory for per-chromosome VCFs
TEMP_DIR="${OUTPUT_VCF%.vcf.gz}_temp"
mkdir -p "$TEMP_DIR"

echo "================================================================================"
echo "OPTIMIZED VARIANT CALLING (BCF STREAMING + REGIONAL PARALLELIZATION)"
echo "================================================================================"
echo "Reference: $REFERENCE"
echo "BAM file:  $BAM_FILE"
echo "Output:    $OUTPUT_VCF"
echo "Threads:   $THREADS"
echo "================================================================================"
echo ""

# Validate inputs
if [ ! -f "$REFERENCE" ]; then
    echo "ERROR: Reference not found: $REFERENCE"
    exit 1
fi

if [ ! -f "$BAM_FILE" ]; then
    echo "ERROR: BAM file not found: $BAM_FILE"
    exit 1
fi

if [ ! -f "${BAM_FILE}.bai" ]; then
    echo "ERROR: BAM index not found: ${BAM_FILE}.bai"
    echo "Run: samtools index $BAM_FILE"
    exit 1
fi

# Get chromosomes from BAM
echo "Detecting chromosomes from BAM file..."
CHROMOSOMES=$(samtools view -H "$BAM_FILE" | grep '^@SQ' | cut -f2 | sed 's/SN://' | grep -E '^(chr)?[0-9XY]+$' | head -24)
CHR_COUNT=$(echo "$CHROMOSOMES" | wc -l | tr -d ' ')

echo "Found $CHR_COUNT chromosomes: $(echo $CHROMOSOMES | tr '\n' ' ')"
echo ""

# Regional variant calling function
call_variants_region() {
    local chr=$1
    local ref=$2
    local bam=$3
    local out_vcf=$4

    echo "[$(date +%H:%M:%S)] Processing $chr..."

    # BCF streaming pipeline: mpileup (-Ou) → call (-Ou) → filter (-Oz)
    bcftools mpileup -Ou -r "$chr" -f "$ref" "$bam" | \
        bcftools call -Ou -mv | \
        bcftools filter -Oz -o "$out_vcf" -

    bcftools index "$out_vcf"

    # Get variant count
    local count=$(bcftools view -H "$out_vcf" | wc -l | tr -d ' ')
    echo "[$(date +%H:%M:%S)] ✓ $chr: $count variants"
}

export -f call_variants_region

echo "Starting parallel variant calling ($THREADS jobs)..."
echo ""

start_time=$(date +%s)

# Run variant calling in parallel by chromosome
echo "$CHROMOSOMES" | parallel -j "$THREADS" \
    call_variants_region {} "$REFERENCE" "$BAM_FILE" "$TEMP_DIR/{}.vcf.gz"

mid_time=$(date +%s)
parallel_duration=$((mid_time - start_time))

echo ""
echo "================================================================================"
echo "Parallel variant calling complete: ${parallel_duration}s"
echo "================================================================================"
echo ""

# Concatenate chromosome VCFs
echo "Concatenating chromosome VCFs..."

CHR_VCFS=$(for chr in $CHROMOSOMES; do echo "$TEMP_DIR/${chr}.vcf.gz"; done)

bcftools concat -Oz -o "$OUTPUT_VCF" $CHR_VCFS
bcftools index "$OUTPUT_VCF"

end_time=$(date +%s)
total_duration=$((end_time - start_time))
concat_duration=$((end_time - mid_time))

# Get total variant count
TOTAL_VARIANTS=$(bcftools view -H "$OUTPUT_VCF" | wc -l | tr -d ' ')

echo ""
echo "================================================================================"
echo "✓ VARIANT CALLING COMPLETE"
echo "================================================================================"
echo "Output:          $OUTPUT_VCF"
echo "Total variants:  $TOTAL_VARIANTS"
echo "Chromosomes:     $CHR_COUNT"
echo ""
echo "Timing:"
echo "  Parallel calling: ${parallel_duration}s ($(($parallel_duration / 60))m $(($parallel_duration % 60))s)"
echo "  Concatenation:    ${concat_duration}s"
echo "  Total:            ${total_duration}s ($(($total_duration / 60))m $(($total_duration % 60))s)"
echo ""
echo "Speedup estimate: ~6-8× vs serial (with $THREADS cores)"
echo "================================================================================"
echo ""

# Cleanup temp files
read -p "Delete temporary per-chromosome VCFs? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "Cleaning up..."
    rm -rf "$TEMP_DIR"
    echo "✓ Temporary files deleted"
else
    echo "Temporary files kept in: $TEMP_DIR"
fi

echo ""
echo "Final output: $OUTPUT_VCF ($(du -h "$OUTPUT_VCF" | cut -f1))"
