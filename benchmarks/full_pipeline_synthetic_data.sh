#!/bin/bash
#
# GenomeVault End-to-End Pipeline with Synthetic Genomic Data
#
# *** IMPORTANT: CHR22-ONLY BENCHMARK ***
# This benchmark uses chromosome 22 (~50MB) for testing.
# Results are appropriate for:
#   ✓ Chromosome-level analysis
#   ✓ Regional genomic analysis
#   ✗ NOT representative of whole-genome analysis (would be 60× larger)
#   ✗ NOT appropriate for single-SNP identification (too large)
#   ✗ NOT representative of simultaneous whole-genome analysis (too small)
#
# This script:
# 1. Downloads a reference genome (chr22)
# 2. Uses simuG to generate synthetic variants
# 3. Uses NEAT to generate realistic sequencing reads
# 4. Processes through GenomeVault pipeline
# 5. Measures actual performance at each stage
#
# Requirements:
# - simuG installed at ~/simuG
# - NEAT conda environment available
# - GenomeVault Python environment
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="${PROJECT_ROOT}/benchmark_results/full_pipeline_synthetic"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================================================"
echo "GenomeVault Full Pipeline Benchmark with Synthetic Data"
echo "========================================================================"
echo "Timestamp: $TIMESTAMP"
echo "Work Directory: $WORK_DIR"
echo ""

# Create work directory
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Function to log with timestamp
log() {
    echo "[$(date +%H:%M:%S)] $1"
}

# Function to measure command time
time_command() {
    local desc="$1"
    shift
    log "Starting: $desc"
    local start=$(date +%s.%N)
    "$@"
    local end=$(date +%s.%N)
    local duration=$(echo "$end - $start" | bc)
    log "✓ Completed: $desc (${duration}s)"
    echo "$desc,$duration" >> "${WORK_DIR}/timings.csv"
}

# Initialize timing log
echo "Stage,Duration_Seconds" > "${WORK_DIR}/timings.csv"

echo "========================================================================"
echo "Stage 1: Download Reference Genome (chr22)"
echo "========================================================================"

REF_DIR="${WORK_DIR}/reference"
mkdir -p "$REF_DIR"

if [ ! -f "${REF_DIR}/chr22.fa" ]; then
    log "Downloading chr22 from UCSC..."
    time_command "Download chr22 reference" \
        curl -o "${REF_DIR}/chr22.fa.gz" \
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"

    log "Decompressing..."
    gunzip "${REF_DIR}/chr22.fa.gz"

    log "Creating FASTA index..."
    samtools faidx "${REF_DIR}/chr22.fa" 2>/dev/null || {
        log "Warning: samtools not found, skipping indexing (not critical)"
    }
else
    log "✓ chr22.fa already exists"
fi

REF_GENOME="${REF_DIR}/chr22.fa"
REF_SIZE=$(du -h "$REF_GENOME" | cut -f1)
log "Reference genome ready: $REF_SIZE"

echo ""
echo "========================================================================"
echo "Stage 2: Generate Synthetic Variants with simuG"
echo "========================================================================"

SIMUG_DIR="${WORK_DIR}/simug_output"
mkdir -p "$SIMUG_DIR"

# Create simuG configuration
cat > "${SIMUG_DIR}/simug_config.txt" << 'EOF'
# simuG configuration for realistic human variant simulation
# Format: -snp_count <num> -indel_count <num> -cnv_count <num> -inversion_count <num>

# Realistic variant counts for a 50Mb chromosome region:
# - SNPs: ~1 per 1000bp = 50,000 SNPs
# - Indels: ~1 per 10,000bp = 5,000 indels
# - CNVs: ~10-20 per chromosome
# - Inversions: ~1-2 per chromosome

# For this benchmark, we'll use moderate numbers:
-snp_count 10000
-snp_vcf ${SIMUG_DIR}/snps.vcf
-indel_count 2000
-indel_vcf ${SIMUG_DIR}/indels.vcf
-cnv_count 20
-cnv_vcf ${SIMUG_DIR}/cnvs.vcf
-inversion_count 3
-inversion_vcf ${SIMUG_DIR}/inversions.vcf
-titv_ratio 2.0
-prefix ${SIMUG_DIR}/simulated
-seed 42
EOF

log "Running simuG to generate variants..."
log "NOTE: Generating realistic variant counts for chr22 (~50Mb region)"

time_command "simuG variant generation" \
    perl ~/simuG/simuG.pl \
    -refseq "$REF_GENOME" \
    -snp_count 10000 \
    -indel_count 2000 \
    -cnv_count 20 \
    -inversion_count 3 \
    -titv_ratio 2.0 \
    -prefix "${SIMUG_DIR}/simulated" \
    -seed 42

# simuG creates VCF files with specific naming
# Look for the generated files
log "Locating generated VCF files..."
ls -lh "${SIMUG_DIR}/"*.vcf 2>/dev/null || log "Note: VCF files may have different names"

# Find and merge all VCF files
log "Merging VCF files..."
find "${SIMUG_DIR}" -name "*.vcf" -type f -exec cat {} \; \
    | grep -v "^#" | sort -k1,1 -k2,2n > "${SIMUG_DIR}/all_variants_sorted.vcf.tmp"

# Add VCF header
cat > "${SIMUG_DIR}/all_variants.vcf" << 'VCFHEADER'
##fileformat=VCFv4.2
##reference=chr22
##source=simuG
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE
VCFHEADER

cat "${SIMUG_DIR}/all_variants_sorted.vcf.tmp" >> "${SIMUG_DIR}/all_variants.vcf"
rm "${SIMUG_DIR}/all_variants_sorted.vcf.tmp"

TOTAL_VARIANTS=$(grep -v "^#" "${SIMUG_DIR}/all_variants.vcf" | wc -l)
log "✓ Generated $TOTAL_VARIANTS total variants"
log "  - SNPs: 10,000"
log "  - Indels: 2,000"
log "  - CNVs: 20"
log "  - Inversions: 3"

SIMULATED_GENOME="${SIMUG_DIR}/simulated.simseq.genome.fa"
log "✓ Simulated genome: $(du -h $SIMULATED_GENOME | cut -f1)"

echo ""
echo "========================================================================"
echo "Stage 3: Generate Sequencing Reads with NEAT"
echo "========================================================================"

NEAT_DIR="${WORK_DIR}/neat_output"
mkdir -p "$NEAT_DIR"

# Create NEAT configuration
cat > "${NEAT_DIR}/neat_config.yml" << EOFNEAT
# NEAT configuration for realistic Illumina sequencing simulation

# Reference genome (use the simulated genome from simuG)
reference: ${SIMULATED_GENOME}

# Output directory
output: ${NEAT_DIR}

# Read length and coverage
read_len: 150
coverage: 30

# Paired-end configuration
paired_ended: true
fragment_mean: 300
fragment_st_dev: 50

# Number of threads
threads: 4

# Random seed for reproducibility
rng_seed: 42
EOFNEAT

log "Activating NEAT conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate neat 2>/dev/null || {
    log "Warning: Could not activate NEAT conda environment"
    log "Trying to run NEAT directly..."
}

log "Running NEAT to generate sequencing reads..."
log "This will take 5-15 minutes for 30x coverage of chr22..."

time_command "NEAT read generation" \
    neat read-simulator \
    -c "${NEAT_DIR}/neat_config.yml" \
    -o "$NEAT_DIR"

# Check output
if [ -f "${NEAT_DIR}/read1.fq" ] && [ -f "${NEAT_DIR}/read2.fq" ]; then
    READ1_SIZE=$(du -h "${NEAT_DIR}/read1.fq" | cut -f1)
    READ2_SIZE=$(du -h "${NEAT_DIR}/read2.fq" | cut -f1)
    READ1_READS=$(echo $(cat "${NEAT_DIR}/read1.fq" | wc -l) / 4 | bc)
    log "✓ Generated paired-end reads:"
    log "  - Read 1: $READ1_SIZE (${READ1_READS} reads)"
    log "  - Read 2: $READ2_SIZE"
else
    log "❌ NEAT output not found, checking for alternative file names..."
    ls -lh "$NEAT_DIR"/*.fq "$NEAT_DIR"/*.fastq 2>/dev/null || true
fi

echo ""
echo "========================================================================"
echo "Stage 4: Run GenomeVault Differential Encoding Pipeline"
echo "========================================================================"

GV_DIR="${WORK_DIR}/genomevault_output"
mkdir -p "$GV_DIR"

log "Converting VCF to GenomeVault format..."

# Create Python script to run the pipeline
cat > "${GV_DIR}/run_pipeline.py" << 'EOFPYTHON'
#!/usr/bin/env python3
"""
Run GenomeVault pipeline on synthetic genomic data.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add GenomeVault to path
sys.path.insert(0, '/Users/rohanvinaik/genomevault')

from genomevault.differential_encoding import (
    Genome,
    Variant,
    ReferenceGenome,
    compute_reference_hash,
)
from genomevault.hypervector_transform import (
    UnifiedGenomicEncoder,
    EncodingMode,
)

def parse_vcf(vcf_file: Path) -> Genome:
    """Parse VCF file into GenomeVault Genome format."""
    variants_by_chr = {}

    with open(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 5:
                continue

            chrom = fields[0]
            pos = int(fields[1])
            ref = fields[3]
            alt = fields[4]

            # Simple genotype assignment (assume heterozygous for variants)
            genotype = "0/1"
            quality = 99.0

            variant = Variant(
                chromosome=chrom,
                position=pos,
                ref=ref,
                alt=alt,
                genotype=genotype,
                quality=quality
            )

            if chrom not in variants_by_chr:
                variants_by_chr[chrom] = []
            variants_by_chr[chrom].append(variant)

    return Genome(
        genome_id="synthetic_sample_001",
        assembly="hg38",
        chromosomes=variants_by_chr
    )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vcf', required=True, help='Input VCF file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'input_vcf': str(args.vcf),
        'stages': {}
    }

    # Stage 1: Parse VCF
    print("=" * 70)
    print("Stage 1: Parsing VCF")
    print("=" * 70)
    start = time.perf_counter()
    genome = parse_vcf(Path(args.vcf))
    parse_time = time.perf_counter() - start

    total_variants = sum(len(v) for v in genome.chromosomes.values())
    print(f"✓ Parsed {total_variants} variants in {parse_time:.3f}s")
    print(f"  Chromosomes: {list(genome.chromosomes.keys())}")
    print()

    results['stages']['vcf_parsing'] = {
        'time_seconds': parse_time,
        'total_variants': total_variants,
        'chromosomes': list(genome.chromosomes.keys())
    }

    # Stage 2: Create reference genome (use half of variants as reference)
    print("=" * 70)
    print("Stage 2: Creating Reference Genome")
    print("=" * 70)
    start = time.perf_counter()

    ref_variants = {}
    for chrom, variants in genome.chromosomes.items():
        # Use every other variant as reference
        ref_variants[chrom] = variants[::2]

    temp_ref = ReferenceGenome(
        genome_id="reference_001",
        assembly="hg38",
        variants=ref_variants,
        cryptographic_hash="temp"
    )
    reference = ReferenceGenome(
        genome_id="reference_001",
        assembly="hg38",
        variants=ref_variants,
        cryptographic_hash=compute_reference_hash(temp_ref)
    )

    ref_time = time.perf_counter() - start
    ref_variant_count = sum(len(v) for v in ref_variants.values())
    print(f"✓ Created reference with {ref_variant_count} variants in {ref_time:.3f}s")
    print()

    results['stages']['reference_creation'] = {
        'time_seconds': ref_time,
        'reference_variants': ref_variant_count
    }

    # Stage 3: Initialize encoder
    print("=" * 70)
    print("Stage 3: Initializing GenomeVault Encoder")
    print("=" * 70)
    start = time.perf_counter()

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        encoder = UnifiedGenomicEncoder(
            mode=EncodingMode.DIFFERENTIAL,
            dimension=8192,
            reference_dir=tmpdir,
        )
        encoder.differential_encoder.add_reference(reference)

        init_time = time.perf_counter() - start
        print(f"✓ Encoder initialized in {init_time:.3f}s")
        print(f"  Mode: {encoder.mode}")
        print(f"  Dimension: 8192")
        print(f"  References loaded: 1")
        print()

        results['stages']['encoder_init'] = {
            'time_seconds': init_time,
            'dimension': 8192,
            'mode': 'differential'
        }

        # Stage 4: Differential Encoding
        print("=" * 70)
        print("Stage 4: Differential Encoding")
        print("=" * 70)
        start = time.perf_counter()

        result = encoder.encode_genome(genome)

        diff_time = time.perf_counter() - start
        print(f"✓ Differential encoding completed in {diff_time:.3f}s")
        print(f"  Hypervector shape: {result.hypervector.shape if hasattr(result.hypervector, 'shape') else 'N/A'}")
        print(f"  Throughput: {total_variants / diff_time:.0f} variants/sec")
        print()

        results['stages']['differential_encoding'] = {
            'time_seconds': diff_time,
            'throughput_variants_per_sec': total_variants / diff_time,
        }

    # Stage 5: Save results
    print("=" * 70)
    print("Stage 5: Saving Results")
    print("=" * 70)

    results_file = output_dir / 'pipeline_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {results_file}")
    print()

    # Summary
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    total_time = sum(s['time_seconds'] for s in results['stages'].values())
    print(f"Total variants processed: {total_variants}")
    print(f"Total pipeline time: {total_time:.3f}s")
    print(f"Overall throughput: {total_variants / total_time:.0f} variants/sec")
    print()
    print("Breakdown:")
    for stage, data in results['stages'].items():
        print(f"  {stage}: {data['time_seconds']:.3f}s")
    print("=" * 70)

    return 0

if __name__ == '__main__':
    sys.exit(main())
EOFPYTHON

chmod +x "${GV_DIR}/run_pipeline.py"

log "Running GenomeVault pipeline..."

time_command "GenomeVault full pipeline" \
    python "${GV_DIR}/run_pipeline.py" \
    --vcf "${SIMUG_DIR}/all_variants.vcf" \
    --output-dir "$GV_DIR"

echo ""
echo "========================================================================"
echo "BENCHMARK COMPLETE"
echo "========================================================================"

# Generate summary report
REPORT_FILE="${WORK_DIR}/benchmark_report_${TIMESTAMP}.md"

cat > "$REPORT_FILE" << EOFREPORT
# GenomeVault Full Pipeline Benchmark Report

**Date**: $(date)
**Reference**: chr22 (hg38)
**Simulated Variants**: ${TOTAL_VARIANTS}
**Coverage**: 30x

## ⚠️ IMPORTANT: Benchmark Scope

**This benchmark uses chromosome 22 (~50MB) ONLY.**

**Results are appropriate for:**
- ✓ Chromosome-level analysis performance
- ✓ Regional genomic analysis (targeted sequencing)
- ✓ Per-chromosome processing pipelines

**Results are NOT representative of:**
- ✗ Whole-genome analysis (would be ~60× larger, ~3GB)
- ✗ Single-SNP identification (this test is too large)
- ✗ Simultaneous whole-genome analysis (this test is too small)

**Extrapolation to whole genome:**
- Multiply timings by ~60× for whole genome (assuming linear scaling)
- Actual whole-genome performance may vary due to memory/cache effects

## Pipeline Stages

EOFREPORT

# Add timing data
echo "" >> "$REPORT_FILE"
echo "### Execution Times" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| Stage | Duration (seconds) |" >> "$REPORT_FILE"
echo "|-------|-------------------|" >> "$REPORT_FILE"

while IFS=, read -r stage duration; do
    if [ "$stage" != "Stage" ]; then  # Skip header
        echo "| $stage | $duration |" >> "$REPORT_FILE"
    fi
done < "${WORK_DIR}/timings.csv"

# Add GenomeVault results if available
if [ -f "${GV_DIR}/pipeline_results.json" ]; then
    echo "" >> "$REPORT_FILE"
    echo "### GenomeVault Performance" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo '```json' >> "$REPORT_FILE"
    cat "${GV_DIR}/pipeline_results.json" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"
echo "## Files Generated" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
find "$WORK_DIR" -type f -size +1M -exec du -h {} \; | \
    awk '{print "- " $2 " (" $1 ")"}' >> "$REPORT_FILE"

log "Report generated: $REPORT_FILE"
echo ""
cat "$REPORT_FILE"

echo ""
echo "========================================================================"
echo "All results saved to: $WORK_DIR"
echo "Report: $REPORT_FILE"
echo "========================================================================"
