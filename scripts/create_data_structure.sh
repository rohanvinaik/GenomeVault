#!/bin/bash
#
# create_data_structure.sh
# Creates the organized directory structure for GenomeVault data acquisition
#

set -e

BASE_DIR="${1:-data/raw_fastq}"

echo "Creating GenomeVault data directory structure..."
echo "Base directory: $BASE_DIR"
echo ""

# Reference pool directories
ANCESTRY_GROUPS=("european_ancestry" "east_asian_ancestry" "african_ancestry" "south_asian_ancestry")

for ancestry in "${ANCESTRY_GROUPS[@]}"; do
    mkdir -p "${BASE_DIR}/reference_pools/${ancestry}/k10_pool_v1"
    echo "✓ Created ${ancestry} reference pool directory"
done

# Query sample directories
mkdir -p "${BASE_DIR}/query_samples/baseline/european"
mkdir -p "${BASE_DIR}/query_samples/baseline/east_asian"
mkdir -p "${BASE_DIR}/query_samples/baseline/african"
mkdir -p "${BASE_DIR}/query_samples/baseline/south_asian"

mkdir -p "${BASE_DIR}/query_samples/edge_cases/low_quality"
mkdir -p "${BASE_DIR}/query_samples/edge_cases/technical_variation"
mkdir -p "${BASE_DIR}/query_samples/edge_cases/complex_genomes"

mkdir -p "${BASE_DIR}/query_samples/clinical_validation/giab_gold_standard"
mkdir -p "${BASE_DIR}/query_samples/clinical_validation/clinvar_annotated"

echo "✓ Created query sample directories"

# Metadata directory
mkdir -p "${BASE_DIR}/metadata"
echo "✓ Created metadata directory"

# Processed data directories
mkdir -p "data/processed/alignments/layer2_reference_bams"
mkdir -p "data/processed/alignments/layer3_query_bams"
mkdir -p "data/processed/variants/layer2_reference_vcfs"
mkdir -p "data/processed/variants/layer3_query_vcfs"
mkdir -p "data/processed/benchmarks/by_pool"
mkdir -p "data/processed/benchmarks/by_ancestry"
mkdir -p "data/processed/benchmarks/by_scenario"

echo "✓ Created processed data directories"

# Acquisition plan directories
mkdir -p "data/acquisition_plan/download_scripts"
mkdir -p "data/acquisition_plan/validation_checksums"
mkdir -p "data/acquisition_plan/progress_tracking"

echo "✓ Created acquisition plan directories"

# Create README files
cat > "${BASE_DIR}/README.md" << 'EOF'
# GenomeVault Raw FASTQ Data

This directory contains raw whole-genome sequencing data organized for GenomeVault privacy-preserving genomic analysis.

## Structure

- `reference_pools/` - Reference genome pools for k-anonymity (k=10 per ancestry group)
- `query_samples/` - Query samples organized by test scenario
- `metadata/` - Central metadata registry and manifests

## Data Sources

All data is from publicly accessible repositories:
- European Nucleotide Archive (ENA)
- NCBI Sequence Read Archive (SRA)
- 1000 Genomes Project
- GIAB (Genome in a Bottle)

## Usage

See `data/acquisition_plan/DATA_ACQUISITION_PLAN.md` for complete documentation.
EOF

cat > "data/processed/README.md" << 'EOF'
# GenomeVault Processed Data

This directory contains processed genomic data from the GenomeVault pipeline.

## Structure

- `alignments/` - BAM files from Layer 2 (reference) and Layer 3 (query)
- `variants/` - VCF files with called variants
- `benchmarks/` - Performance metrics and validation results

## Data Lifecycle

Raw FASTQ → Layer 2 BAM/VCF → Layer 3 BAM/VCF → Layer 4 HDC+ZK output

See pipeline documentation for details.
EOF

echo ""
echo "=========================================="
echo "✓ Directory structure created successfully"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review data acquisition plan: data/acquisition_plan/DATA_ACQUISITION_PLAN.md"
echo "2. Begin Phase 1 downloads: bash scripts/download_phase1_european_k10.sh"
echo "3. Validate downloads: bash scripts/validate_downloads.sh"
echo ""
