#!/bin/bash
# Download 1000 Genomes Project WHOLE GENOME reference panels
# Public-facing simple reference data for GenomeVault superposition

set -e

OUTPUT_DIR="vcf_pool"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

BASE_URL="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"

echo "================================================================================"
echo "Downloading 1000 Genomes Project WHOLE GENOME Reference Panels"
echo "================================================================================"
echo "Source: 1000 Genomes Phase 3 (May 2013 release)"
echo "Target: ALL chromosomes (1-22, X, Y)"
echo "Output: $OUTPUT_DIR/"
echo "Total size: ~15-20 GB (compressed)"
echo "================================================================================"
echo ""

# Download ALL chromosomes sequentially
for chr in {1..22} X Y; do
    echo "================================================================================"
    echo "[Chr $chr] Downloading ALL populations - chromosome $chr"
    echo "================================================================================"
    
    if [ "$chr" == "X" ] || [ "$chr" == "Y" ]; then
        filename="ALL.chr${chr}.phase3_shapeit2_mvncall_integrated_v1b.20130502.genotypes.vcf.gz"
    else
        filename="ALL.chr${chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    fi
    
    output_file="$OUTPUT_DIR/ALL.chr${chr}.phase3.vcf.gz"
    
    # Skip if already downloaded
    if [ -f "$output_file" ]; then
        size=$(ls -lh "$output_file" | awk '{print $5}')
        echo "✓ Already downloaded: $output_file ($size)"
        echo "   Skipping..."
        echo ""
        continue
    fi
    
    # Download with progress
    wget -c -O "$output_file" \
        "$BASE_URL/$filename" \
        2>&1 | tee "logs/download_1000g_chr${chr}.log"
    
    # Verify download
    if [ -f "$output_file" ]; then
        size=$(ls -lh "$output_file" | awk '{print $5}')
        echo "✓ Downloaded: chr$chr ($size)"
    else
        echo "✗ Failed to download chr$chr"
        exit 1
    fi
    
    # Index with tabix
    echo "   Indexing with tabix..."
    tabix -p vcf "$output_file"
    echo "   ✓ Indexed"
    echo ""
done

echo "================================================================================"
echo "Download Complete Summary"
echo "================================================================================"
ls -lh "$OUTPUT_DIR"/*.vcf.gz | awk '{print $9 " - " $5}'

echo ""
echo "Total data downloaded:"
du -sh "$OUTPUT_DIR"

echo ""
echo "================================================================================"
echo "Next steps:"
echo "1. Build superposition genome with statistical uncertainty"
echo "2. Validate reference pool separation (k=12 FASTQ samples)"
echo "3. Run complete privacy-preserving pipeline"
echo "================================================================================"
