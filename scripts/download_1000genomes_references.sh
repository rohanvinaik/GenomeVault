#!/bin/bash
# Download 1000 Genomes Project standard reference panels
# These are "public facing" simple reference genomes for superposition alignment

set -e

OUTPUT_DIR="vcf_pool"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Using chromosome 22 for faster downloads (full genome would be ~450MB per population)
# Chr22 is ~50MB per population - much more manageable for testing

# 1000 Genomes Phase 3 release - chromosome 22
BASE_URL="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"

echo "================================================================================"
echo "Downloading 1000 Genomes Project Standard Reference Panels"
echo "================================================================================"
echo "Target: Chromosome 22 (for speed and manageability)"
echo "Source: 1000 Genomes Phase 3 (May 2013 release)"
echo "Output: $OUTPUT_DIR/"
echo "================================================================================"
echo ""

# Download ALL populations integrated VCF for chr22
echo "[1/5] Downloading ALL populations (integrated) - chr22"
echo "This file contains variants from ALL 2,504 individuals across all populations"
wget -c -O "$OUTPUT_DIR/ALL.chr22.phase3.vcf.gz" \
    "$BASE_URL/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz" \
    2>&1 | tee logs/download_1000g_ALL.log

echo ""
echo "[2/5] Downloading EUR (European) panel - chr22"
echo "European ancestry: CEU, TSI, FIN, GBR, IBS populations"
wget -c -O "$OUTPUT_DIR/EUR.chr22.phase3.vcf.gz" \
    "$BASE_URL/supporting/EUR.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz" \
    2>&1 | tee logs/download_1000g_EUR.log || echo "EUR-specific file may not exist, using ALL instead"

echo ""
echo "[3/5] Downloading AFR (African) panel - chr22"
echo "African ancestry: YRI, LWK, GWD, MSL, ESN, ASW, ACB populations"
wget -c -O "$OUTPUT_DIR/AFR.chr22.phase3.vcf.gz" \
    "$BASE_URL/supporting/AFR.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz" \
    2>&1 | tee logs/download_1000g_AFR.log || echo "AFR-specific file may not exist, using ALL instead"

echo ""
echo "[4/5] Downloading EAS (East Asian) panel - chr22"
echo "East Asian ancestry: CHB, JPT, CHS, CDX, KHV populations"
wget -c -O "$OUTPUT_DIR/EAS.chr22.phase3.vcf.gz" \
    "$BASE_URL/supporting/EAS.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz" \
    2>&1 | tee logs/download_1000g_EAS.log || echo "EAS-specific file may not exist, using ALL instead"

echo ""
echo "[5/5] Downloading SAS (South Asian) panel - chr22"
echo "South Asian ancestry: GIH, PJL, BEB, STU, ITU populations"
wget -c -O "$OUTPUT_DIR/SAS.chr22.phase3.vcf.gz" \
    "$BASE_URL/supporting/SAS.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz" \
    2>&1 | tee logs/download_1000g_SAS.log || echo "SAS-specific file may not exist, using ALL instead"

echo ""
echo "================================================================================"
echo "Download Summary"
echo "================================================================================"
ls -lh "$OUTPUT_DIR"/*.vcf.gz 2>/dev/null || echo "No files downloaded yet"

echo ""
echo "Verifying file integrity..."
for file in "$OUTPUT_DIR"/*.vcf.gz; do
    if [ -f "$file" ]; then
        size=$(ls -l "$file" | awk '{print $5}')
        if [ "$size" -lt 1000 ]; then
            echo "WARNING: $file appears incomplete (size: $size bytes)"
        else
            echo "✓ $file (size: $size bytes)"
        fi
    fi
done

echo ""
echo "================================================================================"
echo "Next steps:"
echo "1. Index VCF files: for f in vcf_pool/*.vcf.gz; do tabix -p vcf \$f; done"
echo "2. Validate with GenomeVault: python scripts/genomevault_setup_references.py"
echo "================================================================================"
