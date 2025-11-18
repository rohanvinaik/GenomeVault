#!/bin/bash
# Download additional complete human reference genome assemblies
# For robust public-facing REFERENCE superposition strand

set -e

REF_DIR="data/reference_genomes"
mkdir -p "$REF_DIR"
mkdir -p logs

echo "================================================================================"
echo "Downloading Additional REFERENCE Genome Assemblies"
echo "================================================================================"
echo "Purpose: Build robust superposition genome with statistical uncertainty"
echo "Target: Complete genome assemblies (FASTA format)"
echo "Output: $REF_DIR/"
echo "================================================================================"
echo ""

# Check existing genomes
echo "Current REFERENCE genomes:"
ls -lh "$REF_DIR"/*.fa.gz 2>/dev/null || echo "  None yet"
echo ""

# GRCh38 no-alt analysis set (recommended for variant calling)
echo "================================================================================"
echo "[1/4] Downloading GRCh38 no-alt analysis set"
echo "================================================================================"
echo "Source: NCBI/UCSC"
echo "Size: ~3.0 GB"
echo "Purpose: Standard analysis set without alternate loci"
echo ""

if [ -f "$REF_DIR/GRCh38_no_alt.fa.gz" ]; then
    echo "✓ Already downloaded: GRCh38_no_alt.fa.gz"
else
    echo "Downloading from NCBI..."
    wget -c -O "$REF_DIR/GRCh38_no_alt.fa.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz" \
        2>&1 | tee logs/download_grch38_noalt.log

    if [ -f "$REF_DIR/GRCh38_no_alt.fa.gz" ]; then
        size=$(ls -lh "$REF_DIR/GRCh38_no_alt.fa.gz" | awk '{print $5}')
        echo "✓ Downloaded: GRCh38_no_alt.fa.gz ($size)"
    else
        echo "✗ Failed to download GRCh38_no_alt"
    fi
fi
echo ""

# GRCh37/hg19 (if not present)
echo "================================================================================"
echo "[2/4] Verifying GRCh37/hg19"
echo "================================================================================"
if [ -f "$REF_DIR/hg19.fa.gz" ]; then
    size=$(ls -lh "$REF_DIR/hg19.fa.gz" | awk '{print $5}')
    echo "✓ Already present: hg19.fa.gz ($size)"
else
    echo "Downloading hg19 from UCSC..."
    wget -c -O "$REF_DIR/hg19.fa.gz" \
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz" \
        2>&1 | tee logs/download_hg19.log

    if [ -f "$REF_DIR/hg19.fa.gz" ]; then
        size=$(ls -lh "$REF_DIR/hg19.fa.gz" | awk '{print $5}')
        echo "✓ Downloaded: hg19.fa.gz ($size)"
    fi
fi
echo ""

# GRCh38/hg38 (if not present)
echo "================================================================================"
echo "[3/4] Verifying GRCh38/hg38"
echo "================================================================================"
if [ -f "$REF_DIR/hg38.fa.gz" ]; then
    size=$(ls -lh "$REF_DIR/hg38.fa.gz" | awk '{print $5}')
    echo "✓ Already present: hg38.fa.gz ($size)"
else
    echo "Downloading hg38 from UCSC..."
    wget -c -O "$REF_DIR/hg38.fa.gz" \
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz" \
        2>&1 | tee logs/download_hg38.log

    if [ -f "$REF_DIR/hg38.fa.gz" ]; then
        size=$(ls -lh "$REF_DIR/hg38.fa.gz" | awk '{print $5}')
        echo "✓ Downloaded: hg38.fa.gz ($size)"
    fi
fi
echo ""

# T2T-CHM13v2.0 (if not present)
echo "================================================================================"
echo "[4/4] Verifying T2T-CHM13v2.0"
echo "================================================================================"
if [ -f "$REF_DIR/chm13v2.0.fa.gz" ]; then
    size=$(ls -lh "$REF_DIR/chm13v2.0.fa.gz" | awk '{print $5}')
    echo "✓ Already present: chm13v2.0.fa.gz ($size)"
else
    echo "Downloading T2T-CHM13v2.0 from NCBI..."
    wget -c -O "$REF_DIR/chm13v2.0.fa.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/009/914/755/GCA_009914755.4_T2T-CHM13v2.0/GCA_009914755.4_T2T-CHM13v2.0_genomic.fna.gz" \
        2>&1 | tee logs/download_chm13.log

    if [ -f "$REF_DIR/chm13v2.0.fa.gz" ]; then
        size=$(ls -lh "$REF_DIR/chm13v2.0.fa.gz" | awk '{print $5}')
        echo "✓ Downloaded: chm13v2.0.fa.gz ($size)"
    fi
fi
echo ""

# Index all genomes with samtools faidx
echo "================================================================================"
echo "Indexing all REFERENCE genomes with samtools"
echo "================================================================================"
for genome in "$REF_DIR"/*.fa.gz; do
    if [ -f "$genome" ]; then
        base=$(basename "$genome")
        if [ ! -f "$genome.fai" ]; then
            echo "Indexing $base..."
            samtools faidx "$genome"
            echo "✓ Indexed: $base"
        else
            echo "✓ Already indexed: $base"
        fi
    fi
done
echo ""

# Summary
echo "================================================================================"
echo "REFERENCE Genome Download Complete"
echo "================================================================================"
echo ""
echo "Downloaded REFERENCE genomes:"
ls -lh "$REF_DIR"/*.fa.gz 2>/dev/null | awk '{print "  " $9 " - " $5}'
echo ""
echo "Total REFERENCE data:"
du -sh "$REF_DIR"
echo ""
echo "Index files (.fai):"
ls -lh "$REF_DIR"/*.fa.gz.fai 2>/dev/null | awk '{print "  " $9 " - " $5}' || echo "  None yet"
echo ""
echo "================================================================================"
echo "Next Steps:"
echo "1. Build superposition genome with 5% statistical uncertainty"
echo "2. Validate REFERENCE vs GUIDE vs EXPERIMENTAL separation"
echo "3. Run complete privacy-preserving pipeline with 'big boy data'"
echo "================================================================================"
