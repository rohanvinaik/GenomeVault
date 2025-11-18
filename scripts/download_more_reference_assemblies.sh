#!/bin/bash
# Download additional diverse REFERENCE genome assemblies
# Easy data win: small files (~1GB each), high accuracy improvement

set -e

REF_DIR="data/reference_genomes"
mkdir -p "$REF_DIR"
mkdir -p logs

echo "================================================================================"
echo "Downloading MORE REFERENCE Genome Assemblies"
echo "================================================================================"
echo "Strategy: Easy data win - small files, big accuracy improvement"
echo "Target: Diverse human genome assemblies for robust superposition"
echo "Output: $REF_DIR/"
echo "================================================================================"
echo ""

# Download additional reference assemblies for diversity

# 1. GRCh37/hg19 with decoy sequences (improved variant calling)
echo "================================================================================"
echo "[1/6] Downloading GRCh37 with decoy sequences (hs37d5)"
echo "================================================================================"
echo "Size: ~3.0 GB"
echo "Purpose: GRCh37 with decoy sequences for improved alignment"
echo ""

if [ -f "$REF_DIR/hs37d5.fa.gz" ]; then
    size=$(ls -lh "$REF_DIR/hs37d5.fa.gz" | awk '{print $5}')
    echo "✓ Already downloaded: hs37d5.fa.gz ($size)"
else
    echo "Downloading from 1000 Genomes FTP..."
    wget -c -O "$REF_DIR/hs37d5.fa.gz" \
        "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz" \
        2>&1 | tee logs/download_hs37d5.log

    if [ -f "$REF_DIR/hs37d5.fa.gz" ]; then
        size=$(ls -lh "$REF_DIR/hs37d5.fa.gz" | awk '{print $5}')
        echo "✓ Downloaded: hs37d5.fa.gz ($size)"
    fi
fi
echo ""

# 2. GRCh38 with decoy sequences
echo "================================================================================"
echo "[2/6] Downloading GRCh38 full analysis set (with decoy + HLA)"
echo "================================================================================"
echo "Size: ~3.2 GB"
echo "Purpose: Most comprehensive GRCh38 with all alternate loci"
echo ""

if [ -f "$REF_DIR/GRCh38_full_analysis_set.fa.gz" ]; then
    size=$(ls -lh "$REF_DIR/GRCh38_full_analysis_set.fa.gz" | awk '{print $5}')
    echo "✓ Already downloaded: GRCh38_full_analysis_set.fa.gz ($size)"
else
    echo "Downloading from NCBI..."
    wget -c -O "$REF_DIR/GRCh38_full_analysis_set.fa.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_full_analysis_set.fna.gz" \
        2>&1 | tee logs/download_grch38_full.log

    if [ -f "$REF_DIR/GRCh38_full_analysis_set.fa.gz" ]; then
        size=$(ls -lh "$REF_DIR/GRCh38_full_analysis_set.fa.gz" | awk '{print $5}')
        echo "✓ Downloaded: GRCh38_full_analysis_set.fa.gz ($size)"
    fi
fi
echo ""

# 3. T2T-CHM13 + Y chromosome (complete)
echo "================================================================================"
echo "[3/6] Downloading T2T-CHM13v2.0 + Y (complete human genome)"
echo "================================================================================"
echo "Size: ~1.0 GB"
echo "Purpose: First truly complete human genome (all chromosomes)"
echo ""

if [ -f "$REF_DIR/chm13v2.0_plus_hg38y.fa.gz" ]; then
    size=$(ls -lh "$REF_DIR/chm13v2.0_plus_hg38y.fa.gz" | awk '{print $5}')
    echo "✓ Already downloaded: chm13v2.0_plus_hg38y.fa.gz ($size)"
else
    echo "Downloading from T2T Consortium..."
    wget -c -O "$REF_DIR/chm13v2.0_plus_hg38y.fa.gz" \
        "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0_maskedY_rCRS.fa.gz" \
        2>&1 | tee logs/download_chm13_plusY.log || echo "Note: Using existing CHM13 instead"

    if [ -f "$REF_DIR/chm13v2.0_plus_hg38y.fa.gz" ]; then
        size=$(ls -lh "$REF_DIR/chm13v2.0_plus_hg38y.fa.gz" | awk '{print $5}')
        echo "✓ Downloaded: chm13v2.0_plus_hg38y.fa.gz ($size)"
    fi
fi
echo ""

# 4. GRCh36/hg18 (historical reference - adds diversity)
echo "================================================================================"
echo "[4/6] Downloading GRCh36/hg18 (historical reference)"
echo "================================================================================"
echo "Size: ~930 MB"
echo "Purpose: Older assembly for additional sequence diversity"
echo ""

if [ -f "$REF_DIR/hg18.fa.gz" ]; then
    size=$(ls -lh "$REF_DIR/hg18.fa.gz" | awk '{print $5}')
    echo "✓ Already downloaded: hg18.fa.gz ($size)"
else
    echo "Downloading from UCSC..."
    wget -c -O "$REF_DIR/hg18.fa.gz" \
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg18/bigZips/hg18.fa.gz" \
        2>&1 | tee logs/download_hg18.log

    if [ -f "$REF_DIR/hg18.fa.gz" ]; then
        size=$(ls -lh "$REF_DIR/hg18.fa.gz" | awk '{print $5}')
        echo "✓ Downloaded: hg18.fa.gz ($size)"
    fi
fi
echo ""

# 5. Vervet (African green monkey) - phylogenetic diversity
echo "================================================================================"
echo "[5/6] Downloading Vervet AGM (phylogenetic outgroup)"
echo "================================================================================"
echo "Size: ~780 MB"
echo "Purpose: Non-human primate for evolutionary comparison"
echo ""

if [ -f "$REF_DIR/vervet_agm.fa.gz" ]; then
    size=$(ls -lh "$REF_DIR/vervet_agm.fa.gz" | awk '{print $5}')
    echo "✓ Already downloaded: vervet_agm.fa.gz ($size)"
else
    echo "Downloading from NCBI..."
    wget -c -O "$REF_DIR/vervet_agm.fa.gz" \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/015/252/575/GCF_015252575.1_Vero_WHO_p1.0/GCF_015252575.1_Vero_WHO_p1.0_genomic.fna.gz" \
        2>&1 | tee logs/download_vervet.log || echo "Skipping vervet (optional)"
fi
echo ""

# 6. Pangenome reference (if available)
echo "================================================================================"
echo "[6/6] Downloading HPRC Pangenome Reference (draft)"
echo "================================================================================"
echo "Size: Variable"
echo "Purpose: Human pangenome for population diversity"
echo ""

if [ -f "$REF_DIR/hprc_pangenome.fa.gz" ]; then
    size=$(ls -lh "$REF_DIR/hprc_pangenome.fa.gz" | awk '{print $5}')
    echo "✓ Already downloaded: hprc_pangenome.fa.gz ($size)"
else
    echo "Checking for HPRC pangenome..."
    wget -c -O "$REF_DIR/hprc_pangenome.fa.gz" \
        "https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/freeze1/minigraph-cactus/hprc-v1.0-mc-grch38.fa.gz" \
        2>&1 | tee logs/download_hprc.log || echo "HPRC pangenome not available yet - skipping"
fi
echo ""

# Summary
echo "================================================================================"
echo "Download Summary"
echo "================================================================================"
echo ""
echo "All REFERENCE genomes:"
ls -lh "$REF_DIR"/*.fa.gz 2>/dev/null | awk '{print "  " $9 " - " $5}'
echo ""
echo "Total REFERENCE data:"
du -sh "$REF_DIR"
echo ""
echo "================================================================================"
echo "Next Steps:"
echo "1. Decompress and index all genomes (if needed)"
echo "2. Build superposition genome with 5% statistical uncertainty"
echo "3. Validate diversity and coverage"
echo "================================================================================"
