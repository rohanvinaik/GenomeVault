# GenomeVault Data Acquisition Guide

Complete guide for finding, downloading, and integrating real genomic data into GenomeVault.

## 🎯 Quick Start

```bash
# 1. Install dependencies
conda install -c bioconda sra-tools samtools aria2

# 2. Download curated test dataset (~500 MB)
python scripts/download_genomic_data.py --preset quick-test

# 3. Run GenomeVault pipeline
python benchmarks/run_alignment_optimized_pipeline.py --preset production
```

---

## 📊 Data Sources Overview

### 1. **1000 Genomes Project** (Recommended for diversity)
- **What**: 2,504 genomes from 26 populations worldwide
- **Data types**: FASTQ, BAM, VCF
- **Coverage**: Low (4-6x) to high (30x+)
- **Best for**: Population diversity, common variants, SNPs
- **URL**: https://www.internationalgenome.org/

### 2. **GIAB (Genome in a Bottle)** (Recommended for quality)
- **What**: Gold-standard reference samples with high-confidence calls
- **Samples**: NA12878, AshkenazimTrio, Chinese trio, etc.
- **Data types**: FASTQ, BAM, VCF
- **Coverage**: Very high (30-300x)
- **Best for**: Benchmarking, validation, gold standards
- **URL**: https://www.nist.gov/programs-projects/genome-bottle

### 3. **SRA (Sequence Read Archive)** (Largest collection)
- **What**: 40+ petabases of sequencing data from thousands of studies
- **Data types**: FASTQ (primarily)
- **Coverage**: Varies widely
- **Best for**: Specific studies, rare variants, disease-specific data
- **URL**: https://www.ncbi.nlm.nih.gov/sra

### 4. **Platinum Genomes** (High quality)
- **What**: 17-member pedigree sequenced to 200x coverage
- **Data types**: FASTQ, VCF
- **Best for**: High-confidence variants, Mendelian inheritance
- **URL**: https://github.com/Illumina/PlatinumGenomes

### 5. **gnomAD** (Population frequencies)
- **What**: 125,748 exomes + 71,702 genomes
- **Data types**: VCF only (aggregated variants)
- **Best for**: Variant frequency data, filtering
- **URL**: https://gnomad.broadinstitute.org/

---

## 🚀 Download Methods

### Method 1: Using Our Script (Easiest)

```bash
# Check what's available
python scripts/download_genomic_data.py --list-datasets

# Quick test set (~500 MB - chr22 only)
python scripts/download_genomic_data.py --preset quick-test

# Specific dataset
python scripts/download_genomic_data.py --dataset giab_na12878_illumina

# Reference genome only
python scripts/download_genomic_data.py --reference hg38_chr22

# VCF variants
python scripts/download_genomic_data.py --vcf 1kg_phase3_chr22
```

### Method 2: SRA Toolkit (For SRA data)

```bash
# Install SRA Toolkit
conda install -c bioconda sra-tools

# Download by accession
prefetch SRR000001
fasterq-dump SRR000001 --split-files --threads 4

# Or use our wrapper
python scripts/download_genomic_data.py --sra-accession SRR000001
```

### Method 3: SRA Explorer (Bulk downloads)

**Best for**: Downloading multiple samples from a study

1. Go to https://sra-explorer.info/
2. Search for your study (e.g., "PRJNA000001")
3. Select samples
4. Click "Add to collection"
5. Download as:
   - **Bash script** (wget/curl commands)
   - **Aspera script** (fastest, requires Aspera Connect)
   - **Python script** (easiest to integrate)

Example using generated script:
```bash
# Download the bash script from SRA Explorer
chmod +x download_sra_study.sh
./download_sra_study.sh

# Files saved to current directory
ls -lh *.fastq.gz
```

### Method 4: Direct Download (FTP/HTTP)

For one-off downloads:

```bash
# Using wget
wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/exome_alignment/HG00096.chrom22.ILLUMINA.bwa.GBR.exome.20121211.bam

# Using curl
curl -O https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz

# Using aria2c (fastest, parallel)
aria2c -x 16 -s 16 ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/exome_alignment/HG00096.chrom22.ILLUMINA.bwa.GBR.exome.20121211.bam
```

---

## 📁 Recommended Dataset Combinations

### For Quick Testing (< 1 GB)
```bash
# Reference: chr22 only (~15 MB)
python scripts/download_genomic_data.py --reference hg38_chr22

# Sample: 1000 Genomes chr22 exome (~150 MB)
python scripts/download_genomic_data.py --dataset 1kg_hg00096_exome

# Variants: 1000 Genomes Phase 3 chr22 (~300 MB)
python scripts/download_genomic_data.py --vcf 1kg_phase3_chr22
```

### For Comprehensive Testing (5-10 GB)
```bash
# Multiple samples from different populations
# European (GBR)
python scripts/download_genomic_data.py --dataset 1kg_hg00096_exome

# African (YRI)  
python scripts/download_genomic_data.py --dataset 1kg_na19238_wgs

# Gold standard
python scripts/download_genomic_data.py --dataset giab_na12878_illumina

# Complete reference
python scripts/download_genomic_data.py --reference hg38_full
```

### For Diverse Genomic Variations

**SNPs (Single Nucleotide Polymorphisms)**:
- 1000 Genomes VCF files (millions of SNPs)
- Common in all datasets

**Indels (Insertions/Deletions)**:
- GIAB high-confidence calls
- 1000 Genomes VCF files

**Structural Variants (SVs)**:
- Use SRA to find studies like:
  - PRJNA587799 (Human Genome Structural Variation Consortium)
  - PRJEB26711 (1000 Genomes SVs)

**Copy Number Variants (CNVs)**:
- Look for cancer genomics datasets in SRA
- Example: TCGA (The Cancer Genome Atlas) via SRA

**Mosaic Variants**:
- GTEx project samples
- Aging studies in SRA

---

## 🔍 Finding Specific Data Types in SRA

### Finding Datasets with Specific Characteristics

**1. Search SRA using Entrez Direct:**

```bash
# Install
conda install -c bioconda entrez-direct

# Find whole genome sequencing studies
esearch -db sra -query "whole genome sequencing[Strategy] AND homo sapiens[Organism]" | \
  efetch -format runinfo | \
  head -20

# Find high-coverage datasets
esearch -db sra -query "Illumina[Platform] AND WGS[Strategy] AND 30x[Coverage]" | \
  efetch -format runinfo > high_coverage_runs.csv

# Find structural variant studies
esearch -db sra -query "structural variants[All Fields] AND human[Organism]" | \
  efetch -format docsum
```

**2. Using SRA Run Selector:**
- Go to: https://www.ncbi.nlm.nih.gov/Traces/study/
- Search for project (e.g., "PRJNA000001")
- Filter by:
  - Coverage
  - Library strategy (WGS, WXS, RNA-Seq)
  - Platform (Illumina, PacBio, Nanopore)
  - Library layout (SINGLE, PAIRED)
- Download RunInfo table
- Extract accessions and download

**3. Curated Collections:**

**Cancer Genomes**:
```
Project: PRJNA433409 (TCGA)
Platform: Illumina
Data: Tumor/normal pairs
```

**Population Diversity**:
```
Project: PRJEB31736 (Human Genome Diversity Project)
Samples: 929 individuals, 54 populations
```

**Structural Variants**:
```
Project: PRJNA587799 (HGSVC)
Data: Long-read + short-read
Variants: SVs, CNVs, complex rearrangements
```

**Ancient DNA**:
```
Project: PRJNA516872
Data: Ancient human genomes
Variants: Population-specific variants
```

---

## 🔧 Data Processing Pipeline

### 1. Download and Organize

```bash
# Create directory structure
mkdir -p data/downloaded/{fastq,vcf,bam,reference}

# Download using our script
python scripts/download_genomic_data.py --preset quick-test --output-dir data/downloaded
```

### 2. Quality Check

```bash
# Check FASTQ quality with FastQC
conda install -c bioconda fastqc
fastqc data/downloaded/fastq/*.fastq.gz

# Check file integrity
md5sum -c checksums.md5
```

### 3. Convert Formats if Needed

```bash
# BAM to FASTQ
samtools fastq input.bam > output.fastq

# BAM to FASTQ (paired-end)
samtools fastq -1 read1.fq -2 read2.fq input.bam

# Compress FASTQ
gzip output.fastq
```

### 4. Integrate with GenomeVault

```bash
# Method 1: Direct FASTQ input (with alignment)
python benchmarks/run_alignment_optimized_pipeline.py \
  --format fastq \
  --fastq data/downloaded/fastq/sample_R1.fastq.gz \
  --reference data/downloaded/reference/hg38_chr22.fa

# Method 2: Pre-called VCF input (faster)
python benchmarks/run_alignment_optimized_pipeline.py \
  --format vcf \
  --vcf data/downloaded/vcf/variants.vcf.gz

# Method 3: BAM input
python benchmarks/run_alignment_optimized_pipeline.py \
  --format bam \
  --bam data/downloaded/bam/aligned.bam
```

---

## 📦 Curated Starter Datasets

### Dataset 1: "Quick Test Set" (~500 MB)
**Purpose**: Fast testing, CI/CD
```
- Reference: hg38 chr22 (15 MB)
- Sample: 1000G chr22 exome (150 MB)  
- Variants: 1000G Phase 3 chr22 VCF (300 MB)
- Time to download: ~5 minutes
- Time to process: ~2-3 seconds
```

### Dataset 2: "Gold Standard Set" (~5 GB)
**Purpose**: Benchmarking, validation
```
- Reference: hg38 full genome (900 MB)
- Sample: GIAB NA12878 (2.5 GB)
- Variants: GIAB high-confidence VCF (450 MB)
- Coverage: 30-50x
- Time to download: ~30 minutes
- Time to process: ~10-15 seconds
```

### Dataset 3: "Diversity Set" (~10 GB)
**Purpose**: Testing on diverse populations
```
- Samples from 5 populations:
  - EUR (European): HG00096
  - AFR (African): NA19238
  - EAS (East Asian): NA18525
  - SAS (South Asian): HG03052
  - AMR (American): NA19625
- Each: ~1-2 GB
```

### Dataset 4: "Structural Variant Set" (~20 GB)
**Purpose**: Testing SVs, CNVs, complex variants
```
- HGSVC long-read data
- Includes:
  - Large deletions (>50bp)
  - Insertions
  - Inversions
  - Translocations
  - CNVs
```

---

## 🎓 Tips and Best Practices

### Choosing Data

1. **Start small**: Use chr22 datasets first (~15-50 MB)
2. **Validate with gold standards**: GIAB for benchmarking
3. **Diversify**: Include samples from multiple populations
4. **Match your use case**:
   - Clinical: High coverage (>30x), gold standards
   - Research: Diverse populations, specific variants
   - Development: Small, fast datasets (chr22)

### Download Optimization

```bash
# Use aria2c for parallel downloads
conda install -c conda-forge aria2

# Download with 16 connections
aria2c -x 16 -s 16 <URL>

# Resume interrupted downloads
aria2c --continue <URL>

# Download multiple files
aria2c -i urls.txt
```

### Storage Management

```bash
# Check sizes before downloading
curl -sI <URL> | grep -i Content-Length

# Compress FASTQ files (saves 70-80%)
gzip *.fastq

# Remove duplicates
fdupes -r data/downloaded/

# Monitor disk usage
du -sh data/downloaded/*
```

### Reference Management

Keep multiple reference versions:
```
data/reference/
├── hg38_chr22.fa          # Quick testing
├── hg38_full.fa           # Complete genome
├── grch38_chr22.fa        # Alternative assembly
└── grch37_chr22.fa        # Legacy support
```

---

## 🔗 Useful Resources

### Data Portals
- **1000 Genomes**: https://www.internationalgenome.org/
- **GIAB**: https://www.nist.gov/programs-projects/genome-bottle
- **SRA**: https://www.ncbi.nlm.nih.gov/sra
- **ENA**: https://www.ebi.ac.uk/ena/browser/
- **DDBJ**: https://www.ddbj.nig.ac.jp/
- **gnomAD**: https://gnomad.broadinstitute.org/

### Tools
- **SRA Toolkit**: https://github.com/ncbi/sra-tools
- **SRA Explorer**: https://sra-explorer.info/
- **Entrez Direct**: https://www.ncbi.nlm.nih.gov/books/NBK179288/
- **Aspera Connect**: https://www.ibm.com/products/aspera

### Documentation
- **1000 Genomes Data**: https://www.internationalgenome.org/data
- **GIAB Data**: https://github.com/genome-in-a-bottle/giab_data_indexes
- **SRA Handbook**: https://www.ncbi.nlm.nih.gov/books/NBK47528/

---

## 🐛 Troubleshooting

### Issue: SRA downloads failing

**Solution 1**: Configure SRA Toolkit cache
```bash
vdb-config --interactive
# Set cache location with plenty of space
```

**Solution 2**: Use prefetch first
```bash
prefetch SRR000001  # Download to cache
fasterq-dump SRR000001  # Extract from cache
```

### Issue: Slow downloads

**Solution**: Use Aspera for SRA
```bash
# Install Aspera
conda install -c hcc aspera-connect

# Download with Aspera
prefetch --ascp-path /path/to/ascp SRR000001
```

### Issue: Out of disk space

**Solution**: Stream processing
```bash
# Don't save intermediate files
fasterq-dump SRR000001 --stdout | \
  your_processing_pipeline | \
  gzip > output.fastq.gz
```

### Issue: FASTQ files are huge

**Solution**: Subsample for testing
```bash
# Take every 4th read (25% of data)
seqtk sample -s100 input.fastq.gz 0.25 > subset.fastq

# Or take first N reads
head -n 4000 input.fastq > subset.fastq  # 1000 reads
```

---

## 📝 Example: Complete Download Workflow

```bash
#!/bin/bash
# Complete workflow for setting up GenomeVault test data

# 1. Setup environment
conda create -n genomevault python=3.10
conda activate genomevault
conda install -c bioconda sra-tools samtools aria2 fastqc

# 2. Download reference genome
python scripts/download_genomic_data.py --reference hg38_chr22

# 3. Download test samples
python scripts/download_genomic_data.py --dataset 1kg_hg00096_exome
python scripts/download_genomic_data.py --dataset 1kg_na19238_wgs

# 4. Download variants
python scripts/download_genomic_data.py --vcf 1kg_phase3_chr22

# 5. Quality check
fastqc data/downloaded/fastq/*.fastq.gz

# 6. Run GenomeVault pipeline
python benchmarks/run_alignment_optimized_pipeline.py --preset production

echo "✓ Setup complete! Data ready for GenomeVault testing."
```

---

## 📊 Data Summary Template

Track your downloads:

```yaml
# data/downloaded/MANIFEST.yaml
datasets:
  - name: HG00096_chr22
    source: 1000genomes
    type: BAM
    size_gb: 0.15
    coverage: 60x
    platform: Illumina
    date_downloaded: 2025-10-22
    path: data/downloaded/bam/HG00096.chrom22.bam
    
  - name: hg38_chr22
    source: UCSC
    type: reference
    size_gb: 0.015
    date_downloaded: 2025-10-22
    path: data/downloaded/reference/hg38_chr22.fa
    
  - name: 1kg_phase3_chr22
    source: 1000genomes
    type: VCF
    size_gb: 0.3
    samples: 2504
    variants: ~1M
    date_downloaded: 2025-10-22
    path: data/downloaded/vcf/1kg_phase3_chr22.vcf.gz
```

---

**Need help?** Check the [GenomeVault documentation](../README.md) or open an issue on GitHub.
