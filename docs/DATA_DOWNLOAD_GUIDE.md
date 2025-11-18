# GenomeVault Data Download Guide

## Files Created

```
scripts/download_genomic_data.py      # Main download script
scripts/sra_bulk_download.sh          # Bulk SRA downloads
scripts/setup_genomic_data.sh         # Automated setup
data_config.yaml                      # Configuration template
docs/DATA_ACQUISITION_GUIDE.md        # Detailed documentation
```

## Quick Commands

### 1. Download Variant Data (VCF)

```bash
# 1000 Genomes Phase 3, chr22 (~300 MB, ~1M variants)
python scripts/download_genomic_data.py --vcf 1kg_phase3_chr22
# Output: data/downloaded/vcf/1kg_phase3_chr22.vcf.gz

# GIAB high-confidence variants (~450 MB)
python scripts/download_genomic_data.py --vcf giab_na12878_vcf
# Output: data/downloaded/vcf/giab_na12878_vcf.vcf.gz
```

### 2. Download Whole-Genome FASTQ Files

```bash
# Download from SRA (single accession)
./scripts/sra_bulk_download.sh --accession SRR000001
# Output: data/downloaded/fastq/SRR000001_1.fastq.gz
#         data/downloaded/fastq/SRR000001_2.fastq.gz

# Download from SRA (entire project)
./scripts/sra_bulk_download.sh --project PRJNA000001
# Output: data/downloaded/fastq/

# Download from accession list
echo "SRR000001" > accessions.txt
echo "SRR000002" >> accessions.txt
./scripts/sra_bulk_download.sh --from-list accessions.txt
```

### 3. Download Reference Genome

```bash
# chr22 only (15 MB)
python scripts/download_genomic_data.py --reference hg38_chr22
# Output: data/downloaded/reference/hg38_chr22.fa

# Full genome (900 MB)
python scripts/download_genomic_data.py --reference hg38_full
# Output: data/downloaded/reference/hg38_full.fa
```

### 4. Download Pre-curated Samples

```bash
# GIAB gold standard (~2.5 GB FASTQ)
python scripts/download_genomic_data.py --dataset giab_na12878_illumina

# 1000 Genomes samples (BAM format, can convert to FASTQ)
python scripts/download_genomic_data.py --dataset 1kg_hg00096_exome
python scripts/download_genomic_data.py --dataset 1kg_na19238_wgs

# Platinum Genomes (~100 MB FASTQ)
python scripts/download_genomic_data.py --dataset platinum_na12877
```

## Finding SRA Accessions for Whole-Genome Data

### Method 1: SRA Explorer (Easiest)
1. Go to https://sra-explorer.info/
2. Search: "whole genome sequencing human"
3. Filter by coverage (e.g., "30x")
4. Select samples
5. Click "Bash script for downloading FastQ files"
6. Run: `./scripts/sra_bulk_download.sh --from-explorer downloaded_script.sh`

### Method 2: Entrez Direct (Command Line)

```bash
# Install
conda install -c bioconda entrez-direct

# Find WGS datasets
esearch -db sra -query "WGS[Strategy] AND homo sapiens[Organism] AND Illumina[Platform]" | \
  efetch -format runinfo | \
  head -20 > wgs_runs.csv

# Extract accessions
cut -d',' -f1 wgs_runs.csv | tail -n +2 > accessions.txt

# Download
./scripts/sra_bulk_download.sh --from-list accessions.txt
```

### Method 3: Direct SRA Search
1. Go to https://www.ncbi.nlm.nih.gov/sra
2. Search: "whole genome sequencing[Strategy] AND human[Organism]"
3. Apply filters: Coverage, Platform, Date
4. Click on study → "Send to" → "File" → "Accession List"
5. Save as `accessions.txt`
6. Run: `./scripts/sra_bulk_download.sh --from-list accessions.txt`

## Specific Whole-Genome Projects

### High-Coverage WGS
```bash
# 1000 Genomes high-coverage
./scripts/sra_bulk_download.sh --project PRJNA275597

# Human Genome Structural Variation Consortium
./scripts/sra_bulk_download.sh --project PRJNA587799

# Personal Genome Project
./scripts/sra_bulk_download.sh --project PRJNA301331
```

### Population-Specific WGS
```bash
# African populations
esearch -db sra -query "PRJEB31736" | efetch -format runinfo > african_wgs.csv

# Asian populations  
esearch -db sra -query "PRJEB37766" | efetch -format runinfo > asian_wgs.csv
```

## Converting BAM to FASTQ

If you download BAM files instead of FASTQ:

```bash
# Install samtools
conda install -c bioconda samtools

# Single-end
samtools fastq input.bam > output.fastq

# Paired-end
samtools fastq -1 read1.fastq -2 read2.fastq input.bam

# Compress
gzip output.fastq
```

## Output Locations

All downloaded files go to:
- **FASTQ**: `data/downloaded/fastq/`
- **VCF**: `data/downloaded/vcf/`
- **BAM**: `data/downloaded/bam/`
- **Reference**: `data/downloaded/reference/`
- **Metadata**: `data/downloaded/metadata/`

Quick access symlinks:
- `data/current/reference.fa`
- `data/current/variants.vcf.gz`

## Example Workflows

### Get variant data + reference
```bash
python scripts/download_genomic_data.py --reference hg38_chr22
python scripts/download_genomic_data.py --vcf 1kg_phase3_chr22
```

### Get whole-genome FASTQ + reference
```bash
python scripts/download_genomic_data.py --reference hg38_full
./scripts/sra_bulk_download.sh --accession SRR000001
```

### Get everything for testing
```bash
./scripts/setup_genomic_data.sh quick
# Downloads: reference (chr22), variant data, sample BAM
```

## Checking What's Available

```bash
# List all curated datasets
python scripts/download_genomic_data.py --list-datasets

# Check dependencies
python scripts/download_genomic_data.py --check-deps

# View what's been downloaded
ls -lh data/downloaded/*/
```

## Tools Required

### For VCF downloads:
- `wget` or `curl` (built-in on most systems)

### For FASTQ downloads from SRA:
```bash
conda install -c bioconda sra-tools
```

### Optional (faster downloads):
```bash
conda install -c conda-forge aria2
```

## Integration with GenomeVault

Once downloaded, use files with GenomeVault:

```bash
# VCF input
python benchmarks/run_alignment_optimized_pipeline.py \
  --format vcf \
  --vcf data/downloaded/vcf/variants.vcf.gz

# FASTQ input  
python benchmarks/run_alignment_optimized_pipeline.py \
  --format fastq \
  --fastq data/downloaded/fastq/sample_1.fastq.gz \
  --reference data/downloaded/reference/hg38_chr22.fa
```

## Troubleshooting

### SRA downloads fail
```bash
# Configure SRA Toolkit
vdb-config --interactive

# Or use prefetch first
prefetch SRR000001
fasterq-dump SRR000001
```

### Files are huge
```bash
# Subsample (take 25%)
seqtk sample input.fastq.gz 0.25 > subset.fastq
```

### Need faster downloads
```bash
# Install aria2c
conda install -c conda-forge aria2

# Or use Aspera for SRA
conda install -c hcc aspera-connect
```
