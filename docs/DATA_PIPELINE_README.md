# Genomic Data Pipeline - Quick Reference

## 🚀 Ultra-Quick Start (2 commands)

```bash
# 1. Download test data (~500 MB, chr22 only)
./scripts/setup_genomic_data.sh quick

# 2. Run GenomeVault pipeline
python benchmarks/run_alignment_optimized_pipeline.py --preset production
```

**Done!** You now have real genomic data integrated into GenomeVault.

---

## 📚 Complete Documentation

- **Full Guide**: [`docs/DATA_ACQUISITION_GUIDE.md`](docs/DATA_ACQUISITION_GUIDE.md)
- **Main Project Docs**: [`CLAUDE.md`](CLAUDE.md)

---

## 📥 Download Options

### Option 1: One-Command Setup (Recommended)

```bash
# Quick test (~500 MB)
./scripts/setup_genomic_data.sh quick

# Full test with multiple samples (~5 GB)
./scripts/setup_genomic_data.sh full

# Custom (edit data_config.yaml first)
./scripts/setup_genomic_data.sh custom
```

### Option 2: Manual Download

```bash
# List available datasets
python scripts/download_genomic_data.py --list-datasets

# Download specific dataset
python scripts/download_genomic_data.py --dataset giab_na12878_illumina

# Download reference genome
python scripts/download_genomic_data.py --reference hg38_chr22

# Download VCF variants
python scripts/download_genomic_data.py --vcf 1kg_phase3_chr22
```

### Option 3: Bulk SRA Download

```bash
# Single accession
./scripts/sra_bulk_download.sh --accession SRR000001

# From accession list
./scripts/sra_bulk_download.sh --from-list accessions.txt

# Entire project
./scripts/sra_bulk_download.sh --project PRJNA000001

# From SRA Explorer script
./scripts/sra_bulk_download.sh --from-explorer sra_export.sh
```

---

## 🗂️ Data Structure

After download, your data will be organized as:

```
data/
├── downloaded/
│   ├── fastq/          # Raw sequencing reads
│   ├── vcf/            # Variant call files
│   ├── bam/            # Aligned sequences
│   ├── reference/      # Reference genomes
│   └── metadata/       # Dataset information
└── current/            # Symlinks to latest data
    ├── reference.fa
    └── variants.vcf.gz
```

---

## 🎯 Finding Specific Genomic Variations

### SNPs (Single Nucleotide Polymorphisms)

**Available in all datasets** - Most common variation type

```bash
# 1000 Genomes (millions of SNPs)
python scripts/download_genomic_data.py --vcf 1kg_phase3_chr22

# GIAB high-confidence SNPs
python scripts/download_genomic_data.py --vcf giab_na12878_vcf
```

### Indels (Insertions/Deletions)

**Included in VCF files** - Small insertions and deletions

```bash
# Same VCF files contain both SNPs and indels
python scripts/download_genomic_data.py --vcf 1kg_phase3_chr22
```

### Structural Variants (SVs)

**Requires special datasets** - Large rearrangements

```bash
# Human Genome Structural Variation Consortium
./scripts/sra_bulk_download.sh --project PRJNA587799

# 1000 Genomes SVs
./scripts/sra_bulk_download.sh --project PRJEB26711
```

### Copy Number Variants (CNVs)

**Cancer genomics and disease studies**

Search SRA for cancer datasets:
```bash
# Example: TCGA samples
esearch -db sra -query "TCGA[All Fields] AND WGS[Strategy]" | \
  efetch -format runinfo | \
  head -20 > tcga_runs.csv
```

### Mosaic Variants

**Tissue-specific mutations, somatic variants**

```bash
# GTEx project (tissue-specific)
esearch -db sra -query "GTEx[All Fields]" | \
  efetch -format runinfo | \
  head -10 > gtex_samples.txt

# Download from list
./scripts/sra_bulk_download.sh --from-list gtex_samples.txt
```

### Rare Variants

**Population-specific or low-frequency variants**

```bash
# Human Genome Diversity Project
./scripts/sra_bulk_download.sh --project PRJEB31736

# 1000 Genomes rare variants
python scripts/download_genomic_data.py --vcf 1kg_phase3_chr22
# Filter VCF for MAF < 0.01
```

---

## 🔍 Finding Data in SRA

### By Study Type

```bash
# Install Entrez Direct
conda install -c bioconda entrez-direct

# Whole genome sequencing
esearch -db sra -query "WGS[Strategy] AND human[Organism]" | \
  efetch -format runinfo | head -20

# Exome sequencing
esearch -db sra -query "WXS[Strategy] AND human[Organism]" | \
  efetch -format runinfo | head -20

# High coverage (>30x)
esearch -db sra -query "Illumina[Platform] AND 50x[Coverage]" | \
  efetch -format runinfo
```

### By Disease/Phenotype

```bash
# Cancer samples
esearch -db sra -query "cancer[All Fields] AND tumor[All Fields]" | \
  efetch -format runinfo > cancer_samples.csv

# Rare diseases
esearch -db sra -query "rare disease[All Fields]" | \
  efetch -format runinfo

# Pharmacogenomics
esearch -db sra -query "pharmacogenomics[All Fields]" | \
  efetch -format runinfo
```

### By Population

```bash
# African populations
esearch -db sra -query "African[All Fields] AND population[All Fields]" | \
  efetch -format runinfo

# Asian populations
esearch -db sra -query "Asian[All Fields] AND population[All Fields]" | \
  efetch -format runinfo
```

---

## 🧪 Running GenomeVault Pipeline

### With FASTQ Input

```bash
python benchmarks/run_alignment_optimized_pipeline.py \
  --format fastq \
  --fastq data/downloaded/fastq/sample_R1.fastq.gz \
  --reference data/downloaded/reference/hg38_chr22.fa
```

### With VCF Input (Fastest)

```bash
python benchmarks/run_alignment_optimized_pipeline.py \
  --format vcf \
  --vcf data/downloaded/vcf/variants.vcf.gz
```

### With BAM Input

```bash
python benchmarks/run_alignment_optimized_pipeline.py \
  --format bam \
  --bam data/downloaded/bam/aligned.bam
```

### Quick Test

```bash
python benchmarks/run_alignment_optimized_pipeline.py \
  --preset production \
  --quick
```

---

## 📊 Recommended Dataset Combinations

### For Development & Testing

```bash
# Minimum viable dataset (~500 MB)
./scripts/setup_genomic_data.sh quick
```

**Includes:**
- Reference: hg38 chr22 (15 MB)
- Sample: 1000 Genomes chr22 (150 MB)
- Variants: 1000G Phase 3 chr22 VCF (300 MB)

### For Benchmarking

```bash
# Download GIAB gold standard
python scripts/download_genomic_data.py --dataset giab_na12878_illumina
python scripts/download_genomic_data.py --vcf giab_na12878_vcf
```

### For Population Diversity

```bash
# European
python scripts/download_genomic_data.py --dataset 1kg_hg00096_exome

# African
python scripts/download_genomic_data.py --dataset 1kg_na19238_wgs

# Add more as needed
```

### For Structural Variants

```bash
# HGSVC project
./scripts/sra_bulk_download.sh --project PRJNA587799
```

---

## 🛠️ Useful Commands

### Check Available Datasets

```bash
python scripts/download_genomic_data.py --list-datasets
```

### Check Dependencies

```bash
python scripts/download_genomic_data.py --check-deps
```

### View Downloaded Data

```bash
cat data/downloaded/DATA_MANIFEST.md
```

### Convert BAM to FASTQ

```bash
samtools fastq input.bam > output.fastq
```

### Subsample Large Files

```bash
# Take 25% of reads
seqtk sample -s100 input.fastq.gz 0.25 > subset.fastq

# Take first 1000 reads
head -n 4000 input.fastq > subset.fastq
```

### Quality Check

```bash
fastqc data/downloaded/fastq/*.fastq.gz
```

---

## 📈 File Sizes Reference

| Dataset | Type | Size | Time (Fast Connection) |
|---------|------|------|----------------------|
| hg38 chr22 | Reference | ~15 MB | < 1 min |
| hg38 full | Reference | ~900 MB | 5-10 min |
| 1kg chr22 exome | FASTQ/BAM | ~150 MB | 2-5 min |
| 1kg phase3 chr22 VCF | VCF | ~300 MB | 3-8 min |
| GIAB NA12878 | FASTQ | ~2.5 GB | 15-30 min |
| Full WGS 30x | FASTQ | ~100 GB | 1-3 hours |

---

## 🔗 Important Links

### Data Sources
- 1000 Genomes: https://www.internationalgenome.org/
- GIAB: https://www.nist.gov/programs-projects/genome-bottle
- SRA: https://www.ncbi.nlm.nih.gov/sra
- gnomAD: https://gnomad.broadinstitute.org/

### Tools
- SRA Toolkit: https://github.com/ncbi/sra-tools
- SRA Explorer: https://sra-explorer.info/
- Entrez Direct: https://www.ncbi.nlm.nih.gov/books/NBK179288/

### Documentation
- Full Guide: [`docs/DATA_ACQUISITION_GUIDE.md`](docs/DATA_ACQUISITION_GUIDE.md)
- GenomeVault Docs: [`CLAUDE.md`](CLAUDE.md)

---

## 🐛 Troubleshooting

### SRA downloads failing?

```bash
# Configure SRA Toolkit
vdb-config --interactive

# Use prefetch first
prefetch SRR000001
fasterq-dump SRR000001
```

### Out of disk space?

```bash
# Check space
df -h

# Clean up
rm -rf ~/ncbi/public/sra/*.sra

# Subsample data instead of full download
```

### Slow downloads?

```bash
# Install aria2c for parallel downloads
conda install -c conda-forge aria2

# Or use Aspera for SRA
conda install -c hcc aspera-connect
```

---

## 💡 Pro Tips

1. **Start small** - Use chr22 datasets for development
2. **Use VCF when possible** - Faster than FASTQ alignment
3. **Check file sizes** before downloading large datasets
4. **Use SRA Explorer** for bulk downloads
5. **Validate downloads** with checksums
6. **Compress FASTQ files** to save 70-80% space
7. **Use references pools** for GenomeVault k-anonymity

---

## 📝 Example Workflows

### Workflow 1: Quick Test

```bash
# 1. Setup (2 minutes)
./scripts/setup_genomic_data.sh quick

# 2. Run pipeline (3 seconds)
python benchmarks/run_alignment_optimized_pipeline.py --preset production --quick
```

### Workflow 2: Population Study

```bash
# 1. Download diverse samples
python scripts/download_genomic_data.py --dataset 1kg_hg00096_exome  # EUR
python scripts/download_genomic_data.py --dataset 1kg_na19238_wgs    # AFR

# 2. Process each sample
for sample in data/downloaded/bam/*.bam; do
    python benchmarks/run_alignment_optimized_pipeline.py \
        --format bam --bam "$sample"
done
```

### Workflow 3: SRA Study Download

```bash
# 1. Find accessions on SRA Explorer (https://sra-explorer.info/)
# 2. Download bash script
# 3. Run our wrapper
./scripts/sra_bulk_download.sh --from-explorer sra_explorer_export.sh

# 4. Process results
python benchmarks/run_alignment_optimized_pipeline.py --format fastq
```

---

**Questions?** See the [full guide](docs/DATA_ACQUISITION_GUIDE.md) or check [CLAUDE.md](CLAUDE.md)
