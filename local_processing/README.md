# GenomeVault Local Processing

This module provides comprehensive local processing capabilities for multi-omics data, ensuring that sensitive biological information never leaves the user's device in raw form.

## Overview

The local processing engine handles five major types of biological data:

1. **Genomic Sequencing** - DNA/RNA sequence analysis
2. **Transcriptomics** - Gene expression profiling
3. **Epigenetics** - DNA methylation and chromatin accessibility
4. **Proteomics** - Protein abundance and modifications
5. **Phenotypes** - Clinical data and health records

## Key Features

- **Privacy-First**: All processing happens locally on user's device
- **Multi-Format Support**: Handles standard bioinformatics file formats
- **Quality Control**: Built-in QC metrics and validation
- **Differential Storage**: Efficient compression using reference-based encoding
- **Container Isolation**: Secure processing in isolated environments

## Module Structure

```
local_processing/
├── sequencing.py       # Genomic variant calling and analysis
├── transcriptomics.py  # RNA-seq expression quantification
├── epigenetics.py      # Methylation and ATAC-seq processing
├── proteomics.py       # Mass spectrometry data analysis
└── phenotypes.py       # Clinical data processing (FHIR/EHR)
```

## Usage Examples

### Processing Genomic Data

```python
from genomevault.local_processing import SequencingProcessor, DifferentialStorage

# Initialize processor
processor = SequencingProcessor()

# Process sequencing data
profile = processor.process(
    input_path=Path("sample.fastq.gz"),
    sample_id="patient_001"
)

# Compress using differential storage
storage = DifferentialStorage()
compressed = storage.compress_profile(profile)

print(f"Found {len(profile.variants)} variants")
print(f"Compression ratio: {len(compressed) / original_size:.2%}")
```

### Processing RNA-seq Data

```python
from genomevault.local_processing import TranscriptomicsProcessor

# Initialize processor
processor = TranscriptomicsProcessor()

# Process RNA-seq data
expression_profile = processor.process(
    input_path=Path("sample_R1.fastq.gz"),
    sample_id="patient_001",
    paired_end=True
)

# Get expressed genes
expressed_genes = expression_profile.filter_by_expression(min_tpm=1.0)
print(f"Found {len(expressed_genes)} expressed genes")

# Batch effect correction for multiple samples
corrected_profiles = processor.batch_effect_correction(
    profiles=[profile1, profile2, profile3],
    batch_labels=['batch1', 'batch1', 'batch2']
)
```

### Processing Methylation Data

```python
from genomevault.local_processing import MethylationProcessor

# Initialize processor
processor = MethylationProcessor()

# Process WGBS data
methylation_profile = processor.process(
    input_path=Path("sample_bisulfite.fastq.gz"),
    sample_id="patient_001"
)

# Get methylation summary
metrics = methylation_profile.quality_metrics
print(f"Mean methylation: {metrics['mean_methylation']:.2%}")
print(f"CpG sites analyzed: {metrics['total_cpg_sites']:,}")
```

### Processing Clinical Data

```python
from genomevault.local_processing import PhenotypeProcessor

# Initialize processor
processor = PhenotypeProcessor()

# Process FHIR bundle
phenotype_profile = processor.process(
    input_data=fhir_bundle,
    sample_id="patient_001",
    data_format="fhir"
)

# Calculate risk factors
risk_factors = phenotype_profile.calculate_risk_factors()
print(f"Risk factors: {risk_factors}")

# Get active conditions
active_conditions = phenotype_profile.get_active_conditions()
for condition in active_conditions:
    print(f"- {condition.name} (ICD-10: {condition.icd10_code})")
```

## Supported File Formats

### Genomics
- FASTQ (.fastq, .fq, .fastq.gz, .fq.gz)
- BAM/SAM (.bam, .sam)
- CRAM (.cram)
- VCF (.vcf, .vcf.gz)

### Transcriptomics
- FASTQ (single/paired-end)
- Expression matrices (TSV/CSV)
- Kallisto/STAR outputs

### Epigenetics
- Bisulfite sequencing (FASTQ)
- BedGraph methylation files
- ATAC-seq (FASTQ/BAM)
- Peak files (narrowPeak/broadPeak)

### Proteomics
- mzML/mzXML (mass spec raw data)
- MaxQuant outputs (proteinGroups.txt)
- MGF (Mascot Generic Format)

### Phenotypes
- FHIR bundles (JSON)
- CSV clinical data
- Custom JSON formats

## Quality Control

Each processor includes comprehensive quality control:

### Genomics QC
- Read quality metrics (Q20/Q30 bases)
- Coverage uniformity
- Duplicate rate
- Mapping quality

### Transcriptomics QC
- Mapping rate
- Gene detection rate
- Expression distribution
- Batch effects

### Epigenetics QC
- Bisulfite conversion rate
- Coverage depth
- Peak calling statistics
- Fragment size distribution

### Proteomics QC
- Peptide identification rate
- Protein sequence coverage
- Dynamic range
- Modification detection

## Performance Optimization

The processors are optimized for:

- **Multi-threading**: Utilize all available CPU cores
- **Memory efficiency**: Stream processing for large files
- **GPU acceleration**: Optional CUDA support for intensive operations
- **Compression**: Efficient storage of processed results

## Security Features

- **Container isolation**: Processing in secure containers
- **No network access**: Air-gapped processing
- **Encrypted temp files**: Secure handling of intermediate data
- **Audit logging**: Complete processing trail

## Configuration

Processing can be configured via the global config:

```python
from genomevault.utils import get_config

config = get_config()

# Set processing parameters
config.processing.max_cores = 8
config.processing.max_memory_gb = 32
config.processing.min_quality_score = 30
config.processing.min_coverage = 30
```

## Dependencies

The local processing module requires these bioinformatics tools:

### Required
- Python 3.9+
- NumPy, Pandas, SciPy
- BioPython
- pysam

### Optional (for full functionality)
- BWA/BWA-MEM2 (genomic alignment)
- GATK4 (variant calling)
- STAR/Kallisto (RNA-seq)
- Bismark (methylation)
- MACS2 (peak calling)
- samtools, bcftools, bedtools

## Troubleshooting

### Common Issues

1. **Missing tools**: Some processors require external bioinformatics tools. Check logs for warnings about missing dependencies.

2. **Memory errors**: Large datasets may require increasing memory limits:
   ```python
   config.processing.max_memory_gb = 64
   ```

3. **Slow processing**: Enable multi-threading:
   ```python
   processor = SequencingProcessor(max_threads=16)
   ```

4. **File format errors**: Ensure input files are properly formatted and not corrupted.

## Contributing

When adding new processors:

1. Follow the existing pattern of Processor classes
2. Include comprehensive quality metrics
3. Add appropriate file format validation
4. Implement differential/compressed storage where applicable
5. Include unit tests with example data

## License

This module is part of GenomeVault 3.0 and is licensed under the Apache License 2.0.
