# GenomeVault Data Architecture - Correct Terminology

## Three-Layer System

### Layer 1: REFERENCE Genomes
**Purpose**: Public-facing variation superposition strand  
**Location**: `data/reference_genomes/` and `vcf_pool/`  
**Content**: Multiple complete human genome assemblies  
**Use**: Build statistical superposition with 5% uncertainty region  

**Current Reference Genomes**:
- hg38 (GRCh38) - 938 MB - Current standard
- hg19 (GRCh37) - 905 MB - Previous standard  
- CHM13v2.0 (T2T) - 936 MB - Latest complete assembly
- **Total: 2.7 GB of diverse reference assemblies**

### Layer 2: GUIDE Genomes
**Purpose**: Alignment guide for FASTQ read data (k=12 samples)  
**Location**: `data/downloaded/fastq/`  
**Content**: Full FASTQ sequences from diverse ancestry pools  
**Use**: k-anonymity through differential encoding (never directly exposed)

**Current Guide Pool (k=12)**:
- European: 7 samples
- African: 2 samples
- East Asian: 2 samples
- South Asian: 2 samples
- **Total: ~102 GB of FASTQ data**

### Layer 3: EXPERIMENTAL Genomes
**Purpose**: Private genomic data being analyzed  
**Location**: Separate from Layers 1 & 2 (test data)  
**Content**: User's actual query genomes  
**Use**: Encoded differentially against GUIDE pool

**CRITICAL**: Experimental data MUST NOT be mixed with REFERENCE or GUIDE pools to maintain test integrity.

## Data Flow

```
REFERENCE (superposition)
    ↓ (statistical baseline)
GUIDE (k=12 FASTQ samples)
    ↓ (differential encoding)
EXPERIMENTAL (private query)
    ↓ (HDC + ZK proofs)
Privacy-Preserved Output
```

## Terminology Mapping

| Old Term | Correct Term | Purpose |
|----------|--------------|---------|
| "Public data" | REFERENCE | Variation superposition strand |
| "Reference pool" | GUIDE | k-anonymity alignment guide |
| "Experimental" | EXPERIMENTAL | Private query data |

---
**Last Updated**: 2025-10-25
