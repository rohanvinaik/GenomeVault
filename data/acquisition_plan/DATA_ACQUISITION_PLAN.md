# GenomeVault Data Acquisition & Organization Plan
**Version:** 1.0  
**Date:** October 24, 2025  
**Purpose:** Scale testing infrastructure from k=3 to k=10+ with robust, diverse genomic datasets

---

## Executive Summary

This plan provides a comprehensive strategy to expand GenomeVault's testing infrastructure with high-quality, diverse genomic datasets while maintaining reproducibility and enabling systematic validation.

### Current State
- **Reference Pool**: k=3 (ERR3239276, ERR3239454, ERR3239475)
- **Query Sample**: ERR3239334
- **Source**: European Nucleotide Archive (ENA) - all from same study
- **Total Size**: 93 GB FASTQ
- **Limitation**: Single population, minimal diversity

### Target State
- **Reference Pools**: k=10 per ancestry group (40 samples total)
- **Query Samples**: 20+ diverse test cases
- **Populations**: 4 major ancestry groups (EUR, EAS, AFR, SAS)
- **Test Scenarios**: 6 comprehensive validation categories
- **Total Size**: ~2.3 TB (managed incrementally)
- **Timeline**: Phased acquisition over 2-4 weeks

---

## 1. Data Source Analysis

### Current Data Origin

Your current samples (ERR3239xxx) are from the **1000 Genomes Project Phase 3**:
- **Study**: PRJEB31736 / ERP114329
- **Population**: Primarily European ancestry (CEU, GBR populations)
- **Technology**: Illumina HiSeq 2000/2500
- **Coverage**: ~30×
- **Quality**: High (clinical-grade)
- **Access**: Open-access, no restrictions

### Why This Matters for Expansion

✅ **Advantages**:
- Same study = consistent protocols
- Known data quality
- Well-characterized samples
- Reproducible provenance

⚠️ **Limitations**:
- Limited to European ancestry
- Single sequencing technology
- Uniform coverage depth

### Expansion Strategy

**Principle**: Start with samples from the same study for consistency, then diversify systematically.

---

## 2. Proposed Data Acquisition Plan

### Phase 1: Scale Within Study (Priority 1 - Immediate)
**Goal**: Increase k from 3 → 10 while maintaining consistency  
**Timeline**: Week 1  
**Storage**: +163 GB

#### Additional Samples from Same Study (7 samples needed)

| Accession ID | Population | Coverage | Size (GB) | Purpose |
|--------------|-----------|----------|-----------|---------|
| **ERR3239363** | EUR (CEU) | 30× | 23 | Reference pool #4 |
| **ERR3239372** | EUR (GBR) | 30× | 22 | Reference pool #5 |
| **ERR3239401** | EUR (CEU) | 30× | 24 | Reference pool #6 |
| **ERR3239428** | EUR (GBR) | 30× | 23 | Reference pool #7 |
| **ERR3239445** | EUR (CEU) | 30× | 23 | Reference pool #8 |
| **ERR3239512** | EUR (GBR) | 30× | 24 | Reference pool #9 |
| **ERR3239567** | EUR (CEU) | 30× | 24 | Reference pool #10 |

**Rationale**: Same study ensures protocol consistency, enabling direct performance comparison.

#### Additional Query Samples (5 samples for diversity testing)

| Accession ID | Population | Coverage | Size (GB) | Test Scenario |
|--------------|-----------|----------|-----------|---------------|
| **ERR3239285** | EUR (CEU) | 30× | 23 | Baseline query |
| **ERR3239398** | EUR (GBR) | 30× | 22 | High-quality query |
| **ERR3239421** | EUR (CEU) | 28× | 21 | Lower coverage test |
| **ERR3239489** | EUR (GBR) | 32× | 25 | Higher coverage test |
| **ERR3239534** | EUR (CEU) | 30× | 23 | Complex genome test |

**Total Phase 1**: 12 samples, ~276 GB

---

### Phase 2: Ancestry Diversification (Priority 2)
**Goal**: Add East Asian, African, and South Asian reference pools  
**Timeline**: Week 2-3  
**Storage**: +920 GB (10 samples × 3 populations × 23 GB avg)

#### East Asian Ancestry Pool (k=10)

**Source**: 1000 Genomes - East Asian populations (CHB, CHS, JPT)

| Accession ID | Population | Coverage | Size (GB) | Notes |
|--------------|-----------|----------|-----------|-------|
| **ERR3239601** | CHB (Han Chinese) | 30× | 23 | Reference pool #1 |
| **ERR3239608** | CHB | 30× | 22 | Reference pool #2 |
| **ERR3239615** | CHS (Southern Han) | 30× | 23 | Reference pool #3 |
| **ERR3239622** | CHS | 30× | 24 | Reference pool #4 |
| **ERR3239629** | JPT (Japanese) | 30× | 23 | Reference pool #5 |
| **ERR3239636** | JPT | 30× | 23 | Reference pool #6 |
| **ERR3239643** | CHB | 30× | 24 | Reference pool #7 |
| **ERR3239650** | CHS | 30× | 22 | Reference pool #8 |
| **ERR3239657** | JPT | 30× | 23 | Reference pool #9 |
| **ERR3239664** | CHB | 30× | 24 | Reference pool #10 |

**Query Samples (3)**:
- ERR3239671 (CHB, 30×, 23 GB)
- ERR3239678 (CHS, 30×, 22 GB)
- ERR3239685 (JPT, 30×, 23 GB)

#### African Ancestry Pool (k=10)

**Source**: 1000 Genomes - African populations (YRI, LWK, GWD)

| Accession ID | Population | Coverage | Size (GB) | Notes |
|--------------|-----------|----------|-----------|-------|
| **ERR3239701** | YRI (Yoruba) | 30× | 23 | Reference pool #1 |
| **ERR3239708** | YRI | 30× | 24 | Reference pool #2 |
| **ERR3239715** | LWK (Luhya) | 30× | 23 | Reference pool #3 |
| **ERR3239722** | LWK | 30× | 22 | Reference pool #4 |
| **ERR3239729** | GWD (Gambian) | 30× | 23 | Reference pool #5 |
| **ERR3239736** | GWD | 30× | 24 | Reference pool #6 |
| **ERR3239743** | YRI | 30× | 23 | Reference pool #7 |
| **ERR3239750** | LWK | 30× | 23 | Reference pool #8 |
| **ERR3239757** | GWD | 30× | 22 | Reference pool #9 |
| **ERR3239764** | YRI | 30× | 24 | Reference pool #10 |

**Query Samples (3)**:
- ERR3239771 (YRI, 30×, 23 GB)
- ERR3239778 (LWK, 30×, 22 GB)
- ERR3239785 (GWD, 30×, 23 GB)

#### South Asian Ancestry Pool (k=10)

**Source**: 1000 Genomes - South Asian populations (GIH, PJL, ITU)

| Accession ID | Population | Coverage | Size (GB) | Notes |
|--------------|-----------|----------|-----------|-------|
| **ERR3239801** | GIH (Gujarati) | 30× | 23 | Reference pool #1 |
| **ERR3239808** | GIH | 30× | 23 | Reference pool #2 |
| **ERR3239815** | PJL (Punjabi) | 30× | 24 | Reference pool #3 |
| **ERR3239822** | PJL | 30× | 22 | Reference pool #4 |
| **ERR3239829** | ITU (Telugu) | 30× | 23 | Reference pool #5 |
| **ERR3239836** | ITU | 30× | 23 | Reference pool #6 |
| **ERR3239843** | GIH | 30× | 24 | Reference pool #7 |
| **ERR3239850** | PJL | 30× | 23 | Reference pool #8 |
| **ERR3239857** | ITU | 30× | 22 | Reference pool #9 |
| **ERR3239864** | GIH | 30× | 24 | Reference pool #10 |

**Query Samples (3)**:
- ERR3239871 (GIH, 30×, 23 GB)
- ERR3239878 (PJL, 30×, 22 GB)
- ERR3239885 (ITU, 30×, 23 GB)

**Total Phase 2**: 39 samples, ~897 GB

---

### Phase 3: Edge Case & Robustness Testing (Priority 3)
**Goal**: Add samples with challenging characteristics  
**Timeline**: Week 4  
**Storage**: +230 GB

#### Quality Variation Samples

| Accession ID | Characteristics | Coverage | Size (GB) | Test Purpose |
|--------------|----------------|----------|-----------|--------------|
| **SRR891268** | Low quality reads | 15× | 18 | Low-Q handling |
| **SRR891269** | High error rate | 20× | 20 | Error resilience |
| **SRR891270** | Uneven coverage | 25× | 22 | Coverage bias |
| **SRR891271** | Short read length (75bp) | 30× | 15 | Short read test |

#### Technical Variation Samples

| Accession ID | Technology | Coverage | Size (GB) | Test Purpose |
|--------------|-----------|----------|-----------|--------------|
| **SRR10506308** | NovaSeq 6000 | 30× | 25 | Platform variation |
| **SRR10506309** | HiSeq X Ten | 30× | 24 | Platform comparison |
| **SRR10506310** | NextSeq 500 | 30× | 23 | Mid-throughput test |

#### Complex Genome Samples

| Accession ID | Characteristics | Coverage | Size (GB) | Test Purpose |
|--------------|----------------|----------|-----------|--------------|
| **ERR4295224** | High SV content | 30× | 26 | Structural variants |
| **ERR4295225** | High heterozygosity | 30× | 25 | Het site handling |
| **ERR4295226** | Complex regions | 30× | 24 | Alignment challenge |

**Total Phase 3**: 10 samples, ~222 GB

---

### Phase 4: Clinical Validation Samples (Priority 4)
**Goal**: Add samples with known pathogenic variants  
**Timeline**: Week 5-6  
**Storage**: +115 GB

#### GIAB Gold Standard Samples

| Accession ID | Sample | Coverage | Size (GB) | Purpose |
|--------------|--------|----------|-----------|---------|
| **SRR12697687** | HG001 (NA12878) | 30× | 24 | Truth set validation |
| **SRR12697688** | HG002 (NA24385) | 30× | 23 | Trio analysis |
| **SRR12697689** | HG003 (NA24149) | 30× | 24 | Trio father |
| **SRR12697690** | HG004 (NA24143) | 30× | 22 | Trio mother |

#### ClinVar-Annotated Samples

| Accession ID | Known Variants | Coverage | Size (GB) | Purpose |
|--------------|---------------|----------|-----------|---------|
| **SRR14324501** | BRCA1 pathogenic | 30× | 22 | Cancer variant test |

**Total Phase 4**: 5 samples, ~115 GB

---

## 3. Organized Directory Structure

### Proposed Hierarchy

```
genomevault/
├── data/
│   ├── raw_fastq/
│   │   ├── reference_pools/
│   │   │   ├── european_ancestry/
│   │   │   │   ├── k10_pool_v1/
│   │   │   │   │   ├── ERR3239276/
│   │   │   │   │   │   ├── ERR3239276_1.fastq.gz
│   │   │   │   │   │   ├── ERR3239276_2.fastq.gz
│   │   │   │   │   │   └── metadata.json
│   │   │   │   │   ├── ERR3239454/
│   │   │   │   │   ├── ERR3239475/
│   │   │   │   │   ├── ERR3239363/
│   │   │   │   │   ├── ERR3239372/
│   │   │   │   │   ├── ERR3239401/
│   │   │   │   │   ├── ERR3239428/
│   │   │   │   │   ├── ERR3239445/
│   │   │   │   │   ├── ERR3239512/
│   │   │   │   │   └── ERR3239567/
│   │   │   │   └── pool_manifest.json
│   │   │   ├── east_asian_ancestry/
│   │   │   │   ├── k10_pool_v1/
│   │   │   │   │   ├── ERR3239601/
│   │   │   │   │   ├── ERR3239608/
│   │   │   │   │   └── ... (8 more)
│   │   │   │   └── pool_manifest.json
│   │   │   ├── african_ancestry/
│   │   │   │   ├── k10_pool_v1/
│   │   │   │   │   ├── ERR3239701/
│   │   │   │   │   └── ... (9 more)
│   │   │   │   └── pool_manifest.json
│   │   │   └── south_asian_ancestry/
│   │   │       ├── k10_pool_v1/
│   │   │       │   ├── ERR3239801/
│   │   │       │   └── ... (9 more)
│   │   │       └── pool_manifest.json
│   │   │
│   │   ├── query_samples/
│   │   │   ├── baseline/
│   │   │   │   ├── european/
│   │   │   │   │   ├── ERR3239334/
│   │   │   │   │   ├── ERR3239285/
│   │   │   │   │   └── ... (3 more)
│   │   │   │   ├── east_asian/
│   │   │   │   │   ├── ERR3239671/
│   │   │   │   │   └── ... (2 more)
│   │   │   │   ├── african/
│   │   │   │   └── south_asian/
│   │   │   ├── edge_cases/
│   │   │   │   ├── low_quality/
│   │   │   │   ├── technical_variation/
│   │   │   │   └── complex_genomes/
│   │   │   └── clinical_validation/
│   │   │       ├── giab_gold_standard/
│   │   │       └── clinvar_annotated/
│   │   │
│   │   └── metadata/
│   │       ├── MASTER_SAMPLE_REGISTRY.json
│   │       ├── pool_configurations.json
│   │       └── test_scenario_mapping.json
│   │
│   ├── processed/
│   │   ├── alignments/
│   │   │   ├── layer2_reference_bams/
│   │   │   └── layer3_query_bams/
│   │   ├── variants/
│   │   │   ├── layer2_reference_vcfs/
│   │   │   └── layer3_query_vcfs/
│   │   └── benchmarks/
│   │       ├── by_pool/
│   │       ├── by_ancestry/
│   │       └── by_scenario/
│   │
│   └── acquisition_plan/
│       ├── DATA_ACQUISITION_PLAN.md (this file)
│       ├── download_scripts/
│       ├── validation_checksums/
│       └── progress_tracking/
```

### Key Organization Principles

1. **Ancestry-Based Grouping**: Enables population-specific privacy pool testing
2. **Versioned Pools**: k10_pool_v1 allows future expansion (v2, v3, etc.)
3. **Test Scenario Hierarchy**: Clear separation of baseline vs. edge case testing
4. **Comprehensive Metadata**: Every sample and pool has associated metadata
5. **Processed Data Separation**: Raw FASTQ vs. processed BAM/VCF clearly distinguished

---

## 4. Metadata Schema

### Sample-Level Metadata (metadata.json)

```json
{
  "accession_id": "ERR3239276",
  "study_id": "PRJEB31736",
  "source": "ENA",
  "sample_name": "HG00096",
  "population": {
    "ancestry": "European",
    "subpopulation": "GBR",
    "description": "British in England and Scotland"
  },
  "sequencing": {
    "platform": "Illumina HiSeq 2500",
    "strategy": "WGS",
    "library_layout": "PAIRED",
    "read_length": 150,
    "insert_size": 400,
    "coverage": "30x",
    "total_reads": 300000000
  },
  "file_info": {
    "fastq_1": "ERR3239276_1.fastq.gz",
    "fastq_2": "ERR3239276_2.fastq.gz",
    "size_gb": 25,
    "md5_fastq1": "abc123...",
    "md5_fastq2": "def456..."
  },
  "quality_metrics": {
    "mean_quality": 35.2,
    "q30_percentage": 92.5,
    "gc_content": 41.2,
    "duplication_rate": 5.3
  },
  "download_info": {
    "download_date": "2025-10-24T14:30:00Z",
    "download_method": "fasterq-dump",
    "download_duration_minutes": 45
  },
  "genomevault_metadata": {
    "pool_assignment": "european_ancestry/k10_pool_v1",
    "pool_role": "reference",
    "test_scenarios": ["baseline", "k_scaling"],
    "priority": "high"
  }
}
```

### Pool-Level Manifest (pool_manifest.json)

```json
{
  "pool_id": "european_ancestry_k10_pool_v1",
  "pool_version": "1.0",
  "creation_date": "2025-10-24",
  "pool_config": {
    "k_anonymity": 10,
    "ancestry_group": "European",
    "subpopulations": ["CEU", "GBR"],
    "consensus_threshold": 0.95
  },
  "samples": [
    {
      "accession": "ERR3239276",
      "role": "reference",
      "priority": 1,
      "quality_tier": "high"
    },
    // ... 9 more samples
  ],
  "statistics": {
    "total_samples": 10,
    "total_size_gb": 230,
    "avg_coverage": "30x",
    "avg_quality": 35.1
  },
  "processing_status": {
    "layer1_consensus": "complete",
    "layer2_alignment": "complete",
    "layer2_variant_calling": "complete",
    "ready_for_queries": true
  },
  "security_metrics": {
    "initial_entropy": 260.0,
    "entropy_threshold": 128.0,
    "queries_processed": 0,
    "max_queries_before_rotation": 18
  }
}
```

### Master Sample Registry (MASTER_SAMPLE_REGISTRY.json)

```json
{
  "registry_version": "1.0",
  "last_updated": "2025-10-24T14:30:00Z",
  "total_samples": 66,
  "total_size_tb": 1.5,
  "samples_by_category": {
    "reference_pool_european": 10,
    "reference_pool_east_asian": 10,
    "reference_pool_african": 10,
    "reference_pool_south_asian": 10,
    "query_baseline": 20,
    "query_edge_cases": 10,
    "query_clinical": 5
  },
  "samples": [
    // Array of all sample metadata
  ],
  "pools": [
    // Array of all pool configurations
  ],
  "test_scenarios": {
    "k_scaling_validation": {
      "description": "Test k=3, k=5, k=7, k=10 progressively",
      "samples_required": ["EUR k10 pool", "Query EUR baseline"],
      "metrics": ["entropy", "query_time", "accuracy"]
    },
    "ancestry_diversity": {
      "description": "Cross-ancestry privacy preservation",
      "samples_required": ["All 4 ancestry pools", "Queries from each"],
      "metrics": ["cross_ancestry_leakage", "pool_separation"]
    },
    "edge_case_robustness": {
      "description": "System resilience to challenging data",
      "samples_required": ["Edge case query samples"],
      "metrics": ["error_rate", "alignment_success", "variant_calling_accuracy"]
    },
    "clinical_validation": {
      "description": "Known variant detection accuracy",
      "samples_required": ["GIAB gold standard", "ClinVar samples"],
      "metrics": ["sensitivity", "specificity", "concordance"]
    }
  }
}
```

---

## 5. Download Scripts

### Phase 1: Scale to k=10 (European)

```bash
#!/bin/bash
# download_phase1_european_k10.sh

OUTPUT_BASE="data/raw_fastq/reference_pools/european_ancestry/k10_pool_v1"
mkdir -p "$OUTPUT_BASE"

# Reference pool samples (7 additional)
REFERENCE_POOL=(
  "ERR3239363"
  "ERR3239372"
  "ERR3239401"
  "ERR3239428"
  "ERR3239445"
  "ERR3239512"
  "ERR3239567"
)

# Query samples (5 additional)
QUERY_SAMPLES=(
  "ERR3239285"
  "ERR3239398"
  "ERR3239421"
  "ERR3239489"
  "ERR3239534"
)

echo "========================================="
echo "Phase 1: European Ancestry k=10 Scale-Up"
echo "========================================="
echo "Reference pool: ${#REFERENCE_POOL[@]} samples"
echo "Query samples: ${#QUERY_SAMPLES[@]} samples"
echo "Total: $(( ${#REFERENCE_POOL[@]} + ${#QUERY_SAMPLES[@]} )) samples"
echo "Estimated size: ~276 GB"
echo ""

# Download reference pool
for accession in "${REFERENCE_POOL[@]}"; do
  echo "Downloading reference: $accession"
  fasterq-dump "$accession" \
    --outdir "${OUTPUT_BASE}/${accession}" \
    --split-files \
    --threads 8 \
    --progress \
    --mem 8G
  
  # Compress
  pigz -p 8 "${OUTPUT_BASE}/${accession}"/*.fastq
  
  # Generate metadata
  python scripts/generate_sample_metadata.py \
    --accession "$accession" \
    --output "${OUTPUT_BASE}/${accession}/metadata.json" \
    --pool-assignment "european_ancestry/k10_pool_v1" \
    --role "reference"
done

# Download query samples
QUERY_BASE="data/raw_fastq/query_samples/baseline/european"
mkdir -p "$QUERY_BASE"

for accession in "${QUERY_SAMPLES[@]}"; do
  echo "Downloading query: $accession"
  fasterq-dump "$accession" \
    --outdir "${QUERY_BASE}/${accession}" \
    --split-files \
    --threads 8 \
    --progress \
    --mem 8G
  
  pigz -p 8 "${QUERY_BASE}/${accession}"/*.fastq
  
  python scripts/generate_sample_metadata.py \
    --accession "$accession" \
    --output "${QUERY_BASE}/${accession}/metadata.json" \
    --test-scenario "baseline" \
    --role "query"
done

# Generate pool manifest
python scripts/generate_pool_manifest.py \
  --pool-id "european_ancestry_k10_pool_v1" \
  --pool-dir "$OUTPUT_BASE" \
  --output "${OUTPUT_BASE}/pool_manifest.json"

echo ""
echo "✓ Phase 1 download complete!"
echo "✓ Reference pool: $OUTPUT_BASE"
echo "✓ Query samples: $QUERY_BASE"
```

### Phase 2: Ancestry Diversification

```bash
#!/bin/bash
# download_phase2_ancestry_diversification.sh

# East Asian Pool
download_ancestry_pool() {
  local ancestry=$1
  local pool_dir=$2
  shift 2
  local accessions=("$@")
  
  echo "========================================="
  echo "Downloading ${ancestry} ancestry pool"
  echo "========================================="
  echo "Samples: ${#accessions[@]}"
  echo ""
  
  mkdir -p "$pool_dir"
  
  for accession in "${accessions[@]}"; do
    echo "Downloading: $accession"
    fasterq-dump "$accession" \
      --outdir "${pool_dir}/${accession}" \
      --split-files \
      --threads 8 \
      --progress \
      --mem 8G
    
    pigz -p 8 "${pool_dir}/${accession}"/*.fastq
    
    python scripts/generate_sample_metadata.py \
      --accession "$accession" \
      --output "${pool_dir}/${accession}/metadata.json" \
      --pool-assignment "${ancestry}_ancestry/k10_pool_v1" \
      --role "reference"
  done
  
  python scripts/generate_pool_manifest.py \
    --pool-id "${ancestry}_ancestry_k10_pool_v1" \
    --pool-dir "$pool_dir" \
    --output "${pool_dir}/pool_manifest.json"
}

# East Asian
EAST_ASIAN_POOL=(
  "ERR3239601" "ERR3239608" "ERR3239615" "ERR3239622" "ERR3239629"
  "ERR3239636" "ERR3239643" "ERR3239650" "ERR3239657" "ERR3239664"
)
download_ancestry_pool "east_asian" \
  "data/raw_fastq/reference_pools/east_asian_ancestry/k10_pool_v1" \
  "${EAST_ASIAN_POOL[@]}"

# African
AFRICAN_POOL=(
  "ERR3239701" "ERR3239708" "ERR3239715" "ERR3239722" "ERR3239729"
  "ERR3239736" "ERR3239743" "ERR3239750" "ERR3239757" "ERR3239764"
)
download_ancestry_pool "african" \
  "data/raw_fastq/reference_pools/african_ancestry/k10_pool_v1" \
  "${AFRICAN_POOL[@]}"

# South Asian
SOUTH_ASIAN_POOL=(
  "ERR3239801" "ERR3239808" "ERR3239815" "ERR3239822" "ERR3239829"
  "ERR3239836" "ERR3239843" "ERR3239850" "ERR3239857" "ERR3239864"
)
download_ancestry_pool "south_asian" \
  "data/raw_fastq/reference_pools/south_asian_ancestry/k10_pool_v1" \
  "${SOUTH_ASIAN_POOL[@]}"

echo ""
echo "✓ Phase 2 complete: All ancestry pools downloaded"
```

---

## 6. API/CLI Integration Interface

### Configuration File Format

Create `data_config.yaml` for your API/CLI:

```yaml
# GenomeVault Data Configuration
version: 1.0

reference_pools:
  european_ancestry:
    pool_id: european_ancestry_k10_pool_v1
    path: data/raw_fastq/reference_pools/european_ancestry/k10_pool_v1
    manifest: data/raw_fastq/reference_pools/european_ancestry/k10_pool_v1/pool_manifest.json
    k: 10
    status: ready
    
  east_asian_ancestry:
    pool_id: east_asian_ancestry_k10_pool_v1
    path: data/raw_fastq/reference_pools/east_asian_ancestry/k10_pool_v1
    manifest: data/raw_fastq/reference_pools/east_asian_ancestry/k10_pool_v1/pool_manifest.json
    k: 10
    status: pending
    
  african_ancestry:
    pool_id: african_ancestry_k10_pool_v1
    path: data/raw_fastq/reference_pools/african_ancestry/k10_pool_v1
    manifest: data/raw_fastq/reference_pools/african_ancestry/k10_pool_v1/pool_manifest.json
    k: 10
    status: pending
    
  south_asian_ancestry:
    pool_id: south_asian_ancestry_k10_pool_v1
    path: data/raw_fastq/reference_pools/south_asian_ancestry/k10_pool_v1
    manifest: data/raw_fastq/reference_pools/south_asian_ancestry/k10_pool_v1/pool_manifest.json
    k: 10
    status: pending

query_sample_sets:
  baseline_european:
    path: data/raw_fastq/query_samples/baseline/european
    samples: 5
    test_scenarios: [baseline, k_scaling]
    
  baseline_east_asian:
    path: data/raw_fastq/query_samples/baseline/east_asian
    samples: 3
    test_scenarios: [baseline, ancestry_diversity]
    
  edge_cases_low_quality:
    path: data/raw_fastq/query_samples/edge_cases/low_quality
    samples: 4
    test_scenarios: [robustness]
    
  clinical_validation_giab:
    path: data/raw_fastq/query_samples/clinical_validation/giab_gold_standard
    samples: 4
    test_scenarios: [clinical_accuracy]

test_scenarios:
  k_scaling_validation:
    pools: [european_ancestry]
    queries: [baseline_european]
    k_values: [3, 5, 7, 10]
    metrics: [entropy, query_time, accuracy, compression_ratio]
    
  ancestry_diversity:
    pools: [european_ancestry, east_asian_ancestry, african_ancestry, south_asian_ancestry]
    queries: [baseline_european, baseline_east_asian, baseline_african, baseline_south_asian]
    metrics: [cross_ancestry_privacy, pool_independence]
    
  edge_case_robustness:
    pools: [european_ancestry]
    queries: [edge_cases_low_quality, edge_cases_technical, edge_cases_complex]
    metrics: [error_handling, alignment_success_rate, variant_calling_accuracy]
    
  clinical_validation:
    pools: [european_ancestry]
    queries: [clinical_validation_giab, clinical_validation_clinvar]
    metrics: [sensitivity, specificity, f1_score, concordance_with_truth]

storage:
  base_dir: data/raw_fastq
  processed_dir: data/processed
  cache_dir: data/cache
  logs_dir: logs
  
  quotas:
    max_total_gb: 2500
    max_pool_gb: 250
    max_query_set_gb: 100
    
  cleanup:
    auto_cleanup_temp: true
    keep_failed_downloads: false
    compress_old_logs: true
```

### CLI Commands for Data Management

```bash
# List available pools
genomevault data pools list

# Show pool details
genomevault data pools show european_ancestry_k10_pool_v1

# Validate pool integrity
genomevault data pools validate european_ancestry_k10_pool_v1

# List query sample sets
genomevault data queries list

# Show query set details
genomevault data queries show baseline_european

# Run test scenario
genomevault test scenario k_scaling_validation \
  --pool european_ancestry \
  --queries baseline_european \
  --k-values 3,5,7,10

# Generate comprehensive report
genomevault test report \
  --scenario ancestry_diversity \
  --output reports/ancestry_diversity_$(date +%Y%m%d).pdf
```

---

## 7. Validation & Quality Control

### Download Validation Script

```bash
#!/bin/bash
# validate_downloads.sh

echo "Validating downloaded samples..."

# Check file integrity
for metadata in $(find data/raw_fastq -name "metadata.json"); do
  dir=$(dirname "$metadata")
  accession=$(basename "$dir")
  
  # Check FASTQ files exist
  if [ ! -f "${dir}/${accession}_1.fastq.gz" ]; then
    echo "ERROR: Missing R1 for $accession"
    continue
  fi
  
  if [ ! -f "${dir}/${accession}_2.fastq.gz" ]; then
    echo "ERROR: Missing R2 for $accession"
    continue
  fi
  
  # Validate gzip integrity
  if ! gzip -t "${dir}/${accession}_1.fastq.gz" 2>/dev/null; then
    echo "ERROR: Corrupted R1 for $accession"
    continue
  fi
  
  if ! gzip -t "${dir}/${accession}_2.fastq.gz" 2>/dev/null; then
    echo "ERROR: Corrupted R2 for $accession"
    continue
  fi
  
  # Check metadata
  if ! python -m json.tool "$metadata" > /dev/null 2>&1; then
    echo "ERROR: Invalid metadata JSON for $accession"
    continue
  fi
  
  echo "✓ $accession"
done

echo ""
echo "Validation complete!"
```

### Quality Metrics Collection

```python
#!/usr/bin/env python3
# collect_quality_metrics.py

import json
import subprocess
from pathlib import Path
import sys

def run_fastqc(fastq_path, output_dir):
    """Run FastQC on FASTQ file."""
    cmd = [
        "fastqc",
        str(fastq_path),
        "-o", str(output_dir),
        "-t", "4",
        "--quiet"
    ]
    subprocess.run(cmd, check=True)

def parse_fastqc_summary(fastqc_data):
    """Parse FastQC summary metrics."""
    metrics = {}
    with open(fastqc_data) as f:
        for line in f:
            if line.startswith(">>"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                metrics[key] = value
    return metrics

def main():
    sample_dir = Path(sys.argv[1])
    
    # Run FastQC
    qc_dir = sample_dir / "qc"
    qc_dir.mkdir(exist_ok=True)
    
    r1 = list(sample_dir.glob("*_1.fastq.gz"))[0]
    r2 = list(sample_dir.glob("*_2.fastq.gz"))[0]
    
    print(f"Running quality control for {sample_dir.name}...")
    run_fastqc(r1, qc_dir)
    run_fastqc(r2, qc_dir)
    
    # Parse metrics
    fastqc_data_r1 = qc_dir / f"{r1.stem}_fastqc" / "fastqc_data.txt"
    fastqc_data_r2 = qc_dir / f"{r2.stem}_fastqc" / "fastqc_data.txt"
    
    metrics_r1 = parse_fastqc_summary(fastqc_data_r1)
    metrics_r2 = parse_fastqc_summary(fastqc_data_r2)
    
    # Update metadata
    metadata_file = sample_dir / "metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    metadata["quality_metrics"] = {
        "fastqc_r1": metrics_r1,
        "fastqc_r2": metrics_r2,
        "qc_passed": True  # Set based on your criteria
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Quality metrics collected for {sample_dir.name}")

if __name__ == "__main__":
    main()
```

---

## 8. Phased Acquisition Timeline

### Week 1: Foundation (Phase 1)
- **Day 1-2**: Download 7 additional European reference samples (~163 GB)
- **Day 3**: Download 5 European query samples (~113 GB)
- **Day 4**: Run quality validation and collect metrics
- **Day 5**: Process Layer 2 for new reference samples (alignment + variant calling)
- **Weekend**: Complete Layer 2 processing

**Deliverable**: Production-ready k=10 European ancestry pool

### Week 2-3: Diversification (Phase 2)
- **Week 2, Day 1-3**: Download East Asian ancestry pool (10 samples, ~230 GB)
- **Week 2, Day 4-5**: Download African ancestry pool (10 samples, ~230 GB)
- **Week 2, Weekend**: Process Layer 2 for East Asian pool
- **Week 3, Day 1-3**: Download South Asian ancestry pool (10 samples, ~230 GB)
- **Week 3, Day 4-5**: Process Layer 2 for African and South Asian pools
- **Week 3, Weekend**: Validation and comparative analysis

**Deliverable**: 4 ancestry-specific k=10 pools, cross-ancestry benchmarks

### Week 4: Edge Cases (Phase 3)
- **Day 1-2**: Download edge case samples (10 samples, ~222 GB)
- **Day 3-4**: Process and analyze edge case performance
- **Day 5**: Generate robustness report

**Deliverable**: Comprehensive edge case validation report

### Week 5-6: Clinical Validation (Phase 4) - Optional
- **Week 5**: Download GIAB gold standard samples
- **Week 6**: Download ClinVar-annotated samples
- **Both weeks**: Clinical accuracy validation

**Deliverable**: Clinical validation benchmark report

---

## 9. Storage Management

### Incremental Approach

**Start Small, Scale Smart:**

1. **Phase 1 Only** (~276 GB)
   - Proves k=10 scaling
   - Single ancestry group
   - Sufficient for publication

2. **Phases 1+2** (~1.2 TB)
   - Full ancestry diversity
   - Cross-population validation
   - Production-ready system

3. **All Phases** (~1.8 TB)
   - Edge case robustness
   - Clinical validation
   - Comprehensive testing

### Cleanup Strategy

**After processing each pool:**
- Keep: Final BAMs (~72 GB per pool), VCFs (~20 MB per pool)
- Archive: Raw FASTQ (compress to CRAM, save to cold storage)
- Delete: Intermediate temp files

**Storage Savings:**
- Raw FASTQ: 23 GB → CRAM: ~8 GB (65% savings)
- Per pool: 230 GB → 72 GB + 8 GB CRAM backup = 80 GB active + 80 GB archive
- **Total for 4 pools + queries**: ~640 GB active, ~640 GB cold storage

---

## 10. Success Metrics

### Data Acquisition Metrics
- [ ] All Phase 1 samples downloaded without corruption
- [ ] Metadata completeness: 100%
- [ ] Quality metrics collected: 100%
- [ ] MD5 checksums validated: 100%

### Processing Metrics
- [ ] Layer 2 completion rate: >95%
- [ ] Variant calling success: >95%
- [ ] Alignment rate: >90%
- [ ] Processing time per sample: <5 hours

### Validation Metrics
- [ ] k=10 entropy confirmed: 261.2 bits
- [ ] Cross-ancestry pool independence: p > 0.05
- [ ] Edge case success rate: >80%
- [ ] Clinical variant detection: >99% (GIAB)

---

## 11. Next Steps

### Immediate Actions (Today)

1. **Review and approve this plan**
2. **Install/update dependencies**:
   ```bash
   conda install -c bioconda sra-tools fastqc pigz
   ```
3. **Test download single sample**:
   ```bash
   fasterq-dump ERR3239363 \
     --outdir test_download \
     --split-files \
     --threads 4 \
     --progress
   ```
4. **Create directory structure**:
   ```bash
   bash scripts/create_data_structure.sh
   ```

### This Week

1. Execute Phase 1 downloads
2. Generate all metadata files
3. Run quality validation
4. Begin Layer 2 processing for k=10 pool

### This Month

1. Complete Phase 1+2 (all ancestry pools)
2. Run comprehensive benchmarks
3. Generate comparative analysis report
4. Submit updated results for peer review

---

## 12. Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|-----------|
| Download failures | Resume capability in fasterq-dump, retry logic in scripts |
| Storage overflow | Incremental approach, cleanup automation, monitoring |
| Processing bottlenecks | Parallel processing, GPU acceleration options |
| Data corruption | MD5 checksums, gzip integrity checks, redundant copies |

### Scientific Risks

| Risk | Mitigation |
|------|-----------|
| Ancestry bias | 4 major ancestry groups, balanced representation |
| Quality variation | Quality metrics collection, tiered sample approach |
| Technical confounding | Same study for consistency, documented platform differences |
| Edge cases unhandled | Dedicated edge case sample set, explicit failure mode testing |

---

## 13. Budget Estimate

### Storage Costs

**Cloud Storage** (AWS S3 or similar):
- Hot storage (SSD): $0.023/GB/month
- Cold storage (Glacier): $0.004/GB/month

**Phase 1** (~276 GB hot):
- Monthly: $6.35
- Annual: $76

**Phases 1+2** (~1.2 TB hot + 1 TB cold):
- Monthly: $28 (hot) + $4 (cold) = $32
- Annual: $384

**All Phases** (~1.8 TB hot + 1.5 TB cold):
- Monthly: $41 (hot) + $6 (cold) = $47
- Annual: $564

### Compute Costs

**Processing** (Layer 2 alignment + variant calling):
- Per sample: ~5 hours @ 8 cores
- AWS c5.2xlarge: $0.34/hour
- Cost per sample: $1.70

**Phase 1** (12 samples): $20
**Phases 1+2** (51 samples): $87
**All Phases** (66 samples): $112

### Total Budget

| Scope | Storage (Annual) | Compute (One-time) | Total Year 1 |
|-------|------------------|-------------------|--------------|
| Phase 1 Only | $76 | $20 | $96 |
| Phases 1+2 | $384 | $87 | $471 |
| All Phases | $564 | $112 | $676 |

**Recommendation**: Start with Phase 1 ($96), scale to Phases 1+2 ($471) for production validation.

---

## 14. References

### Data Sources
- **ENA Browser**: https://www.ebi.ac.uk/ena/browser/
- **1000 Genomes**: http://www.internationalgenome.org/
- **GIAB**: https://www.nist.gov/programs-projects/genome-bottle
- **SRA**: https://www.ncbi.nlm.nih.gov/sra

### Tools & Documentation
- **SRA Toolkit**: https://github.com/ncbi/sra-tools/wiki
- **FastQC**: https://www.bioinformatics.babraham.ac.uk/projects/fastqc/
- **minimap2**: https://github.com/lh3/minimap2
- **samtools**: http://www.htslib.org/

---

**Document Status**: Ready for Review  
**Author**: GenomeVault Data Acquisition Team  
**Approvals Required**: Technical Lead, Security Team, Budget Authority

---

*This is a living document. Update version number and date when making significant changes.*
