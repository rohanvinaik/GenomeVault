# Clinical SNP Reference Library - Quick Start Guide

## Overview

This implementation adds a queryable clinical SNP reference database to GenomeVault, enabling identification of clinically-relevant variants while maintaining privacy guarantees.

## 🚀 Quick Start (5 minutes)

```bash
# 1. Build the clinical database (~15 min download + 5 min processing)
python -m genomevault.clinical_db.data_acquisition \
    --genome-build GRCh38 \
    --output-dir data \
    --pathogenic-only \
    --min-stars 1

# 2. Query a specific variant
python -m genomevault.cli.clinical_query query-position \
    --chr chr11 \
    --pos 5227002

# 3. Query a gene
python -m genomevault.cli.clinical_query query-gene BRCA1

# 4. Analyze a VCF file
python -m genomevault.cli.clinical_query analyze-vcf patient.vcf -o report.txt
```

---

## 📦 Installation

### 1. Install Dependencies

```bash
# Core dependencies
pip install requests click tabulate

# If using API
pip install fastapi uvicorn

# Optional: for VCF parsing
conda install -c bioconda bcftools
```

### 2. Add Clinical DB Module

```bash
# Create directory structure
mkdir -p genomevault/clinical_db
mkdir -p data/raw

# Copy implementation files
# - database.py
# - data_acquisition.py
# Copy API endpoint file to genomevault/api/endpoints/
# Copy CLI tool to genomevault/cli/
```

---

## 🗄️ Building the Database

### Option 1: Full ClinVar Database (~2GB download, ~150K variants)

```bash
python -m genomevault.clinical_db.data_acquisition \
    --genome-build GRCh38 \
    --output-dir data
```

**Expected output:**
```
data/
├── raw/
│   └── clinvar_GRCh38.vcf.gz         (~2GB)
└── clinical_snps_v1.0.0.json.gz      (~50MB)
```

### Option 2: Pathogenic-Only Database (Recommended for testing)

```bash
python -m genomevault.clinical_db.data_acquisition \
    --genome-build GRCh38 \
    --output-dir data \
    --pathogenic-only \
    --min-stars 2
```

**Expected output:**
```
~10,000-20,000 high-confidence pathogenic variants
Database size: ~5-10MB
Processing time: ~2-3 minutes
```

### Option 3: Download-Only (Manual Build)

```bash
# Just download the data
python -m genomevault.clinical_db.data_acquisition \
    --download-only \
    --genome-build GRCh38

# Build database later with custom filters
python -m genomevault.clinical_db.data_acquisition \
    --build-only \
    --pathogenic-only \
    --min-stars 3
```

---

## 💻 Usage Examples

### Command-Line Interface

#### 1. Database Statistics

```bash
python -m genomevault.cli.clinical_query stats
```

Output:
```
==============================================================
CLINICAL SNP DATABASE STATISTICS
==============================================================
total_snps                    : 15234
pathogenic_count              : 12450
pharmaco_count                : 2890
genes_covered                 : 4523
conditions_covered            : 8934
genome_build                  : GRCh38
version                       : 1.0.0
build_date                    : 2025-10-24
==============================================================
```

#### 2. Query Specific Position (e.g., Sickle Cell)

```bash
python -m genomevault.cli.clinical_query query-position \
    --chr chr11 \
    --pos 5227002 \
    --format detailed
```

Output:
```
============================================================
VARIANT 1/1
============================================================
SNP ID:                 rs334
Position:               chr11:5227002
Gene:                   HBB
Alleles:                A → T
Clinical Significance:  pathogenic

Conditions:
  • Sickle Cell Anemia
    OMIM: 603903
    Inheritance: autosomal_recessive

Review Status:          practice_guideline
Stars:                  ⭐⭐⭐⭐

Functional Impact:
  Consequence:          missense_variant
  Protein Change:       p.Glu6Val
```

#### 3. Query Gene (e.g., BRCA1)

```bash
python -m genomevault.cli.clinical_query query-gene BRCA1 --format summary
```

Output:
```
============================================================
CLINICAL VARIANTS IN BRCA1
============================================================
Total variants:         1234
Pathogenic:             456
Benign:                 234
VUS:                    544

Chromosome:             chr17

Top Pathogenic Variants:
  • rs80357906 (chr17:43094464) ⭐⭐⭐⭐
    Breast-Ovarian Cancer, Familial 1
  • rs80357914 (chr17:43095859) ⭐⭐⭐
    Hereditary Breast Cancer
```

#### 4. Analyze VCF File

```bash
python -m genomevault.cli.clinical_query analyze-vcf \
    patient.vcf \
    --output report.txt \
    --detailed
```

Output:
```
================================================================================
CLINICAL VARIANT ANALYSIS REPORT
================================================================================
VCF File:               patient.vcf
Database:               data/clinical_snps_v1.0.0.json.gz

SUMMARY
--------------------------------------------------------------------------------
Total variants:         45,892
Clinical hits:          127 (0.3%)
Pathogenic:             3
Pharmacogenomic:        12

DETAILED RESULTS
--------------------------------------------------------------------------------

1. chr17:43094464 C>T
   • BRCA1: pathogenic
     Condition: Breast-Ovarian Cancer, Familial 1

2. chr11:5227002 A>T
   • HBB: pathogenic
     Condition: Sickle Cell Anemia
```

### Python API

```python
from genomevault.clinical_db.database import ClinicalSNPDatabase

# Load database
db = ClinicalSNPDatabase('data/clinical_snps_v1.0.0.json.gz')

# Query position
snps = db.query_position('chr11', 5227002)
for snp in snps:
    print(f"{snp.snp_id}: {snp.gene} - {snp.clinical_significance}")

# Query gene
brca1_variants = db.query_gene('BRCA1')
pathogenic = [v for v in brca1_variants if v.is_pathogenic()]
print(f"BRCA1 has {len(pathogenic)} pathogenic variants")

# Analyze VCF
results = db.analyze_vcf_file('patient.vcf')
print(f"Found {results['clinical_hits']} clinical variants")
print(f"Pathogenic: {results['pathogenic_count']}")
```

---

## 🌐 REST API

### 1. Setup API

Add to `genomevault/api/app.py`:

```python
from fastapi import FastAPI
from genomevault.api.endpoints import clinical_query

app = FastAPI(title="GenomeVault API")

# Register clinical query endpoints
app.include_router(clinical_query.router)

# Load clinical database at startup
@app.on_event("startup")
async def startup_event():
    clinical_query.init_clinical_database("data/clinical_snps_v1.0.0.json.gz")
```

### 2. Start Server

```bash
uvicorn genomevault.api.app:app --reload --port 8000
```

### 3. API Endpoints

#### Query Position

```bash
curl -X POST "http://localhost:8000/api/v1/clinical/query/positions" \
     -H "Content-Type: application/json" \
     -d '{
       "chromosome": "chr11",
       "positions": [5227002],
       "ref_alleles": ["A"],
       "alt_alleles": ["T"]
     }'
```

#### Query Gene

```bash
curl "http://localhost:8000/api/v1/clinical/query/gene/BRCA1"
```

#### Get Pathogenic Variants

```bash
curl "http://localhost:8000/api/v1/clinical/pathogenic?limit=50"
```

#### Analyze VCF

```bash
curl -X POST "http://localhost:8000/api/v1/clinical/analyze/vcf" \
     -F "vcf_file=@patient.vcf"
```

#### Full Pipeline with Clinical Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/clinical/analyze/genome-with-clinical" \
     -F "query_vcf=@patient.vcf" \
     -F "include_clinical=true"
```

---

## 🔒 Privacy-Preserving Integration

### The Challenge

Direct clinical queries could leak information about the query genome:

```
❌ BAD: Query VCF → Clinical DB (Direct linkage!)
```

### The Solution

Always use the reference pool for k-anonymity:

```
✅ GOOD: Query VCF → Reference Pool → Differential Encoding → Clinical DB
```

### Implementation

```python
from genomevault.differential_encoding import DifferentialEncoder
from genomevault.clinical_db.database import ClinicalSNPDatabase

# Step 1: Privacy-preserving encoding
encoder = DifferentialEncoder(k_anonymity=3)
encoded = encoder.encode(
    query_vcf='patient.vcf',
    reference_pool=['ref1.vcf', 'ref2.vcf', 'ref3.vcf']
)

# Step 2: Query clinical database with ENCODED variants
db = ClinicalSNPDatabase('data/clinical_snps_v1.0.0.json.gz')
clinical_hits = []

for encoded_var in encoded.variants:
    # Query uses anonymized position, not original
    hits = db.query_position(
        encoded_var.chromosome,
        encoded_var.position  # This is k-anonymous!
    )
    clinical_hits.append(hits)

# Privacy guarantee: Clinical DB never sees raw query
```

---

## 📊 Database Contents

### ClinVar

**Source:** NCBI ClinVar (https://www.ncbi.nlm.nih.gov/clinvar/)

**Included Variants:**
- All pathogenic/likely pathogenic variants
- Minimum 1-star review status
- Associated with genetic conditions
- Includes inheritance patterns, penetrance

**Example Conditions:**
- Hereditary Cancer Syndromes (BRCA1/2, Lynch syndrome)
- Cardiovascular Conditions (Long QT, Cardiomyopathies)
- Metabolic Disorders (PKU, Tay-Sachs)
- Neurodevelopmental Disorders
- Blood Disorders (Sickle Cell, Thalassemia)

### PharmGKB (Future)

**Source:** PharmGKB (https://www.pharmgkb.org/)

**To Include:**
- Level 1A/1B/2A pharmacogenomic associations
- Drug-gene interactions
- Dosing guidelines (CPIC)
- Drug response phenotypes

**Example Drugs:**
- Warfarin (CYP2C9, VKORC1)
- Clopidogrel (CYP2C19)
- Statins (SLCO1B1)
- Codeine (CYP2D6)

### GWAS Catalog (Future)

**Source:** GWAS Catalog (https://www.ebi.ac.uk/gwas/)

**To Include:**
- Genome-wide significant associations (P < 5e-8)
- Common complex diseases
- Replicated findings only

**Example Traits:**
- Type 2 Diabetes
- Coronary Artery Disease
- Alzheimer's Disease
- Height, BMI

---

## 🔧 Customization

### Filter Variants

```python
from genomevault.clinical_db.data_acquisition import ClinVarDownloader

downloader = ClinVarDownloader()
vcf_path = downloader.download_vcf()

# Custom filtering
snps = downloader.parse_vcf(
    vcf_path,
    filter_pathogenic=True,        # Only pathogenic
    min_review_stars=3             # Only 3+ star reviews
)

# Additional filtering
snps_filtered = [
    s for s in snps
    if s.gene in ['BRCA1', 'BRCA2', 'TP53']  # Cancer genes only
    and any('cancer' in c.name.lower() for c in s.conditions)
]
```

### Build Custom Database

```python
from genomevault.clinical_db.database import ClinicalDatabaseBuilder, ClinicalSNP

builder = ClinicalDatabaseBuilder(genome_build="GRCh38")

# Add custom variant
custom_variant = ClinicalSNP(
    snp_id='rs123456',
    chromosome='chr1',
    position=1000000,
    ref_allele='A',
    alt_alleles=['G'],
    gene='MYGENE',
    clinical_significance='pathogenic',
    conditions=[...],
    sources={'internal_db': 'VAR001'}
)

builder.add_snp(custom_variant)
builder.save('custom_clinical_db.json.gz')
```

---

## 📈 Performance

### Database Size

| Configuration | SNP Count | Compressed Size | Load Time | Query Time |
|--------------|-----------|----------------|-----------|------------|
| Full ClinVar | ~150,000 | ~50 MB | ~2-3 sec | <1 ms |
| Pathogenic Only (1⭐+) | ~45,000 | ~15 MB | ~1 sec | <1 ms |
| Pathogenic Only (2⭐+) | ~25,000 | ~8 MB | <1 sec | <1 ms |
| High Confidence (3⭐+) | ~10,000 | ~3 MB | <500 ms | <1 ms |

### Query Performance

- **Position Query:** O(1) with hash index, <1ms
- **Gene Query:** O(1) with hash index, <1ms  
- **Region Query:** O(n) where n = region size, <10ms for 1Mb
- **VCF Analysis:** ~1,000 variants/second

---

## 🚀 Next Steps

### Phase 1: Basic Implementation (Week 1-2)
- ✅ Core database structure
- ✅ ClinVar data acquisition
- ✅ CLI tools
- ✅ API endpoints
- ✅ Privacy-preserving integration

### Phase 2: Enhanced Features (Week 3-4)
- ⬜ PharmGKB integration
- ⬜ GWAS Catalog integration
- ⬜ Automated monthly updates
- ⬜ SQLite backend for scalability

### Phase 3: HDC Integration (Future)
- ⬜ HDC-encoded clinical database
- ⬜ Similarity-based variant search
- ⬜ Topological privacy guarantees
- ⬜ Complementary HDC system

---

## 🆘 Troubleshooting

### Issue: Database not found

```bash
# Build the database first
python -m genomevault.clinical_db.data_acquisition
```

### Issue: Download fails

```bash
# Check internet connection
# ClinVar FTP may be slow, be patient (15-20 min download)
# Consider using --download-only first
```

### Issue: Out of memory

```bash
# Use filtered database
python -m genomevault.clinical_db.data_acquisition \
    --pathogenic-only \
    --min-stars 2
```

### Issue: API returns 500 error

```bash
# Check database is loaded
curl http://localhost:8000/api/v1/clinical/status

# Verify database path in app.py
```

---

## 📚 Additional Resources

- **ClinVar Documentation:** https://www.ncbi.nlm.nih.gov/clinvar/docs/
- **PharmGKB API:** https://www.pharmgkb.org/page/api
- **GWAS Catalog API:** https://www.ebi.ac.uk/gwas/docs/api
- **GenomeVault Documentation:** `docs/CLAUDE.md`

---

## 📝 License

This implementation uses public databases:
- **ClinVar:** Public domain (NCBI)
- **PharmGKB:** Requires registration
- **GWAS Catalog:** Open access

Please cite the original databases when using this implementation.