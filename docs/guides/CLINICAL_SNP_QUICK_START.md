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
python -m genomevault.cli.clinical_query_cli query-position \
    --chr chr11 \
    --pos 5227002

# 3. Query a gene
python -m genomevault.cli.clinical_query_cli query-gene BRCA1

# 4. Check database stats
python -m genomevault.cli.clinical_query_cli stats
```

---

## 📦 Installation

### Dependencies

```bash
# Install required packages
pip install requests click tabulate

# If using API
pip install fastapi uvicorn
```

### File Structure

The clinical database module is located at:

```
genomevault/clinical_db/
├── __init__.py              # Module exports
├── database.py              # Core database class
└── data_acquisition.py      # Data download/build pipeline

genomevault/api/routers/
└── clinical_query.py        # REST API endpoints

genomevault/cli/
└── clinical_query_cli.py    # Command-line tools
```

---

## 🗄️ Building the Database

### Option 1: Pathogenic-Only Database (Recommended for testing)

```bash
python -m genomevault.clinical_db.data_acquisition \
    --genome-build GRCh38 \
    --output-dir data \
    --pathogenic-only \
    --min-stars 2
```

**Expected output:**
- ~10,000-20,000 high-confidence pathogenic variants
- Database size: ~5-10MB
- Processing time: ~2-3 minutes

### Option 2: Full ClinVar Database

```bash
python -m genomevault.clinical_db.data_acquisition \
    --genome-build GRCh38 \
    --output-dir data
```

**Expected output:**
- ~150K variants
- Database size: ~50MB
- Download time: 15-20 minutes
- Processing time: 5-10 minutes

---

## 💻 Usage Examples

### Command-Line Interface

#### 1. Database Statistics

```bash
python -m genomevault.cli.clinical_query_cli stats
```

#### 2. Query Specific Position (e.g., Sickle Cell)

```bash
python -m genomevault.cli.clinical_query_cli query-position \
    --chr chr11 \
    --pos 5227002 \
    --format detailed
```

#### 3. Query Gene (e.g., BRCA1)

```bash
python -m genomevault.cli.clinical_query_cli query-gene BRCA1 --format summary
```

#### 4. Query by rsID

```bash
python -m genomevault.cli.clinical_query_cli query-rsid rs334
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

# Get statistics
stats = db.get_statistics()
print(f"Database contains {stats['total_snps']} SNPs")
```

---

## 🌐 REST API

### 1. Setup API

Add to your FastAPI app initialization (e.g., in `genomevault/api/app.py`):

```python
from genomevault.api.routers import clinical_query

# Include the router
app.include_router(clinical_query.router, prefix="/api/v1")

# Initialize database at startup
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
curl -X POST "http://localhost:8000/api/v1/clinical-db/query/positions" \
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
curl "http://localhost:8000/api/v1/clinical-db/query/gene/BRCA1"
```

#### Get Pathogenic Variants

```bash
curl "http://localhost:8000/api/v1/clinical-db/pathogenic?limit=50"
```

#### Database Status

```bash
curl "http://localhost:8000/api/v1/clinical-db/status"
```

#### Database Statistics

```bash
curl "http://localhost:8000/api/v1/clinical-db/stats"
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

---

## 🆘 Troubleshooting

### Issue: Database not found

```bash
# Build the database first
python -m genomevault.clinical_db.data_acquisition
```

### Issue: Download fails

```bash
# ClinVar FTP may be slow, be patient (15-20 min download)
# Check internet connection
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
curl http://localhost:8000/api/v1/clinical-db/status

# Verify database path in app initialization
```

---

## 📚 Additional Resources

- **ClinVar Documentation:** https://www.ncbi.nlm.nih.gov/clinvar/docs/
- **GenomeVault Documentation:** `docs/CLAUDE.md`
- **Implementation Plan:** `docs/guides/CLINICAL_SNP_IMPLEMENTATION_PLAN.md`

---

## 🚀 Next Steps

1. Build the database with your desired configuration
2. Try the example queries above
3. Integrate with your GenomeVault pipeline
4. Add custom filters or additional data sources as needed

For detailed implementation information, see `CLINICAL_SNP_IMPLEMENTATION_PLAN.md`.
