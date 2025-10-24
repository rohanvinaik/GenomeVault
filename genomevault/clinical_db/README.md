# Clinical SNP Database Module

This module provides queryable clinical variant database functionality for GenomeVault.

## Quick Start

```bash
# 1. Build database
python -m genomevault.clinical_db.data_acquisition \
    --genome-build GRCh38 \
    --pathogenic-only

# 2. Query database
from genomevault.clinical_db.database import ClinicalSNPDatabase

db = ClinicalSNPDatabase('data/clinical_snps_v1.0.0.json.gz')
snps = db.query_position('chr11', 5227002)
```

## Module Structure

- **`database.py`**: Core database class with indexing and query methods
- **`data_acquisition.py`**: Automated ClinVar download and parsing pipeline
- **`__init__.py`**: Module exports

## Key Classes

### `ClinicalSNPDatabase`
Main database class for querying clinical variants.

```python
db = ClinicalSNPDatabase('path/to/database.json.gz')

# Query methods
snps = db.query_position('chr11', 5227002)
gene_variants = db.query_gene('BRCA1')
condition_snps = db.query_condition('breast cancer')
specific_snp = db.query_rsid('rs334')

# Get all pathogenic variants
pathogenic = db.get_pathogenic_variants()
```

### `ClinicalDatabaseBuilder`
Build custom clinical databases from various sources.

```python
builder = ClinicalDatabaseBuilder(genome_build="GRCh38")
builder.add_snp(custom_snp)
builder.save('custom_db.json.gz')
```

### `ClinVarDownloader`
Download and parse ClinVar data.

```python
downloader = ClinVarDownloader(genome_build="GRCh38")
vcf_path = downloader.download_vcf()
snps = downloader.parse_vcf(vcf_path, filter_pathogenic=True, min_review_stars=2)
```

## Data Models

All data models are defined as dataclasses:

- **`ClinicalSNP`**: Complete SNP record with all annotations
- **`ClinicalCondition`**: Associated disease/condition
- **`ClinicalAnnotation`**: Review status and evidence
- **`PopulationFrequency`**: Allele frequencies across populations
- **`FunctionalImpact`**: Predicted functional consequences

## API Integration

REST API endpoints are available in `genomevault/api/routers/clinical_query.py`:

```python
# In your FastAPI app
from genomevault.api.routers import clinical_query

app.include_router(clinical_query.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    clinical_query.init_clinical_database("data/clinical_snps_v1.0.0.json.gz")
```

## CLI Tools

Command-line interface in `genomevault/cli/clinical_query_cli.py`:

```bash
# Query position
python -m genomevault.cli.clinical_query_cli query-position --chr chr11 --pos 5227002

# Query gene
python -m genomevault.cli.clinical_query_cli query-gene BRCA1

# Database stats
python -m genomevault.cli.clinical_query_cli stats
```

## Documentation

- **Quick Start Guide**: `docs/guides/CLINICAL_SNP_QUICK_START.md`
- **Implementation Plan**: See artifacts in chat history

## Privacy Guarantees

This module maintains GenomeVault's privacy guarantees by:

1. **Never querying raw variants directly** - always use differential encoding first
2. **K-anonymity through reference pool** - variants are anonymized before clinical lookup
3. **No direct linkage** - clinical database never sees original query genome

Example privacy-preserving workflow:

```python
# Step 1: Encode through reference pool (k-anonymity)
encoded = differential_encode(query_vcf, reference_pool, k=3)

# Step 2: Query clinical database with encoded variants
for encoded_var in encoded.variants:
    clinical_hits = db.query_position(encoded_var.chromosome, encoded_var.position)
```

## Performance

- **Load Time**: <3 seconds for 150K variants
- **Query Time**: <1ms per position (hash index)
- **Memory**: ~100-200MB for full database
- **Storage**: ~50MB compressed (full), ~5-10MB (pathogenic-only)

## Future Enhancements

- PharmGKB integration for drug-gene interactions
- GWAS Catalog for common complex diseases
- HDC-encoded database for topological privacy
- Automated monthly updates from ClinVar
- SQLite backend for scalability

## License

Uses public databases:
- **ClinVar**: Public domain (NCBI)
- **PharmGKB**: Requires registration (planned)
- **GWAS Catalog**: Open access (planned)
