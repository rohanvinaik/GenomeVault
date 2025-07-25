# SNP Dial Implementation for GenomeVault

This implementation adds a user-tunable single-nucleotide accuracy dial to the GenomeVault codebase, enabling precise variant encoding without exploding RAM or proof size.

## Overview

The SNP dial allows users to choose between different levels of genomic resolution:
- **Off**: Standard genome-wide encoding (baseline)
- **Common**: 1M common SNP positions (+8s GPU, +40MB disk)
- **Clinical**: 10M dbSNP/ClinVar positions (+45s GPU, +400MB disk)
- **Custom**: User-provided BED/VCF file

## Architecture

### 1. Sparse-Panel Encoding (Fast Path)

**Key Components:**
- `hypervector/positional.py`: Memory-efficient position vector generation using 32-bit hash seeds
- `PositionalEncoder`: Creates orthogonal position vectors for up to 10M positions
- `SNPPanel`: Manages variant panels and encoding logic

**Process:**
1. Pre-compute position vectors using hash-based seeds (not full arrays)
2. Bind position vector P_i with observed base N_i
3. XOR all selected positions into one panel hypervector (~100k dimensions)
4. Apply ECC parity(3) compression → ~0.53 MB per genome

### 2. Hierarchical Zoom Tiles

**Zoom Levels:**
- **Level 0**: Existing genome-wide hypervector
- **Level 1**: 1 Mb window hypervectors (~3k vectors)
- **Level 2**: 1 kb tiles (~3M vectors, fetched lazily)

**API Sequence:**
```
POST /query/zoom/query      # Level-0 scan to find hotspots
POST /query/zoom/fetch_tiles # PIR only for candidate windows
```

### 3. Code Integration Points

```
genomevault/
├─ hypervector/
│   ├─ positional.py         # NEW: Positional encoding for SNPs
│   ├─ encoding/genomic.py   # MODIFIED: Added panel & zoom modes
│
├─ pir/
│   ├─ client/batched_query_builder.py  # MODIFIED: Added zoom queries
│
└─ api/
    ├─ routers/query_tuned.py  # NEW: Panel & zoom endpoints
```

## Usage Examples

### Basic Panel Usage

```python
from genomevault.hypervector.encoding.genomic import GenomicEncoder, PanelGranularity

# Create encoder with SNP mode
encoder = GenomicEncoder(
    dimension=100000,
    enable_snp_mode=True,
    panel_granularity=PanelGranularity.CLINICAL
)

# Encode variants with clinical panel
variants = [
    {"chromosome": "chr1", "position": 100000, "ref": "A", "alt": "G"},
    {"chromosome": "chr1", "position": 100100, "ref": "C", "alt": "T"}
]

encoded_genome = encoder.encode_genome_with_panel(variants)
```

### Custom Panel Loading

```python
# Load custom BED file
encoder.load_custom_panel("my_variants.bed", "custom_panel")

# Use custom panel for encoding
encoded = encoder.encode_genome_with_panel(variants, "custom_panel")
```

### Hierarchical Zoom Queries

```python
# API endpoint for zoom queries
POST /api/query/zoom
{
    "cohort_id": "study123",
    "chromosome": "chr1",
    "start_position": 1000000,
    "end_position": 5000000,
    "initial_level": 0,
    "auto_zoom": true,
    "epsilon": 0.01,
    "delta_exp": 15
}
```

## Performance Characteristics

| Panel Setting | Positions | Extra Encode Time | Extra RAM | JL Error Bound |
|--------------|-----------|-------------------|-----------|----------------|
| Off          | 0         | 0s                | 0 MB      | ≤1% baseline   |
| Common       | 1M        | +8s GPU           | +40 MB    | ≤1% maintained |
| Clinical     | 10M       | +45s GPU          | +400 MB   | ≤1% maintained |

## API Endpoints

### Panel Query Endpoint
```
POST /api/query/panel
```
Execute queries with SNP panel encoding for single-nucleotide accuracy.

### Zoom Query Endpoint
```
POST /api/query/zoom
```
Perform hierarchical zoom queries on genomic regions.

### Panel Info Endpoint
```
GET /api/query/panel/info?panel_name=all
```
Get information about available SNP panels.

### Panel Overhead Estimation
```
POST /api/query/panel/estimate
```
Estimate performance overhead for different panel configurations.

## Testing

Run the test script to see the SNP dial in action:
```bash
python test_snp_dial.py
```

This will demonstrate:
1. Different panel granularity settings
2. Custom panel loading
3. Performance benchmarking
4. Hierarchical zoom functionality

## Integration with Existing Systems

The SNP dial integrates seamlessly with:
- **PIR System**: Panel-encoded hypervectors work with existing PIR queries
- **ZK Proofs**: Proof aggregation supports multiple zoom levels
- **Error Tuning**: Johnson-Lindenstrauss bounds maintained with sparse encoding

## Future Extensions

1. **Dynamic Panel Selection**: Auto-select panel based on query requirements
2. **Streaming Panel Updates**: Add new variants to panels without full reload
3. **Panel Compression**: Further reduce memory using bloom filters
4. **GPU Acceleration**: Optimize sparse vector operations for CUDA

## References

- Johnson-Lindenstrauss lemma for dimensionality bounds
- Sparse random projections (Achlioptas, 2003)
- Hyperdimensional computing for genomics
