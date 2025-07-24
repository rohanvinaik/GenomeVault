# HDC (Hyperdimensional Computing) Implementation

This directory contains the complete implementation of the Hierarchical Hyperdimensional Computing (HDC) encoding system for GenomeVault. HDC provides privacy-preserving transformations of genomic data into high-dimensional vectors that maintain similarity relationships while preventing data reconstruction.

## Overview

The HDC implementation transforms sensitive genomic and clinical data into high-dimensional hypervectors (10,000-20,000 dimensions) that:

- **Preserve similarity relationships** - Similar genomic profiles produce similar hypervectors
- **Ensure privacy** - Original data cannot be reconstructed from hypervectors (one-way transformation)
- **Enable efficient computation** - Support for fast similarity search and pattern matching
- **Provide compression** - Achieve 100-1000x compression ratios

## Architecture

### Core Components

1. **`hdc_encoder.py`** - Main encoding engine
   - Multi-tier compression (Mini, Clinical, Full)
   - Multiple projection types (Gaussian, Sparse, Orthogonal)
   - Support for all omics types (Genomic, Transcriptomic, Epigenomic, etc.)

2. **`binding_operations.py`** - Hypervector operations
   - Binding types: Multiply, Circular convolution, Permutation, XOR, Fourier (HRR)
   - Bundling (superposition) for combining vectors
   - Composite binding for structured data

3. **`registry.py`** - Version and configuration management
   - Deterministic encoding with seed management
   - Version tracking and migration
   - Backward compatibility support

4. **`hdc_api.py`** - RESTful API endpoints
   - `/encode` - Encode single modality
   - `/encode_multimodal` - Encode and bind multiple modalities
   - `/similarity` - Compute vector similarity
   - `/version` - Version information

## Compression Tiers

| Tier | Dimension | Size | Use Case |
|------|-----------|------|----------|
| **Mini** | 5,000 | ~25 KB | Quick similarity searches, population screening |
| **Clinical** | 10,000 | ~300 KB | Clinical decision support, pharmacogenomics |
| **Full** | 20,000 | ~200 KB | Research, drug discovery, precision medicine |

## Mathematical Properties

The HDC encoding preserves key mathematical properties:

- **Commutativity**: `bind(a, b) = bind(b, a)` for multiplication/XOR
- **Associativity**: `bind(bind(a, b), c) = bind(a, bind(b, c))`
- **Approximate Inverse**: `unbind(bind(a, b), b) ≈ a` (similarity > 0.95)
- **Distributivity**: `bind(bundle(a, b), c) ≈ bundle(bind(a, c), bind(b, c))`

## Privacy Guarantees

1. **Non-invertibility**: Cannot recover original data without the projection matrix
2. **Computational infeasibility**: 10,000-dimensional space with random projections
3. **Information theoretic security**: Projection is a many-to-one mapping

## Usage Examples

### Basic Encoding

```python
from genomevault.hypervector_transform import create_encoder, OmicsType

# Create encoder
encoder = create_encoder(dimension=10000)

# Encode genomic data
genomic_features = {"variants": [/* variant data */]}
hv = encoder.encode(genomic_features, OmicsType.GENOMIC)

# Compute similarity
hv2 = encoder.encode(other_features, OmicsType.GENOMIC)
similarity = encoder.similarity(hv, hv2)  # Returns value between -1 and 1
```

### Multi-modal Binding

```python
from genomevault.hypervector_transform import HypervectorBinder, BindingType

# Create binder
binder = HypervectorBinder(dimension=10000)

# Encode multiple modalities
genomic_hv = encoder.encode(genomic_data, OmicsType.GENOMIC)
clinical_hv = encoder.encode(clinical_data, OmicsType.CLINICAL)

# Bind modalities together
combined = binder.bind([genomic_hv, clinical_hv], BindingType.FOURIER)
```

### Using the Registry

```python
from genomevault.hypervector_transform import HypervectorRegistry

# Create registry
registry = HypervectorRegistry()

# Get specific version
encoder = registry.get_encoder("v1.0.0")

# Register custom version
registry.register_version(
    version="custom_v1",
    params={
        "dimension": 15000,
        "projection_type": "sparse_random",
        "seed": 12345
    }
)
```

## API Usage

### Starting the API Server

```bash
uvicorn genomevault.api.main:app --reload
```

### Example API Calls

```bash
# Encode genomic data
curl -X POST http://localhost:8000/api/v1/hdc/encode \
  -H "Content-Type: application/json" \
  -d '{
    "features": {"variants": [1, 2, 3, 4, 5]},
    "omics_type": "genomic",
    "compression_tier": "clinical"
  }'

# Compute similarity
curl -X POST http://localhost:8000/api/v1/hdc/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "vector1": [/* hypervector 1 */],
    "vector2": [/* hypervector 2 */],
    "metric": "cosine"
  }'
```

## Performance Benchmarks

Run benchmarks with:

```bash
# Quick benchmark
make bench-hdc

# Comprehensive benchmark
python scripts/bench_hdc.py --output-dir benchmarks/hdc
```

Expected performance (Clinical tier, 10,000D):
- **Encoding throughput**: >100 operations/second
- **Memory usage**: ~300 KB per vector
- **Similarity computation**: >10,000 operations/second

## Testing

### Unit Tests
```bash
pytest tests/test_hdc_implementation.py -v
```

### Quality Tests
```bash
pytest tests/test_hdc_quality.py -v
```

### Property-based Tests
```bash
pytest tests/property/test_hdc_properties.py -v --hypothesis-show-statistics
```

### Adversarial Tests
```bash
pytest tests/adversarial/test_hdc_adversarial.py -v
```

### All HDC Tests
```bash
make test-hdc
```

## Development

### Code Quality

Run linter checks:
```bash
python scripts/run_hdc_linters.py
```

Or use Make:
```bash
make lint
make format  # Auto-fix formatting issues
```

### Adding New Projection Types

1. Add to `ProjectionType` enum in `hdc_encoder.py`
2. Implement `_create_<type>_projection()` method
3. Add tests in `test_hdc_implementation.py`
4. Update benchmarks in `bench_hdc.py`

### Adding New Binding Operations

1. Add to `BindingType` enum in `binding_operations.py`
2. Implement `_<type>_bind()` and `_<type>_unbind()` methods
3. Add algebraic property tests
4. Document mathematical properties

## Security Considerations

The HDC implementation includes several security features:

1. **Constant-time operations** - Prevent timing side-channels
2. **Input validation** - Reject malformed inputs
3. **Resource limits** - Prevent memory exhaustion attacks
4. **Error message sanitization** - No sensitive information in errors

See `tests/adversarial/test_hdc_adversarial.py` for security testing.

## Migration Guide

### Upgrading Versions

```python
from genomevault.hypervector_transform import VersionMigrator

# Migrate hypervector to new version
migrator = VersionMigrator(registry)
new_hv = migrator.migrate_hypervector(old_hv, "v1.0.0", "v2.0.0")

# Generate migration report
report = migrator.create_migration_report("v1.0.0", "v2.0.0")
```

## Troubleshooting

### Common Issues

1. **Memory errors with large dimensions**
   - Use sparse projections: `projection_type="sparse_random"`
   - Reduce batch size
   - Use GPU acceleration if available

2. **Slow encoding performance**
   - Enable PyTorch optimizations
   - Use batch encoding for multiple samples
   - Consider using Mini or Clinical tier

3. **Poor similarity preservation**
   - Increase dimension
   - Use appropriate projection type for data
   - Normalize input features

## References

1. Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors.
2. Plate, T. (2003). Holographic Reduced Representations.
3. Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines.

## License

See LICENSE file in the project root.

## Contributing

See CONTRIBUTING.md for guidelines on contributing to the HDC implementation.
