# Migration Guide: Differential Encoding

**Last Updated**: 2025-10-19
**Status**: Production Ready
**Backward Compatibility**: ✅ Maintained

## Overview

This guide helps you migrate from legacy direct variant encoding to the new differential encoding system in GenomeVault. Differential encoding provides:

- **Cryptographic Security**: HMAC-based binding and verification
- **Better Compression**: 50-100× compression vs. raw VCF
- **Privacy Preservation**: Reference-based differential storage
- **Mathematical Guarantees**: Provable security properties

**Important**: Legacy encoding remains fully supported for backward compatibility.

## Quick Start

### For New Projects

```python
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode
from genomevault.differential_encoding import AnalysisType, Genome, Variant

# Create encoder with differential mode
encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL)

# Create genome
genome = Genome(
    genome_id="patient_001",
    assembly="GRCh38",
    chromosomes={
        "chr1": [
            Variant(chromosome="chr1", position=100000, ref="A", alt="G"),
            # ... more variants
        ]
    }
)

# Encode
encoded = encoder.encode_genome(genome, AnalysisType.SLIDING_WINDOW)

# Save
encoded.save("patient_001.enc.gz")
```

### For Existing Projects

No changes required! Legacy code continues to work:

```python
# This still works exactly as before
from genomevault.hypervector_transform import HypervectorEncoder
encoder = HypervectorEncoder(config)
vector = encoder.encode(features, OmicsType.GENOMIC)
```

## Migration Paths

### Path 1: Gradual Migration (Recommended)

Keep legacy encoding while gradually adopting differential encoding:

```python
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode

# Create encoder in AUTO mode
encoder = UnifiedGenomicEncoder(mode=EncodingMode.AUTO)

# Legacy features still use legacy encoding
legacy_vector = encoder.encode(features, OmicsType.GENOMIC)

# New genome encoding uses differential encoding
differential_encoded = encoder.encode_genome(genome, AnalysisType.SLIDING_WINDOW)
```

### Path 2: Feature Flag Controlled

Use environment variables to control rollout:

```bash
# Enable differential encoding
export GENOMEVAULT_ENABLE_DIFFERENTIAL=true

# Make it default for new encodings
export GENOMEVAULT_DIFFERENTIAL_DEFAULT=true

# Keep legacy fallback for safety
export GENOMEVAULT_LEGACY_FALLBACK=true
```

```python
from genomevault.hypervector_transform import UnifiedGenomicEncoder

# Automatically uses differential when enabled
encoder = UnifiedGenomicEncoder(mode=EncodingMode.AUTO)
```

### Path 3: Direct Migration

Directly use differential encoding everywhere:

```python
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode

# Explicit differential mode
encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL)

# All encoding uses differential mode
encoded = encoder.encode_genome(genome, AnalysisType.SLIDING_WINDOW)
```

## API Changes

### Legacy API (Still Supported)

```python
POST /api/v1/hdc/encode
{
    "features": {"variant_count": 1000, ...},
    "omics_type": "genomic",
    "compression_tier": "full"
}
```

### New Differential API

```python
POST /api/v1/differential/encode
{
    "genome": {
        "genome_id": "patient_001",
        "assembly": "GRCh38",
        "chromosomes": {
            "chr1": [
                {"chromosome": "chr1", "position": 100000, "ref": "A", "alt": "G"}
            ]
        }
    },
    "analysis_type": "sliding_window",
    "bundle_chunks": true
}
```

## Feature Comparison

| Feature | Legacy Encoding | Differential Encoding |
|---------|----------------|----------------------|
| Compression | ~10× | **50-100×** |
| Security | Basic | **Cryptographic** |
| Query Speed | O(n) | **O(log n + k)** |
| Metadata | Limited | **Complete** |
| Verification | None | **SHA256 + HMAC** |
| Privacy | Basic | **Mathematical guarantees** |
| Storage Format | Torch tensors | **Compressed JSON** |

## Configuration

### Environment Variables

```bash
# Differential encoding control
GENOMEVAULT_ENABLE_DIFFERENTIAL=true          # Enable differential encoding
GENOMEVAULT_DIFFERENTIAL_DEFAULT=false        # Use differential by default
GENOMEVAULT_LEGACY_FALLBACK=true              # Fallback to legacy if differential fails
GENOMEVAULT_HYBRID_MODE=false                 # Enable hybrid mode (experimental)

# Performance tuning
GENOMEVAULT_ENABLE_CACHING=true               # Enable result caching
GENOMEVAULT_ENABLE_BATCHING=true              # Enable batch processing

# Compatibility
GENOMEVAULT_STRICT_COMPATIBILITY=false        # Strict backward compatibility mode
```

### Programmatic Configuration

```python
from genomevault.hypervector_transform import (
    UnifiedGenomicEncoder,
    EncodingMode,
    EncodingFeatureFlags,
)

# Custom feature flags
flags = EncodingFeatureFlags(
    enable_differential=True,
    differential_by_default=False,
    legacy_fallback=True,
    enable_caching=True,
)

encoder = UnifiedGenomicEncoder(
    mode=EncodingMode.AUTO,
    feature_flags=flags,
    dimension=10000,
    seed=42,
)
```

## Common Migration Patterns

### Pattern 1: Encode + Store + Query

```python
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode
from genomevault.differential_encoding import (
    AnalysisType,
    Genome,
    DifferentialGenomeQuery,
)

# Encode
encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL)
encoded = encoder.encode_genome(genome, AnalysisType.SLIDING_WINDOW)

# Store
encoded.save("patient_001.enc.gz")

# Load and query
from genomevault.differential_encoding import EncodedGenome
loaded = EncodedGenome.load("patient_001.enc.gz")

# Query specific region
query = DifferentialGenomeQuery(
    encoder.reference_manager,
    encoder.differential_encoder.hv_encoder
)
result = query.query_region(loaded, "chr1", 100000, 200000)

print(f"Found {result.variant_count} variants")
```

### Pattern 2: Batch Processing

```python
from pathlib import Path
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode

encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL)

# Process multiple genomes
for genome_file in Path("genomes/").glob("*.vcf"):
    genome = load_genome_from_vcf(genome_file)  # Your VCF parser

    encoded = encoder.encode_genome(genome, AnalysisType.SLIDING_WINDOW)

    output_path = f"encoded/{genome.genome_id}.enc.gz"
    encoded.save(output_path)

    print(f"Encoded {genome.genome_id}: {encoded.storage_size_kb():.1f} KB")
```

### Pattern 3: API Integration

```python
from fastapi import FastAPI
from genomevault.hypervector_transform.hdc_api import include_routes
from genomevault.hypervector_transform.differential_api import include_differential_routes

app = FastAPI()

# Include both legacy and differential endpoints
include_routes(app)                    # Legacy: /api/v1/hdc/*
include_differential_routes(app)       # New: /api/v1/differential/*

# Both APIs available simultaneously
```

## Performance Optimization

### Chunk Size Tuning

```python
from genomevault.differential_encoding import ChunkingStrategy, AnalysisType

# For large genomes, increase chunk size
large_genome_strategy = ChunkingStrategy(
    window_size=1_000_000,   # 1 MB chunks
    overlap=100_000,          # 100 KB overlap
    min_variants=5,
)

# Register custom strategy
from genomevault.differential_encoding import STRATEGY_CONFIGS
STRATEGY_CONFIGS[AnalysisType.CUSTOM_INTERVALS] = large_genome_strategy
```

### Reference Genome Caching

```python
from pathlib import Path
from genomevault.differential_encoding import SecureReferenceGenomeManager

# Pre-load references for faster encoding
reference_manager = SecureReferenceGenomeManager(
    reference_dir=Path("references/")
)

# Load references from VCF files
for vcf_file in Path("references/").glob("*.vcf"):
    reference_manager.add_reference_from_vcf(vcf_file)

# Create encoder with pre-loaded references
encoder = UnifiedGenomicEncoder(
    mode=EncodingMode.DIFFERENTIAL,
    reference_dir=Path("references/"),
)
```

## Testing Migration

### Unit Tests

```python
import pytest
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode

def test_backward_compatibility():
    """Test that legacy encoding still works."""
    encoder = UnifiedGenomicEncoder(mode=EncodingMode.LEGACY)

    # Legacy encoding should work
    vector = encoder.encode(features, OmicsType.GENOMIC)

    assert vector is not None
    assert len(vector) == 10000

def test_differential_encoding():
    """Test differential encoding."""
    encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL)

    # Differential encoding
    encoded = encoder.encode_genome(genome, AnalysisType.SLIDING_WINDOW)

    assert encoded.genome_id == genome.genome_id
    assert len(encoded.chunk_hypervectors) > 0
    assert encoded.verify()
```

### Integration Tests

```python
def test_end_to_end_migration():
    """Test complete migration workflow."""
    encoder = UnifiedGenomicEncoder(mode=EncodingMode.AUTO)

    # Encode with differential
    encoded = encoder.encode_genome(genome, AnalysisType.SLIDING_WINDOW)

    # Save and load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".enc.gz") as f:
        encoded.save(f.name)
        loaded = EncodedGenome.load(f.name)

    # Verify integrity
    assert loaded.verify()
    assert loaded.genome_id == genome.genome_id
```

## Troubleshooting

### Issue: "Differential encoder not initialized"

**Cause**: Reference genomes not loaded
**Solution**:

```python
# Ensure references are available
from pathlib import Path
reference_dir = Path("references/")
reference_dir.mkdir(exist_ok=True)

encoder = UnifiedGenomicEncoder(
    mode=EncodingMode.DIFFERENTIAL,
    reference_dir=reference_dir,
)

# Add at least one reference
from genomevault.differential_encoding import ReferenceGenome
encoder.reference_manager.pool.add_reference(your_reference)
```

### Issue: "API compatibility issues"

**Cause**: Using old parameter names
**Solution**: Check API compatibility section in code

### Issue: Performance degradation

**Cause**: Too many small chunks
**Solution**: Tune chunking strategy

```python
from genomevault.differential_encoding import STRATEGY_CONFIGS, AnalysisType

# Increase chunk size
STRATEGY_CONFIGS[AnalysisType.SLIDING_WINDOW].window_size = 1_000_000
```

## Rollback Plan

If you need to rollback:

1. **Environment Variable**:
```bash
export GENOMEVAULT_ENABLE_DIFFERENTIAL=false
```

2. **Code Change**:
```python
encoder = UnifiedGenomicEncoder(mode=EncodingMode.LEGACY)
```

3. **API**: Continue using `/api/v1/hdc/encode` endpoint

## Support

- **Documentation**: `docs/differential_encoding/`
- **Examples**: `examples/differential_encoding_demo.py`
- **Tests**: `tests/differential_encoding/`
- **API Reference**: `/api/v1/differential/docs`

## Timeline

- **Phase 1 (Current)**: Dual support - both encodings available
- **Phase 2 (Q1 2026)**: Differential encoding default for new deployments
- **Phase 3 (Q2 2026)**: Legacy encoding deprecated (still available)
- **Phase 4 (2027)**: Legacy encoding removed (with migration tool)

## Checklist

Use this checklist to track your migration:

- [ ] Read this migration guide
- [ ] Test differential encoding in development environment
- [ ] Set up feature flags for gradual rollout
- [ ] Update API clients to handle new response formats
- [ ] Load reference genomes
- [ ] Run performance benchmarks
- [ ] Test backward compatibility
- [ ] Deploy to staging
- [ ] Monitor performance and errors
- [ ] Gradually increase differential encoding percentage
- [ ] Complete migration to differential encoding

## Next Steps

1. **Try differential encoding**: Start with `examples/differential_encoding_demo.py`
2. **Run benchmarks**: See `docs/performance_comparison.md`
3. **Update tests**: Add differential encoding to your test suite
4. **Gradual rollout**: Use feature flags to control adoption
5. **Monitor**: Track performance and error rates

For questions or issues, please consult the main documentation or file an issue.
