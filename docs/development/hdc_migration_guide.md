# HDC Version Migration Guide

This guide describes how to migrate between different versions of the HDC encoding system in GenomeVault.

## Overview

The HDC implementation supports multiple encoding versions to maintain backward compatibility while allowing for improvements. Each version is identified by:

- Version string (e.g., "v1.0.0")
- Dimension
- Projection type
- Seed for reproducibility

## Current Versions

| Version | Dimension | Type | Description |
|---------|-----------|------|-------------|
| v1.0.0 | 10,000 | sparse_random | Default clinical tier |
| mini_v1.0.0 | 5,000 | sparse_random | Mini tier for screening |
| full_v1.0.0 | 20,000 | sparse_random | Full tier for research |

## Migration Process

### 1. Check Version Compatibility

```python
from genomevault.hypervector_transform import HypervectorRegistry

registry = HypervectorRegistry()
compatibility = registry.check_compatibility("v1.0.0", "v2.0.0")
print(compatibility)
```

### 2. Migrate Individual Vectors

```python
from genomevault.hypervector_transform import VersionMigrator

migrator = VersionMigrator(registry)

# Migrate a single hypervector
old_hv = ...  # Your v1.0.0 hypervector
new_hv = migrator.migrate_hypervector(
    old_hv, 
    from_version="v1.0.0", 
    to_version="v2.0.0"
)
```

### 3. Batch Migration

For large datasets:

```python
def migrate_dataset(vectors, from_version, to_version):
    migrator = VersionMigrator(registry)
    migrated = []
    
    for hv in vectors:
        new_hv = migrator.migrate_hypervector(hv, from_version, to_version)
        migrated.append(new_hv)
    
    return migrated
```

### 4. Generate Migration Report

```python
# Test migration quality
report = migrator.create_migration_report("v1.0.0", "v2.0.0", test_vectors=100)
print(f"Round-trip similarity: {report['tests']['round_trip_similarity']['mean']:.3f}")
```

## Migration Strategies

### Dimension Changes

#### Dimension Reduction (e.g., 20,000 → 10,000)
- Uses stable random projection
- Preserves similarity relationships
- Some information loss is expected

#### Dimension Expansion (e.g., 5,000 → 10,000)
- Uses structured padding with correlated noise
- Maintains original information
- Additional dimensions provide capacity for new features

### Projection Type Changes

When changing projection types (e.g., gaussian → sparse):
- Approximate transformation is applied
- Some similarity degradation may occur
- Test thoroughly before production use

## Best Practices

1. **Test Before Production**
   - Always test migration on a sample dataset
   - Verify similarity preservation meets requirements
   - Check downstream task performance

2. **Gradual Migration**
   - Migrate in batches rather than all at once
   - Maintain both versions during transition
   - Monitor performance metrics

3. **Version Documentation**
   - Document why migration is needed
   - Record migration parameters used
   - Keep migration logs for audit

## Common Migration Scenarios

### Upgrading from Mini to Clinical Tier

```python
# Mini (5,000D) to Clinical (10,000D)
mini_encoder = registry.get_encoder("mini_v1.0.0")
mini_hv = mini_encoder.encode(features, OmicsType.GENOMIC)

# Migrate to clinical
clinical_hv = migrator.migrate_hypervector(
    mini_hv, 
    from_version="mini_v1.0.0", 
    to_version="v1.0.0"  # Clinical tier
)
```

### Changing Projection Type

```python
# Register new version with different projection
registry.register_version(
    version="v1.1.0",
    params={
        "dimension": 10000,
        "projection_type": "orthogonal",
        "seed": 42
    }
)

# Migrate
new_hv = migrator.migrate_hypervector(old_hv, "v1.0.0", "v1.1.0")
```

## Troubleshooting

### Poor Similarity After Migration

If similarity preservation is poor:
1. Check dimension compatibility
2. Verify seed consistency
3. Consider intermediate migration steps
4. Test with preserve_norm=True

### Performance Issues

For large-scale migrations:
1. Use batch processing
2. Enable GPU acceleration if available
3. Consider parallel processing
4. Monitor memory usage

## Version Deprecation

When deprecating old versions:

1. Announce deprecation at least 3 months in advance
2. Provide migration tools and documentation
3. Maintain read-only support for 6 months
4. Archive version specifications

## API Changes

### v1.0.0 → v2.0.0 (Future)
- No API changes required
- Internal optimization only
- Fully backward compatible

## Support

For migration assistance:
- Check the FAQ in docs/faq.md
- Open an issue on GitHub
- Contact support@genomevault.org
