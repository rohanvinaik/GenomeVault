# Experimental Features Guide

## Overview

GenomeVault includes experimental features for cutting-edge genomic privacy research. These features are isolated in the `genomevault.experimental` package to clearly separate stable production code from research prototypes.

## Status Definitions

### 🔬 Experimental
- Research prototype or proof of concept
- API will change without notice
- Not tested for production use
- May have security vulnerabilities

### 🔍 Preview
- API stabilizing but may still change
- Performance being optimized
- Limited production testing
- Security review in progress

### ✅ Stable
- API frozen with semantic versioning
- Performance optimized
- Production tested
- Security audited

### ⛔ Deprecated
- Will be removed in next major version
- Migration path provided
- No new features or fixes

## Current Experimental Features

| Feature | Module | Status | Added | Target Stable |
|---------|--------|--------|-------|---------------|
| KAN Networks | `experimental.kan` | 🔬 Experimental | v0.9.0 | v2.0.0 |
| Advanced PIR | `experimental.pir_advanced` | 🔬 Experimental | v0.8.0 | v2.1.0 |
| Recursive ZK | `experimental.zk_circuits` | 🔬 Experimental | v0.9.0 | v2.2.0 |
| Post-Quantum ZK | `experimental.zk_circuits.post_quantum` | 🔬 Experimental | v0.9.5 | v3.0.0 |

## Usage Guidelines

### 1. Explicit Opt-In

Some experimental features require explicit environment variable:

```python
import os
os.environ["GENOMEVAULT_EXPERIMENTAL"] = "true"

# Now you can import
from genomevault.experimental.zk_circuits import RecursiveSNARKProver
```

### 2. Handle Warnings

All experimental imports trigger warnings:

```python
import warnings

# Suppress experimental warnings (not recommended)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from genomevault.experimental.kan import KANHDEncoder
```

### 3. Version Pinning

When using experimental features, pin your version:

```toml
# pyproject.toml
dependencies = [
    "genomevault==0.9.0",  # Pin to specific version
]
```

## Migration Examples

### Moving from Stable to Experimental

```python
# OLD (before v0.9.0)
from genomevault.kan import KANHDEncoder

# NEW (v0.9.0+)
from genomevault.experimental.kan import KANHDEncoder
# Note: Compatibility shim exists but will be removed
```

### Checking Feature Availability

```python
from genomevault.experimental import is_experimental_enabled

if is_experimental_enabled():
    # Use experimental features
    from genomevault.experimental.zk_circuits import STARKProver
else:
    # Fall back to stable implementation
    from genomevault.zk_proofs import StandardProver as STARKProver
```

## Risk Assessment

### KAN Networks
- **Security Risk**: Low (no cryptographic operations)
- **Stability Risk**: High (API changes frequently)
- **Performance Risk**: High (unoptimized)
- **Recommendation**: Research only

### Advanced PIR
- **Security Risk**: High (unaudited crypto)
- **Stability Risk**: Medium (protocol stabilizing)
- **Performance Risk**: High (network intensive)
- **Recommendation**: Testing only, not for sensitive data

### ZK Circuits
- **Security Risk**: Critical (unverified proofs)
- **Stability Risk**: High (active development)
- **Performance Risk**: Medium (proof generation slow)
- **Recommendation**: Research only, DO NOT use for real privacy

## Development Workflow

### Adding New Experimental Feature

1. Create module under `genomevault/experimental/`
2. Add warnings in `__init__.py`
3. Document in this file
4. Add tests under `tests/experimental/`
5. Update compatibility matrix

### Promoting to Stable

1. Security audit completed
2. API frozen for 2+ releases
3. Performance benchmarks acceptable
4. Documentation complete
5. Migration guide written

### Deprecation Process

1. Mark as deprecated in documentation
2. Add deprecation warnings (2 releases)
3. Move to `genomevault.deprecated` (1 release)
4. Remove completely (major version)

## Testing Experimental Features

Run experimental tests separately:

```bash
# Run only experimental tests
pytest tests/experimental/ -v

# Run with experimental features enabled
GENOMEVAULT_EXPERIMENTAL=true pytest tests/

# Benchmark experimental features
python devtools/bench_experimental.py
```

## Reporting Issues

When reporting issues with experimental features:

1. Specify exact version: `genomevault.__version__`
2. Include experimental module: `experimental.kan`
3. Provide minimal reproduction
4. Note any warnings received
5. Describe expected vs actual behavior

## Future Experimental Features

Planned experimental additions:

- **Homomorphic encryption** (v1.0.0)
- **Quantum-resistant signatures** (v1.1.0)
- **Federated learning protocols** (v1.2.0)
- **Secure multi-party computation** (v1.3.0)

## FAQ

**Q: Can I use experimental features in production?**
A: Not recommended. They lack security audits and stability guarantees.

**Q: Will experimental APIs change?**
A: Yes, without deprecation warnings. Pin versions if you depend on them.

**Q: How do features graduate to stable?**
A: Through security audit, API stabilization, and performance validation.

**Q: Can I contribute experimental features?**
A: Yes! Open an RFC issue first to discuss the design.

---
*Last Updated: 2025-08-09*
*Version: 0.9.0*
