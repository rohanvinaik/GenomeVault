# GenomeVault Module Structure

This document describes the module organization and migration paths for the GenomeVault codebase.

## Overview

GenomeVault is undergoing a module reorganization to improve code clarity and maintainability. This document serves as a guide for developers working with the codebase.

## Current Module Organization

### Core Modules (Stable)

#### `genomevault.core`
- **Purpose**: Core functionality and constants
- **Key Components**:
  - `constants.py` - System-wide constants (HYPERVECTOR_DIMENSIONS, etc.)
  - `exceptions.py` - Custom exception classes
  - `config.py` - Configuration management
- **Status**: ✅ Stable - Use these imports

#### `genomevault.config`
- **Purpose**: Path and configuration management
- **Key Components**:
  - `paths.py` - Centralized path management (PROJECT_ROOT, DATA_DIR, etc.)
- **Status**: ✅ Stable - Use for all path operations

### Hypervector Modules (In Transition)

#### `genomevault.hypervector_transform` (NEW - Recommended)
- **Purpose**: Primary hypervector encoding and transformation
- **Key Components**:
  - `encoding.py` - Main encoder classes (HypervectorEncoder, HypervectorConfig)
  - `binding_operations.py` - Binding and bundling operations
  - `hdc_encoder.py` - HDC encoder wrapper
  - `registry.py` - Version registry and migration
  - `hierarchical.py` - Hierarchical encoding strategies
- **Status**: ✅ Active Development - **USE THIS MODULE**

#### `genomevault.hypervector` (DEPRECATED)
- **Purpose**: Legacy hypervector operations
- **Components**:
  - `encoder.py` - ⚠️ DEPRECATED - Compatibility shim to hypervector_transform.encoding
  - `encoding/genomic.py` - ⚠️ DEPRECATED - Use hypervector_transform.encoding
  - `operations/binding.py` - ⚠️ DEPRECATED - Use hypervector_transform.binding_operations
  - `error_handling.py` - Still active for error correction
  - `positional.py` - Still active for positional encoding
- **Status**: ⚠️ Deprecated - Migrate to hypervector_transform

### API Modules

#### `genomevault.api`
- **Purpose**: FastAPI application and endpoints
- **Key Components**:
  - `app.py` - Main FastAPI application
  - `main.py` - Application entry point
  - `routers/` - API route handlers
    - `healthz.py` - Health check endpoints (/healthz)
    - `hv.py` - Hypervector encoding endpoints (/api/v1/hv/*)
    - `metrics.py` - Prometheus metrics (/metrics)
  - `models/` - Pydantic models
    - `hv.py` - Request/response models for HV endpoints
- **Status**: ✅ Stable

### Privacy & Security Modules

#### `genomevault.zk_proofs`
- **Purpose**: Zero-knowledge proof generation and verification
- **Status**: ✅ Stable

#### `genomevault.pir`
- **Purpose**: Private Information Retrieval implementations
- **Status**: ✅ Stable

#### `genomevault.federated`
- **Purpose**: Federated learning infrastructure
- **Status**: ✅ Stable

### Specialized Modules

#### `genomevault.kan`
- **Purpose**: Kolmogorov-Arnold Network implementations
- **Status**: ✅ Stable

#### `genomevault.nanopore`
- **Purpose**: Nanopore sequencing support
- **Status**: ✅ Stable

#### `genomevault.blockchain`
- **Purpose**: Blockchain governance and control
- **Status**: ✅ Stable

## Migration Guide

### For Imports

#### Old Import → New Import

```python
# ❌ OLD (Deprecated)
from genomevault.hypervector.encoder import HypervectorEncoder
from genomevault.hypervector.encoding.genomic import GenomicEncoder
from genomevault.hypervector.operations.binding import circular_convolution

# ✅ NEW (Recommended)
from genomevault.hypervector_transform.encoding import HypervectorEncoder
from genomevault.hypervector_transform.encoding import HypervectorEncoder as GenomicEncoder
from genomevault.hypervector_transform.binding_operations import circular_bind
```

### Common Import Patterns

```python
# Core imports (stable)
from genomevault.core.constants import HYPERVECTOR_DIMENSIONS, OmicsType
from genomevault.core.exceptions import HypervectorError, GVError
from genomevault.config.paths import PROJECT_ROOT, DATA_DIR

# Hypervector operations (use new module)
from genomevault.hypervector_transform.encoding import (
    HypervectorEncoder,
    HypervectorConfig
)
from genomevault.hypervector_transform.binding_operations import (
    HypervectorBinder,
    BindingOperations
)

# API models
from genomevault.api.models.hv import (
    HVEncodeRequest,
    HVEncodeResponse
)
```

## Deprecation Timeline

### Phase 1 (Current)
- Compatibility shims in place with deprecation warnings
- Both old and new imports work
- Console warnings guide migration

### Phase 2 (Next Minor Version)
- Deprecation warnings become more prominent
- Documentation updated to only show new imports
- Test suite migrated to new imports

### Phase 3 (Next Major Version)
- Old module paths removed
- Only new imports supported
- Breaking change noted in release notes

## Module Dependencies

```
genomevault.hypervector_transform
├── Depends on:
│   ├── genomevault.core (constants, exceptions)
│   ├── genomevault.config (paths)
│   └── genomevault.utils (logging)
│
genomevault.api
├── Depends on:
│   ├── genomevault.hypervector_transform (encoding)
│   ├── genomevault.core (constants, exceptions)
│   └── genomevault.utils (logging)
│
genomevault.hypervector (DEPRECATED)
├── Compatibility shims to:
│   └── genomevault.hypervector_transform
```

## Best Practices

1. **Always use absolute imports** from the genomevault package
2. **Prefer hypervector_transform** over hypervector module
3. **Use config.paths** for all file path operations
4. **Import from core.constants** for system constants
5. **Check deprecation warnings** in development and testing

## Testing

When writing tests:
- Use the new module paths for new tests
- Existing tests using old paths will continue to work with warnings
- Run tests with `-W error::DeprecationWarning` to catch deprecated usage

```bash
# Run tests and treat deprecation warnings as errors
pytest -W error::DeprecationWarning

# Run tests showing deprecation warnings
pytest -W default::DeprecationWarning
```

## Questions?

For questions about module organization or migration:
1. Check this document first
2. Look for deprecation warnings in console output
3. Review the codebase examples in tests/
4. Open an issue for clarification

---
*Last Updated: 2025-08-09*
*Version: 1.0.0*
