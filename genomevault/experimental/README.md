# Experimental Features

This directory contains experimental features that are under active development. These features should be considered unstable and their APIs may change without notice.

## ⚠️ Warning

**These features are EXPERIMENTAL and should NOT be used in production environments without careful consideration:**

- APIs may change without deprecation notices
- Performance characteristics are not guaranteed
- Security properties may not be fully verified
- Features may be removed in future versions

## Enabling Experimental Features

By default, some experimental features require explicit opt-in. To enable them:

```bash
export GENOMEVAULT_EXPERIMENTAL=true
```

Or in Python:
```python
import os
os.environ["GENOMEVAULT_EXPERIMENTAL"] = "true"
```

## Current Experimental Modules

### 1. KAN Networks (`experimental.kan`)
**Status**: Research prototype
**Description**: Kolmogorov-Arnold Networks for genomic data compression
**Risks**:
- Unoptimized performance
- Memory intensive operations
- API instability

**Usage**:
```python
from genomevault.experimental.kan import KANHDEncoder
# Warning will be displayed about experimental status
```

### 2. Advanced PIR (`experimental.pir_advanced`)
**Status**: Research implementation
**Description**: Information-theoretic PIR protocols
**Risks**:
- Not cryptographically audited
- Performance not optimized
- Network overhead not characterized

**Usage**:
```python
from genomevault.experimental.pir_advanced import InformationTheoreticPIR
# Warning about security audit status
```

### 3. ZK Circuits (`experimental.zk_circuits`)
**Status**: Proof of concept
**Description**: Advanced zero-knowledge proof circuits
**Risks**:
- Security properties not formally verified
- May contain vulnerabilities
- Not suitable for sensitive data

**Usage**:
```python
# Requires explicit opt-in
os.environ["GENOMEVAULT_EXPERIMENTAL"] = "true"
from genomevault.experimental.zk_circuits import RecursiveSNARKProver
```

## Development Guidelines

When working with experimental features:

1. **Always check warnings** - Pay attention to FutureWarning messages
2. **Test thoroughly** - Experimental features may have edge cases
3. **Monitor performance** - These features are not optimized
4. **Report issues** - Help improve experimental features
5. **Don't use in production** - Unless you fully understand the risks

## Migration Path

Experimental features follow this lifecycle:

1. **Experimental** (current) - Unstable, may change
2. **Preview** - API stabilizing, performance improving
3. **Stable** - Ready for production use
4. **Deprecated** - Being phased out

Features typically remain experimental for 2-3 release cycles before promotion to stable (or removal).

## Contributing

To add new experimental features:

1. Create a new subdirectory under `experimental/`
2. Add clear warnings in `__init__.py`
3. Document risks and limitations
4. Include examples and tests
5. Update this README

## Feature Requests

Have ideas for experimental features? Please open an issue with:
- Use case description
- Performance requirements
- Security considerations
- API design proposal

## Feedback

Your feedback on experimental features helps us decide what to stabilize:
- Performance benchmarks
- API usability
- Bug reports
- Feature requests

---

**Remember**: Experimental means experimental. Use at your own risk!
