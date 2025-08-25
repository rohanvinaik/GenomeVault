# Migration Guide - GenomeVault Clean Slate Refactor

## Overview

This guide helps existing GenomeVault users migrate to the refactored codebase completed in Phase 10. The refactoring improves code organization, security, and maintainability while preserving most APIs.

## Breaking Changes

### 1. Module Path Changes

#### Hypervector Modules
**Old:**
```python
from genomevault.hypervector.encoder import HDEncoder
```

**New:**
```python
from genomevault.hypervector_transform.encoding import HDEncoder
```

**Compatibility:** Shims exist at old paths with deprecation warnings (will be removed in v2.0)

#### Experimental Features
**Old:**
```python
from genomevault.kan import KANHDEncoder
from genomevault.pir.advanced import InformationTheoreticPIR
```

**New:**
```python
from genomevault.experimental.kan import KANHDEncoder
from genomevault.experimental.pir_advanced import InformationTheoreticPIR
```

**Note:** Experimental imports now trigger FutureWarning

### 2. Blockchain Contracts

**Old Structure:**
- Multiple copies of contracts in different locations
- Inconsistent versions

**New Structure:**
- Single source: `genomevault/blockchain/contracts/`
- Symlinks for build compatibility

**Impact:** Update deployment scripts to use new paths

### 3. Dependencies

**Old:**
- Mixed requirements.txt and pyproject.toml
- No version locking

**New:**
- All dependencies in pyproject.toml
- Locked versions via pip-compile
- Pydantic pinned to v1.x

**Action Required:**
```bash
pip install -r requirements.txt  # Use locked versions
```

### 4. Development Tools

**Old Location:** `tools/`
**New Location:** `devtools/`

Update any scripts that reference tools:
```bash
# Old
python tools/debug_genomevault.py

# New
python devtools/debug_genomevault.py
```

## Step-by-Step Migration

### Step 1: Update Dependencies

```bash
# Backup current environment
pip freeze > old_requirements.txt

# Install new dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### Step 2: Update Imports

Run the migration script to identify deprecated imports:
```bash
python devtools/check_deprecated_imports.py
```

Update your code:
```python
# Add at top of files using deprecated imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Plan to update imports before v2.0
```

### Step 3: Handle Experimental Features

If using experimental features:

```python
# Required for some experimental features
import os
os.environ["GENOMEVAULT_EXPERIMENTAL"] = "true"

# Import with warning suppression
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from genomevault.experimental.kan import KANHDEncoder
```

### Step 4: Update Configuration

Check for hardcoded paths:
```bash
grep -r "genomevault/hypervector/" --include="*.py"
grep -r "tools/" --include="*.sh"
```

### Step 5: Test Your Code

```bash
# Run your tests
pytest your_tests/

# Check for type errors
mypy your_code/

# Security scan
bandit -r your_code/
```

## API Changes

### FastAPI Endpoints

No changes to existing endpoints. New health endpoint added:
- `GET /healthz` - Health check with component status

### Python APIs

Most APIs unchanged. Notable updates:

#### HDEncoder
- Moved to `hypervector_transform.encoding`
- API unchanged
- Performance improvements

#### Training Attestation
- Import path unchanged
- New verification methods added

## Configuration Changes

### Environment Variables

New variables:
- `GENOMEVAULT_EXPERIMENTAL=true` - Enable experimental features
- `GENOMEVAULT_LOG_LEVEL` - Set logging level

Deprecated:
- None

### Settings Files

No changes to settings structure.

## Database Schema

No database schema changes in this refactor.

## Common Issues and Solutions

### Issue: ImportError for old module paths

**Solution:**
```python
try:
    from genomevault.hypervector.encoder import HDEncoder
except ImportError:
    from genomevault.hypervector_transform.encoding import HDEncoder
```

### Issue: Pydantic v2 incompatibility

**Solution:**
Project is pinned to Pydantic v1. Don't upgrade to v2 yet.

### Issue: Missing experimental features

**Solution:**
```python
import os
os.environ["GENOMEVAULT_EXPERIMENTAL"] = "true"
# Now import experimental features
```

### Issue: Contract compilation fails

**Solution:**
```bash
cd blockchain/
rm -rf cache artifacts
npm install
npx hardhat compile
```

## Testing Your Migration

### Validation Checklist

- [ ] All imports resolve without errors
- [ ] API endpoints respond correctly
- [ ] Blockchain contracts compile
- [ ] Tests pass (if you have them)
- [ ] No security warnings from bandit
- [ ] Type checking passes (mypy)

### Smoke Test Script

```python
#!/usr/bin/env python3
"""Smoke test for migration validation."""

def test_imports():
    """Test critical imports work."""
    try:
        from genomevault.core import hdencoder
        from genomevault.api.app import app
        from genomevault.blockchain.contracts.training_attestation import create_attestation
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_api():
    """Test API starts."""
    import requests
    import subprocess
    import time

    # Start API
    proc = subprocess.Popen(
        ["uvicorn", "genomevault.api.app:app", "--port", "8001"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(3)

    try:
        resp = requests.get("http://localhost:8001/healthz")
        assert resp.status_code == 200
        print("✅ API health check passed")
        return True
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    finally:
        proc.terminate()

if __name__ == "__main__":
    import sys

    success = all([
        test_imports(),
        test_api(),
    ])

    if success:
        print("\n✅ Migration validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ Migration validation FAILED")
        sys.exit(1)
```

## Rollback Plan

If issues arise:

1. **Restore old code:**
   ```bash
   git checkout <previous-commit>
   ```

2. **Restore dependencies:**
   ```bash
   pip install -r old_requirements.txt
   ```

3. **Report issues:**
   Create issue at: https://github.com/genomevault/genomevault/issues

## Timeline

### Deprecation Schedule

- **Now**: Compatibility shims active, warnings displayed
- **v1.5** (Q2 2025): Warnings become errors for some paths
- **v2.0** (Q3 2025): Compatibility shims removed

### Support

- Old module paths: Supported until v2.0
- Pydantic v1: Supported until v2.0
- Python 3.9: Deprecated (use 3.10+)

## Getting Help

### Resources

1. **Documentation**: See `/docs` directory
2. **Examples**: Updated examples in `/examples`
3. **Issues**: GitHub issues for questions
4. **Validation**: Run `VALIDATION_REPORT.md` checks

### Contact

For migration assistance:
- Open GitHub issue with `[MIGRATION]` tag
- Include error messages and environment details
- Reference this guide

## Benefits After Migration

✅ **Improved Performance**
- Faster imports
- Reduced memory usage
- Better caching

✅ **Better Security**
- Isolated experimental features
- Security scanning integrated
- Type safety improvements

✅ **Cleaner Code**
- Organized module structure
- Clear separation of concerns
- Better documentation

✅ **Future-Proof**
- Prepared for Pydantic v2
- Ready for Python 3.12+
- Extensible architecture

---

*Last Updated: 2025-08-09*
*Version: 1.0.0*
