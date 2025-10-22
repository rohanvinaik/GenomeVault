# Dependency Management Guide

## Overview

GenomeVault uses a centralized dependency management approach with `pyproject.toml` as the single source of truth and `pip-tools` for generating locked requirement files.

## Quick Start

### Installing Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install with optional features
pip install -e ".[ml]"        # Machine learning dependencies
pip install -e ".[zk]"        # Zero-knowledge proof dependencies
pip install -e ".[nanopore]"  # Nanopore sequencing support
pip install -e ".[full]"      # All optional dependencies
```

## Dependency Structure

```
pyproject.toml          # Source of truth for all dependencies
requirements.in         # Core dependencies (generated from pyproject.toml)
requirements.txt        # Locked core dependencies (pip-compile output)
requirements-dev.in     # Development dependencies
requirements-dev.txt    # Locked dev dependencies (pip-compile output)
```

## Managing Dependencies

### Adding a New Dependency

1. **Core dependency**: Add to `dependencies` in `pyproject.toml`
2. **Dev dependency**: Add to `[project.optional-dependencies.dev]`
3. **Optional feature**: Add to appropriate optional group (ml, zk, etc.)

Then regenerate the requirements files:

```bash
# Update requirements.in from pyproject.toml
# (Manual step - copy dependencies from pyproject.toml)

# Regenerate locked files
pip-compile requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt
```

### Updating Dependencies

```bash
# Update all dependencies to latest compatible versions
pip-compile --upgrade requirements.in -o requirements.txt
pip-compile --upgrade requirements-dev.in -o requirements-dev.txt

# Update specific package
pip-compile --upgrade-package fastapi requirements.in -o requirements.txt
```

### Syncing Your Environment

```bash
# Ensure your environment matches requirements exactly
pip-sync requirements.txt         # Core only
pip-sync requirements-dev.txt     # Core + dev
```

## Version Pinning Strategy

### Core Dependencies
- **FastAPI**: Pinned to 0.103.x for Pydantic v1 compatibility
- **Pydantic**: Pinned to v1 (>=1.10.0,<2.0.0) - see ADR-001
- **NumPy**: Pinned to <2.0.0 to avoid breaking changes
- **Pandas**: Minimum 2.0.0 for modern features

### Development Dependencies
- More flexible versioning for dev tools
- Minimum versions specified for features we use
- Regular updates encouraged

## Pydantic v1 vs v2

**Current Status**: Pinned to Pydantic v1

**Rationale** (see ARCHITECTURE_DECISIONS.md):
- Existing code uses v1 syntax (`@validator`)
- FastAPI 0.103.x fully compatible with v1
- Migration to v2 requires significant code changes

**Future Migration Path**:
1. Update FastAPI to >=0.100.0
2. Run `bump-pydantic` migration tool
3. Update validators to v2 syntax
4. Test all API endpoints

## Docker Considerations

The Dockerfile uses multi-stage builds:

```dockerfile
# Install core dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development image includes dev dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
```

## CI/CD Integration

GitHub Actions should use locked requirements:

```yaml
- name: Install dependencies
  run: |
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
```

## Troubleshooting

### pip-compile Issues

If pip-compile fails with path errors:
```bash
# Use absolute paths
pip-compile /full/path/to/requirements.in -o requirements.txt
```

### Version Conflicts

Check for conflicts:
```bash
pip check
```

Resolve by adjusting version constraints in pyproject.toml.

### Missing Dependencies

If imports fail after installation:
```bash
# Verify installation
pip list | grep package-name

# Reinstall with verbose output
pip install -r requirements.txt -v
```

## Security Updates

Regular security audit:
```bash
# Check for security vulnerabilities
pip audit -r requirements.txt

# Or using bandit (included in dev dependencies)
bandit -r genomevault/
```

## Best Practices

1. **Always commit both .in and .txt files** - .in for source, .txt for reproducibility
2. **Update regularly** - Weekly for dev dependencies, monthly for core
3. **Test after updates** - Run full test suite after dependency updates
4. **Document breaking changes** - Note in CHANGELOG if updates affect users
5. **Use virtual environments** - Isolate project dependencies

## FAQ

**Q: Why not just use pyproject.toml directly?**
A: pip-compile generates locked files with exact versions for reproducible builds.

**Q: When should I run pip-compile?**
A: After modifying dependencies in pyproject.toml or requirements.in files.

**Q: Can I edit requirements.txt directly?**
A: No, it's auto-generated. Edit requirements.in instead.

**Q: How do I know which packages are direct vs transitive dependencies?**
A: Check requirements.in for direct, requirements.txt shows all with comments.
