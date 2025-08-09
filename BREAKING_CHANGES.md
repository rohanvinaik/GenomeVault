# Breaking Changes - Clean Slate Refactor

## Version: 1.0.0-refactor
## Date: 2025-08-09

This document lists all breaking changes introduced during the clean-slate refactoring (Phases 1-10).

## Critical Breaking Changes

### 1. Module Relocations

#### ❗ Hypervector Module Split
- **BREAKING**: `genomevault.hypervector` split into multiple modules
- **Old**: `genomevault.hypervector.encoder`
- **New**: `genomevault.hypervector_transform.encoding`
- **Migration**: Compatibility shims available until v2.0
- **Impact**: High - Core functionality

#### ❗ Experimental Features Isolation
- **BREAKING**: KAN, advanced PIR, and experimental ZK moved
- **Old**: `genomevault.kan`, `genomevault.pir.advanced`
- **New**: `genomevault.experimental.kan`, `genomevault.experimental.pir_advanced`
- **Migration**: Update imports, handle FutureWarning
- **Impact**: Medium - Research features only

### 2. Dependency Changes

#### ❗ Pydantic Version Lock
- **BREAKING**: Locked to Pydantic v1.x
- **Old**: Any Pydantic version
- **New**: `pydantic>=1.10.0,<2.0.0`
- **Migration**: Do not upgrade to Pydantic v2
- **Impact**: High - API models affected

#### ❗ Python Version Requirement
- **BREAKING**: Now requires Python 3.10+
- **Old**: Python 3.9+
- **New**: Python 3.10+
- **Migration**: Upgrade Python installation
- **Impact**: High - Runtime requirement

### 3. Development Tools

#### ⚠️ Tools Directory Renamed
- **BREAKING**: Development tools relocated
- **Old**: `tools/`
- **New**: `devtools/`
- **Migration**: Update script paths
- **Impact**: Low - Development only

### 4. Blockchain Contracts

#### ⚠️ Contract Consolidation
- **BREAKING**: Contracts moved to single location
- **Old**: Multiple locations with duplicates
- **New**: `genomevault/blockchain/contracts/`
- **Migration**: Update deployment scripts
- **Impact**: Low - Deployment only

## API Breaking Changes

### REST API
- **No breaking changes** - All existing endpoints preserved
- **Addition**: New `/healthz` endpoint added

### Python API

#### Class Renames
None - All public classes maintain original names

#### Method Signature Changes
- **`HDEncoder.__init__`**: Added optional `compression_tier` parameter (backward compatible)
- **`create_attestation`**: Added optional `metadata` parameter (backward compatible)

#### Removed Functions
- **`genomevault.utils.deprecated_func`**: Removed without replacement
- **`genomevault.legacy.*`**: All legacy modules removed

## Configuration Breaking Changes

### Environment Variables
- **New Required**: None
- **New Optional**: `GENOMEVAULT_EXPERIMENTAL` for experimental features
- **Deprecated**: None
- **Removed**: None

### Configuration Files
- **No breaking changes** to configuration file formats

## Import Path Changes Summary

| Old Import | New Import | Compatibility |
|------------|------------|---------------|
| `genomevault.hypervector.encoder` | `genomevault.hypervector_transform.encoding` | Shim until v2.0 |
| `genomevault.kan` | `genomevault.experimental.kan` | Warning, shim until v2.0 |
| `genomevault.pir.advanced` | `genomevault.experimental.pir_advanced` | Warning, requires opt-in |
| `genomevault.zk.experimental` | `genomevault.experimental.zk_circuits` | No shim, direct update required |

## Behavioral Changes

### Experimental Features
- **Change**: Now require explicit opt-in
- **Old**: Import directly
- **New**: Set `GENOMEVAULT_EXPERIMENTAL=true` for some features
- **Impact**: Prevents accidental use in production

### Error Handling
- **Change**: Stricter validation in API endpoints
- **Old**: Some endpoints accepted malformed data
- **New**: All endpoints validate input strictly
- **Impact**: Better error messages but may reject previously accepted input

### Logging
- **Change**: Centralized logging configuration
- **Old**: Module-specific loggers
- **New**: Hierarchical logger from `genomevault.utils.logging`
- **Impact**: More consistent log formatting

## Removed Features

### Completely Removed
1. **Legacy compatibility module** (`genomevault.legacy`)
2. **Deprecated utility functions** in `genomevault.utils.deprecated`
3. **Old configuration loader** (`genomevault.config.old_loader`)

### Moved to Experimental
1. **KAN networks** - Now in experimental with warnings
2. **Advanced PIR protocols** - Requires explicit opt-in
3. **Incomplete ZK circuits** - Not production ready

## Database Changes
- **No database schema changes**

## File Format Changes
- **No file format changes**

## Network Protocol Changes
- **No protocol changes**

## Migration Priority

### Must Fix Immediately
1. Update Python to 3.10+ if still on 3.9
2. Update imports for removed modules
3. Pin Pydantic to v1.x

### Should Fix Soon
1. Update hypervector imports (shims available)
2. Move experimental feature usage behind flags
3. Update development tool paths

### Can Defer
1. Update to use new logging system
2. Optimize for new module structure
3. Remove deprecation warning suppressions

## Testing Your Code

After updating, verify:

```bash
# Check imports
python -c "from genomevault.core import hdencoder"

# Run type checking
mypy your_code/

# Check for deprecated imports
grep -r "genomevault.hypervector.encoder" .

# Test API compatibility
curl http://localhost:8000/healthz
```

## Rollback Instructions

If breaking changes cause issues:

1. **Git Rollback**:
   ```bash
   git checkout <commit-before-refactor>
   ```

2. **Dependency Rollback**:
   ```bash
   pip install -r old-requirements.txt
   ```

3. **Report Issues**:
   - File issue with `[BREAKING]` tag
   - Include full error messages
   - Specify which breaking change caused issue

## Support Timeline

| Feature | Deprecation Warning | Removal Date |
|---------|-------------------|--------------|
| Old hypervector paths | Now | v2.0 (Q3 2025) |
| Experimental shims | Now | v2.0 (Q3 2025) |
| Python 3.9 support | Now | v1.5 (Q2 2025) |
| Pydantic v1 support | v2.0 | v3.0 (2026) |

## Contact for Help

For breaking change assistance:
- GitHub Issues: Tag with `[BREAKING]`
- Migration Guide: See `MIGRATION_GUIDE.md`
- Documentation: Updated in `/docs`

---

*This document tracks breaking changes from the clean-slate refactor completed on 2025-08-09.*
