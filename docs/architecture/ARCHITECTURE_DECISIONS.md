# Architecture Decisions

## Overview

This document records significant architectural decisions made in the GenomeVault project. Each decision includes context, the decision made, rationale, and implications.

## Decision Log

### ADR-001: Pydantic Version Strategy
**Date**: 2025-08-09
**Status**: Accepted
**Context**: The codebase currently uses Pydantic v1 syntax (`@validator` decorators, v1 BaseModel). Pydantic v2 was released with significant performance improvements but breaking API changes.

**Decision**: Pin to Pydantic v1 (>=1.10.0,<2.0.0)

**Rationale**:
1. **Stability**: Current code is tested and working with Pydantic v1
2. **FastAPI Compatibility**: FastAPI 0.103.x works well with Pydantic v1
3. **Migration Effort**: Moving to v2 would require:
   - Changing all `@validator` to `@field_validator`
   - Updating model configuration syntax
   - Potential changes to field definitions
   - Testing all API endpoints
4. **Time Constraints**: Focus on stabilization rather than migration

**Consequences**:
- ✅ No code changes required
- ✅ Stable, proven combination
- ✅ All existing tests continue to pass
- ❌ Missing Pydantic v2 performance improvements (2-50x faster)
- ❌ Will need migration eventually for long-term support

**Migration Path** (Future):
When ready to migrate to Pydantic v2:
1. Update to FastAPI >=0.100.0 (full v2 support)
2. Use `bump-pydantic` tool for automated migration
3. Manual review of custom validators
4. Comprehensive testing of all API endpoints

---

### ADR-002: Dependency Management Strategy
**Date**: 2025-08-09
**Status**: Accepted
**Context**: Dependencies were split between requirements.txt and pyproject.toml, making version management difficult.

**Decision**: Centralize all dependencies in pyproject.toml with pip-tools for locking

**Implementation**:
1. All direct dependencies in `pyproject.toml`
2. Use `requirements.in` and `requirements-dev.in` as source files
3. Generate locked files with `pip-compile`
4. Commit both `.in` and `.txt` files

**Rationale**:
- Single source of truth for dependencies
- Reproducible builds with locked versions
- Clear separation of direct vs transitive dependencies
- Compatible with modern Python packaging

**Workflow**:
```bash
# Update dependencies
pip-compile requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt

# Upgrade dependencies
pip-compile --upgrade requirements.in -o requirements.txt
```

---

### ADR-003: Experimental Features Isolation
**Date**: 2025-08-09
**Status**: Accepted
**Context**: Unstable research features (KAN networks, advanced PIR, experimental ZK) were mixed with production code.

**Decision**: Create `genomevault.experimental` package with clear warnings

**Implementation**:
- Move unstable features to `experimental/` subdirectory
- Add FutureWarning on import
- Require explicit opt-in via environment variable for high-risk features
- Maintain compatibility shims with deprecation warnings

**Rationale**:
- Clear separation of stable vs experimental
- Prevents accidental production use
- Allows research to continue without affecting stability
- Provides clear migration path

---

### ADR-004: Progressive Type Checking
**Date**: 2025-08-09
**Status**: Accepted
**Context**: Large codebase with inconsistent type hints. Full strict typing would generate thousands of errors.

**Decision**: Implement progressive mypy configuration by module

**Implementation**:
```toml
# Strict for core modules
[[tool.mypy.overrides]]
module = "genomevault.core.*"
strict = true

# Gradual for others
[[tool.mypy.overrides]]
module = "genomevault.hypervector.*"
ignore_errors = true  # Will be migrated later
```

**Rationale**:
- Immediate type safety for critical modules
- Gradual migration prevents disruption
- Clear tracking of progress
- Allows incremental improvement

**Migration Schedule**:
1. Phase 1: Core modules (completed)
2. Phase 2: API modules (in progress)
3. Phase 3: Hypervector modules (Q1 2025)
4. Phase 4: ZK/PIR modules (Q2 2025)

---

### ADR-005: Python Version Support
**Date**: 2025-08-09
**Status**: Accepted
**Context**: Python 3.9 approaches EOL (Oct 2025), but 3.10+ adoption not universal.

**Decision**: Support Python 3.10+ (requires-python = ">=3.10")

**Rationale**:
- Python 3.10 introduced important features (pattern matching, better errors)
- Ubuntu 22.04 LTS ships with Python 3.10
- Allows use of modern type hints (union with |)
- 3.9 EOL approaching

**Impact**:
- Can use `str | None` instead of `Optional[str]`
- Better error messages
- Performance improvements
- Structural pattern matching available

---

## Decision Template

### ADR-XXX: [Decision Title]
**Date**: YYYY-MM-DD
**Status**: [Proposed|Accepted|Deprecated|Superseded]
**Context**: [What is the issue we're addressing?]

**Decision**: [What have we decided?]

**Rationale**: [Why did we make this decision?]

**Consequences**: [What are the positive and negative outcomes?]

**Alternatives Considered**: [What other options were evaluated?]

---

## Review Schedule

Architecture decisions should be reviewed:
- Quarterly for active decisions
- When major version changes occur
- When external dependencies change significantly
- When team composition changes

Last Review: 2025-08-09
Next Review: 2025-11-09
