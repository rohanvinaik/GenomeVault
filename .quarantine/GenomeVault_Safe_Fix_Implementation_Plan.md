# GenomeVault Safe Fix Implementation Plan

Based on the comprehensive audit playbook, this document provides a structured implementation plan to fix all critical issues and establish a clean, working MVP baseline for GenomeVault.

## Implementation Status Overview

- **Total Python Files**: 479
- **Total Lines of Code**: 100,172
- **Files with Syntax Errors**: 12 (CRITICAL - blocking)
- **Files with Placeholders**: 5 (NotImplementedError)
- **Files with TODO/FIXME**: 24
- **Missing `__init__.py`**: 5 packages

## Phase 1: Critical Syntax Error Fixes (Priority 1)

### Files Requiring Immediate Syntax Fixes:

1. **devtools/trace_import_failure.py** (line 20)
   - Issue: Expected indented block after 'try' statement
   - Fix: Add proper exception handling block

2. **examples/minimal_verification.py** (line 33)
   - Issue: Expected indented block after 'if' statement
   - Fix: Add conditional logic implementation

3. **genomevault/local_processing/epigenetics.py** (line 188)
   - Issue: Unexpected indent
   - Fix: Correct indentation alignment

4. **genomevault/local_processing/proteomics.py** (line 208)
   - Issue: Unexpected indent
   - Fix: Correct indentation alignment

5. **genomevault/local_processing/transcriptomics.py** (line 126)
   - Issue: Unexpected indent
   - Fix: Correct indentation alignment

6. **genomevault/pir/server/enhanced_pir_server.py** (line 454)
   - Issue: Invalid syntax
   - Fix: Review and correct syntax error

7. **genomevault/zk_proofs/circuits/clinical/__init__.py** (line 3)
   - Issue: Invalid syntax
   - Fix: Correct module exports

8. **genomevault/zk_proofs/circuits/clinical_circuits.py** (line 4)
   - Issue: Unexpected indent
   - Fix: Correct indentation

9. **genomevault/zk_proofs/circuits/test_training_proof.py** (line 7)
   - Issue: Invalid syntax
   - Fix: Review test syntax

10. **genomevault/zk_proofs/prover.py** (line 440)
    - Issue: Invalid syntax
    - Fix: Correct proof generation logic

11. **lint_clean_implementation.py** (line 223)
    - Issue: Invalid syntax (assignment vs comparison)
    - Fix: Change '=' to '==' or ':='

12. **tests/test_hdc_pir_integration.py** (line 3)
    - Issue: Mismatched parentheses
    - Fix: Correct bracket matching

## Phase 2: Package Structure Fixes

### Add Missing `__init__.py` Files:

```bash
# Packages requiring __init__.py
- genomevault/clinical/calibration/
- genomevault/contracts/audit/
- genomevault/hypervector/
- genomevault/pir/benchmark/
- genomevault/zk_proofs/circuits/implementations/
```

## Phase 3: Replace Placeholders with MVP Implementations

### Files with NotImplementedError:

1. **genomevault/local_processing/epigenetics.py**
   - Implement basic epigenetic data processing
   - MVP: Return normalized feature matrix

2. **genomevault/local_processing/proteomics.py**
   - Implement protein data processing
   - MVP: Basic normalization pipeline

3. **genomevault/local_processing/transcriptomics.py**
   - Implement transcriptomic data processing
   - MVP: Expression matrix normalization

4. **genomevault/zk_proofs/circuits/clinical/__init__.py**
   - Export clinical circuit implementations
   - MVP: Basic circuit definitions

5. **genomevault/zk_proofs/prover.py**
   - Implement proof generation
   - MVP: Mock proof generation with stable API

## Phase 4: Code Quality Improvements

### Replace print() with logging:
- 50+ files using print statements
- Implement structured logging with Python's logging module

### Remove debug code:
- Remove all `pdb.set_trace()` calls
- Remove `assert` statements in non-test code (13 files)

### Fix long lines:
- 16 files with lines >120 characters
- Apply automatic formatting with ruff

## Phase 5: MVP Contract Implementations

### Core Module Contracts:

#### 5.1 Local Processing (`genomevault/local_processing/*`)
```python
def process(dataset: pd.DataFrame | Path, config: dict) -> np.ndarray:
    """Return normalized numeric feature matrix (n_samples x n_features)."""
    # Load from Path if needed
    # Select numeric columns or raise ValueError
    # Standardize: (X - mean) / (std + 1e-9)
    # Return 2D np.ndarray with finite values
```

#### 5.2 HDC Module (`genomevault/hdc/*`)
```python
D = 10_000  # Hypervector dimension

def encode(X: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Encode features to hypervectors."""

def bundle(vectors: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Bundle multiple hypervectors."""

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute similarity between hypervectors."""
```

#### 5.3 ZK Proofs (`genomevault/zk_proofs/*`)
```python
def prove(payload: dict) -> dict:
    """Generate zero-knowledge proof."""
    return {"proof": "MOCK", "public": {"commitment": "deadbeef"}}

def verify(proof: dict) -> bool:
    """Verify zero-knowledge proof."""
    return "proof" in proof
```

## Phase 6: Testing Infrastructure

### Add Smoke Tests:
- One test file per major module
- Test basic shapes and types
- Validate MVP contracts

### Test Coverage Goals:
- Minimum 60% code coverage
- All public APIs tested
- Critical paths validated

## Phase 7: Tooling Configuration

### Ruff Configuration (`pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
extend-select = ["I", "UP", "B", "C4", "SIM", "T20"]
ignore = ["D", "ANN"]
target-version = "py311"

[tool.ruff.lint.isort]
known-first-party = ["genomevault"]
```

### MyPy Configuration (`mypy.ini`):
```ini
[mypy]
python_version = 3.11
ignore_missing_imports = True
warn_unused_ignores = True
no_implicit_optional = False
disallow_untyped_defs = False

[mypy-genomevault.local_processing.*]
disallow_untyped_defs = True
```

### Pre-commit Configuration (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.7
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
```

## Phase 8: CI/CD Setup

### GitHub Actions Workflow:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -e .[dev] ruff mypy pytest
      - run: ruff check .
      - run: mypy genomevault
      - run: pytest -q
```

## Implementation Checklist

### Immediate Actions (Phase 1-2):
- [ ] Fix all 12 syntax errors
- [ ] Add missing `__init__.py` files
- [ ] Verify imports work correctly

### MVP Implementation (Phase 3-5):
- [ ] Replace all NotImplementedError placeholders
- [ ] Implement MVP contracts for core modules
- [ ] Replace print() with logging
- [ ] Remove debug code

### Quality Assurance (Phase 6-7):
- [ ] Add smoke tests for all modules
- [ ] Configure ruff and mypy
- [ ] Set up pre-commit hooks
- [ ] Run full test suite

### Final Validation (Phase 8):
- [ ] All tests passing
- [ ] Zero ruff errors
- [ ] MyPy checks passing
- [ ] Documentation updated

## Success Criteria

### Green Baseline Achieved When:
1. **Zero syntax errors** - All Python files parse correctly
2. **All imports work** - No missing `__init__.py` files
3. **MVP implementations** - All placeholders replaced
4. **Tests passing** - Smoke tests validate basic functionality
5. **Linting clean** - Ruff reports zero errors
6. **Type checking** - MyPy passes on configured modules
7. **CI/CD green** - GitHub Actions workflow passing

## Execution Timeline

### Day 1: Critical Fixes
- Fix all syntax errors
- Add missing package files
- Basic import validation

### Day 2: MVP Implementation
- Replace placeholders
- Implement core contracts
- Add logging infrastructure

### Day 3: Testing & Quality
- Write smoke tests
- Configure tooling
- Run full validation

### Day 4: Documentation & CI
- Update documentation
- Set up CI/CD
- Final validation

## Risk Mitigation

### Backup Strategy:
- Create branch backup before changes
- Commit after each successful phase
- Test incrementally

### Rollback Plan:
- Git reset to last known good state
- Cherry-pick successful fixes
- Document issues encountered

## Notes for Implementation

1. **Start with syntax errors** - Nothing else works until these are fixed
2. **Test imports frequently** - Verify module structure after each change
3. **Use MVP implementations** - Don't over-engineer initial fixes
4. **Commit often** - Small, atomic commits for easy rollback
5. **Document changes** - Update README with implementation status

---

## Automated Implementation Script

For automated execution, use the following script:

```bash
#!/bin/bash
# GenomeVault Safe Fix Implementation Script

echo "Starting GenomeVault Safe Fix Implementation..."

# Phase 1: Fix syntax errors
echo "Phase 1: Fixing syntax errors..."
python fix_audit_issues.py

# Phase 2: Add missing __init__.py files
echo "Phase 2: Adding missing __init__.py files..."
python quick_fix_init_files.py

# Phase 3: Replace placeholders
echo "Phase 3: Implementing MVP placeholders..."
python genomevault_autofix.py

# Phase 4: Code quality
echo "Phase 4: Applying code quality fixes..."
ruff check --fix .
ruff format .

# Phase 5: Run tests
echo "Phase 5: Running tests..."
pytest -q

# Phase 6: Validate
echo "Phase 6: Final validation..."
python validate_audit_fixes.py

echo "Implementation complete!"
```

---

This implementation plan provides a structured approach to fixing all identified issues in the GenomeVault codebase, establishing a clean MVP baseline for future development.
