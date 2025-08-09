# GenomeVault Development Tools

This directory contains development and maintenance tools for the GenomeVault project.

## Quick Start

For new developers, run the setup script to configure your development environment:

```bash
python devtools/setup_dev.py
```

This will:
- Create a virtual environment
- Install all dependencies with extras
- Set up pre-commit hooks
- Run initial tests to verify installation

## Tool Categories

### Setup & Configuration
- `setup_dev.py` - One-command development environment setup
- `assert_no_stubs.py` - Check for stub implementations in critical paths

### Code Quality & Linting
- `apply_lint_fixes.py` - Apply automated linting fixes
- `check_lint_status.py` - Check current linting status
- `run_hdc_linters.py` - Run HDC-specific linting checks
- `validate_lint_clean.py` - Validate code is lint-clean

### Performance & Benchmarking
- `bench.py` - General benchmarking suite
- `bench_hdc.py` - HDC-specific benchmarks
- `bench_pir.py` - PIR benchmarks
- `generate_perf_report.py` - Generate performance reports

### Code Analysis
- `analyze_complexity.py` - Analyze code complexity metrics
- `dependency_analysis.py` - Analyze package dependencies
- `diagnose_imports.py` - Diagnose import issues
- `trace_import_failure.py` - Trace specific import failures

### Security & Compliance
- `security_check.py` - Run security checks
- `pii_scan.py` - Scan for PII/PHI data

### Cleanup & Maintenance
- `comprehensive_cleanup.py` - Comprehensive codebase cleanup
- `genomevault_cleanup.py` - GenomeVault-specific cleanup
- `fix_python_compatibility.py` - Fix Python version compatibility

### Testing & Validation
- `clinical_eval_run.py` - Run clinical evaluation tests
- `validate_hdc_implementation.py` - Validate HDC implementation
- `pre_push_checklist.py` - Pre-push validation checklist

### Code Transformation
- `convert_print_to_logging.py` - Convert print statements to logging
- `codemods/` - Directory containing code modification scripts
  - `fix_exceptions.py` - Fix exception handling
  - `replace_prints_with_logging.py` - Replace prints with logging

## Usage Examples

### Setting Up Development Environment
```bash
# Full setup with default virtual environment
python devtools/setup_dev.py

# Custom virtual environment name
python devtools/setup_dev.py --venv-name myenv

# Skip initial tests
python devtools/setup_dev.py --skip-tests
```

### Running Benchmarks
```bash
# General benchmarks
python devtools/bench.py

# HDC-specific benchmarks
python devtools/bench_hdc.py

# Generate performance report
python devtools/generate_perf_report.py
```

### Code Quality Checks
```bash
# Check lint status
python devtools/check_lint_status.py

# Apply automated fixes
python devtools/apply_lint_fixes.py

# Validate everything is clean
python devtools/validate_lint_clean.py
```

### Security Scans
```bash
# Run security checks
python devtools/security_check.py

# Scan for PII/PHI
python devtools/pii_scan.py
```

## Notes

- Most tools should be run from the project root directory
- Tools may require additional dependencies installed via `pip install -e ".[dev]"`
- Some tools are legacy and may need updates for current codebase structure

## Contributing

When adding new development tools:
1. Place them in this `devtools/` directory
2. Add appropriate documentation to this README
3. Include proper error handling and logging
4. Make scripts executable with `chmod +x` if appropriate
