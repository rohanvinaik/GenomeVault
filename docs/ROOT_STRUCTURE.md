# Root Directory Structure

This document describes the organization of the GenomeVault root directory following best practices.

## Directory Organization

### Core Directories

- **`genomevault/`** - Main package source code
- **`tests/`** - Test suite organized by category
- **`docs/`** - Documentation including API docs and reports
- **`scripts/`** - Utility scripts for development and deployment
- **`tools/`** - Development tools, linters, and code analysis
- **`examples/`** - Example usage and demonstrations
- **`benchmarks/`** - Performance benchmarking code

### Infrastructure

- **`deployment/`** - Kubernetes and cloud deployment configs
- **`docker/`** - Docker configurations and images
- **`devops/`** - DevOps scripts and configurations
- **`deploy/`** - Deployment manifests (Grafana, Prometheus, K8s)

### Configuration

- **`config/`** - Application configuration files
- **`etc/`** - System configuration and policies
- **`keys/`** - Cryptographic keys (gitignored in production)
- **`schemas/`** - JSON schemas for validation
- **`typings/`** - Type stubs for external libraries

### Data and Assets

- **`data/`** - Data directories (cache, input, output)
- **`assets/`** - Static assets and demo materials
- **`logs/`** - Application logs (gitignored)

### Specialized

- **`blockchain/`** - Blockchain contracts and scripts
- **`clinical_validation/`** - Clinical validation tools
- **`devtools/`** - Development utilities and helpers

## Root Files

### Essential Configuration

- **`pyproject.toml`** - Python project configuration
- **`setup.py`** - Package setup script
- **`setup.cfg`** - Setup configuration
- **`requirements.txt`** - Python dependencies
- **`requirements-*.txt`** - Specialized dependency lists

### Build and Test

- **`Makefile`** - Build automation
- **`pytest.ini`** - Pytest configuration
- **`mypy.ini`** - Type checking configuration
- **`.ruff.toml`** - Linting configuration

### Docker

- **`Dockerfile`** - Main application container
- **`Dockerfile.pir`** - PIR service container
- **`docker-compose.yml`** - Production compose
- **`docker-compose.dev.yml`** - Development compose
- **`docker-compose.obsv.yml`** - Observability stack

### Documentation

- **`README.md`** - Project overview and quickstart
- **`CONTRIBUTING.md`** - Contribution guidelines
- **`LICENSE`** - Software license
- **`SECURITY.md`** - Security policies
- **`INSTALL.md`** - Installation instructions
- **`CLAUDE.md`** - Claude Code integration guide
- **`VERSION.md`** - Version information

### Shell Scripts

- **`setup_dev_environment.sh`** - Development setup
- **`run_tests.sh`** - Test runner
- **`run_mvp_tests.sh`** - MVP test suite
- **`test_and_validate.sh`** - Full validation
- **`pre_push_validation.sh`** - Pre-push checks
- **`verify_mvp.sh`** - MVP verification

## Cleanup History

The root directory was cleaned up to follow best practices:

### Moved to `tools/`
- 19 utility scripts for linting, validation, and fixes
- One-off automation scripts

### Moved to `docs/reports/`
- 6 implementation reports and summaries
- Validation and audit reports

### Deleted
- 23 temporary test files
- 5 temporary directories (htmlcov, task outputs, caches)
- Task-specific scripts and outputs

### Result
- Reduced root directory from ~100+ files to ~55 essential items
- Clear separation of concerns
- Professional project structure
- All functionality preserved

## Maintenance

To keep the root directory clean:

1. Place new scripts in `tools/` or `scripts/`
2. Put documentation in `docs/`
3. Keep test files in `tests/`
4. Use `.gitignore` patterns to exclude generated files
5. Run cleanup scripts periodically

## File Placement Guide

| File Type | Location |
|-----------|----------|
| Python utility scripts | `tools/` |
| Shell scripts | `scripts/` |
| Test files | `tests/` |
| Documentation | `docs/` |
| Implementation reports | `docs/reports/` |
| Examples | `examples/` |
| Config files | Root or `config/` |
| Generated files | Gitignored |
