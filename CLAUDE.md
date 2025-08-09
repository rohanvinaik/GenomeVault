# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GenomeVault is a privacy-preserving genomic computing platform that uses hyperdimensional computing (HDC), Kolmogorov-Arnold Networks (KAN), zero-knowledge proofs, and federated learning to enable secure genomic data analysis. The system achieves 50-100× compression while maintaining privacy and interpretability.

## Key Architecture Components

### Core Modules
- `genomevault/hypervector/` - Hyperdimensional encoding and operations for privacy-preserving genomic vectors
- `genomevault/kan/` - Kolmogorov-Arnold Network implementation for interpretable compression
- `genomevault/zk_proofs/` - Zero-knowledge proof generation and verification
- `genomevault/pir/` - Private Information Retrieval implementations
- `genomevault/federated/` - Federated learning infrastructure
- `genomevault/nanopore/` - Real-time nanopore sequencing support
- `genomevault/blockchain/` - Blockchain governance and decentralized control

### Important Design Patterns
- HD encoding transforms genomic variants into high-dimensional vectors (typically 10,000-100,000 dimensions)
- KAN-HD hybrid architecture combines spline functions with HD vectors for interpretable compression
- Hamming distance operations are optimized using lookup tables (LUTs)
- Privacy is maintained through mathematical guarantees, not just encryption

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -e .                    # Basic installation
pip install -e ".[dev]"             # Development tools
pip install -e ".[ml,zk,nanopore]"  # With ML, ZK proofs, and nanopore support
pip install -e ".[full]"            # All optional dependencies
```

### Common Development Tasks
```bash
# Run tests
pytest                              # Run all tests
pytest tests/test_hypervector.py    # Run specific test file
pytest -k "test_encoding"           # Run tests matching pattern
pytest -v --cov=genomevault         # Run with coverage

# Linting and formatting
ruff check .                        # Run linter
ruff format .                       # Format code
mypy genomevault                    # Type checking

# Using Makefile shortcuts
make test                           # Run test suite
make lint                           # Run linting
make fmt                            # Format code (if black/isort installed)

# Docker operations
make build                          # Build Docker image
make run                            # Run API in foreground
make up                             # Start API detached
make down                           # Stop services
```

### Running Benchmarks
```bash
python scripts/bench.py             # General benchmarks
python scripts/bench_hdc.py         # HD computing benchmarks
python scripts/bench_pir.py         # PIR benchmarks
python scripts/generate_perf_report.py  # Generate performance report
```

### Code Analysis
```bash
python scripts/analyze_complexity.py    # Analyze code complexity
radon cc genomevault -s                # Cyclomatic complexity
```

## Project Configuration

### Key Configuration Files
- `pyproject.toml` - Main project configuration, dependencies, and tool settings
- `.ruff.toml` - Linting and formatting rules (line length: 100, Python 3.11+)
- `.mypy.ini` - Type checking configuration (strict mode enabled)
- `.env.example` - Environment variables template

### Dependency Groups
- `dev` - Development tools (ruff, mypy, pytest, radon)
- `ml` - Machine learning (torch, scikit-learn, numpy)
- `zk` - Zero-knowledge proofs (pysnark)
- `nanopore` - Nanopore sequencing (ont-fast5-api, pyslow5)
- `gpu` - GPU acceleration (cupy)

## Testing Strategy

Tests are organized by module in the `tests/` directory. Key test areas:
- Hypervector encoding correctness
- Privacy preservation guarantees
- Compression/decompression accuracy
- Zero-knowledge proof verification
- Federated learning convergence

Run tests for specific modules:
```bash
pytest tests/test_hypervector_encoding.py
pytest tests/test_zk_proofs.py
pytest tests/test_federated_learning.py
```

## Common Issues and Solutions

### Import Errors
The codebase has some import issues (e.g., missing `ProjectionError`). When encountering import errors:
1. Check if the exception/class exists in the target module
2. Create it if missing or update the import path
3. Common missing imports are in `genomevault/core/exceptions.py`

### Performance Optimization
- HD operations use Hamming LUTs for 10-20× speedup
- Batch operations are preferred over individual computations
- GPU acceleration available via cupy for large-scale operations

## CLI Usage
The project provides two CLI entry points:
```bash
genomevault [command]  # Full command
gv [command]          # Shorthand
```

## Working with Genomic Data

### Input Formats
- VCF files for variant data
- FASTA/FASTQ for sequence data
- BED files for genomic regions
- Custom SNP panels via JSON

### Accuracy Modes
The system supports different accuracy levels via SNP panel selection:
- `OFF` - Basic screening (90-95% single run)
- `COMMON` - Epidemiology (95-98% single run)
- `CLINICAL` - Clinical diagnostics (98-99.5% single run)
- `KAN-HD` - Regulatory approval (99%+ single run)

Multiple runs exponentially increase accuracy due to mathematical error convergence.

## Security and Privacy

- Never commit sensitive data or API keys
- All genomic data operations maintain privacy through HD encoding
- Zero-knowledge proofs enable verification without data exposure
- Federated learning keeps data distributed and private

## Branch Strategy

Current branch workflow suggests:
- `main` - Main development branch (use for PRs)
- `chore/lint-sweep` - Current cleanup branch
- Feature branches for new development

## Note on Current State

The repository is undergoing a linting cleanup (see git status). There are:
- Deleted backup files in `.cleanup_backups/`
- Active refactoring to improve code quality
- Some import errors that need resolution
