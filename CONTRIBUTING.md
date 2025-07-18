# Contributing to GenomeVault

Thank you for your interest in contributing to GenomeVault! We're excited to have you join our mission to revolutionize genomic data privacy and research.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inspiring community for all.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, please include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment details (OS, Python version, etc.)
- Any relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- A clear and descriptive title
- A detailed description of the proposed enhancement
- Any possible implementation approaches
- Why this enhancement would be useful

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. Ensure the test suite passes
4. Make sure your code follows our style guidelines
5. Issue that pull request!

## Development Process

### Setting Up Your Development Environment

```bash
# Clone your fork
git clone https://github.com/yourusername/GenomeVault.git
cd GenomeVault

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Style

- We use Black for Python code formatting
- Maximum line length is 100 characters
- Use meaningful variable names
- Add docstrings to all functions and classes
- Follow PEP 8 guidelines

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=genomevault

# Run specific test file
pytest tests/unit/test_hypervector.py
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Example:
```
Add hypervector compression for clinical tier

- Implement 300KB clinical tier compression
- Add unit tests for compression/decompression
- Update documentation with compression ratios

Fixes #123
```

## Project Structure

- `core/` - Core configuration and constants
- `local_processing/` - Secure data processing
- `hypervector/` - Hyperdimensional computing
- `zk_proofs/` - Zero-knowledge proofs
- `pir/` - Private Information Retrieval
- `blockchain/` - Blockchain integration
- `api/` - REST API
- `tests/` - Test suite

## Documentation

- Update relevant documentation when making changes
- Add docstrings to new functions and classes
- Update the README if adding new features
- Consider adding examples for complex features

## Questions?

Feel free to open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing to GenomeVault! 🧬🔒
