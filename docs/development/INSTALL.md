# GenomeVault Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Optional: GPU with CUDA support for accelerated computations

## Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/genomevault.git
cd genomevault

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

## Dependency Overview

### Required Dependencies

1. **Cryptography Stack**
   - `cryptography`: Core encryption (AES-GCM, RSA, key derivation)
   - `PyNaCl`: Modern crypto (ChaCha20, Ed25519)
   - These are ESSENTIAL for security features

2. **Logging & Compliance**
   - `structlog`: Structured logging with privacy filtering
   - REQUIRED for HIPAA/GDPR compliance audit trails

3. **Scientific Computing**
   - `numpy`: Numerical operations
   - `torch`: Hypervector operations and ML
   - `scikit-learn`: Machine learning utilities
   - These are CORE to the platform's functionality

4. **Web Framework** (for API server)
   - `fastapi`: Modern async web framework
   - `uvicorn`: ASGI server
   - `pydantic`: Data validation

### Optional Dependencies

1. **Configuration**
   - `PyYAML`: For YAML config files (falls back to JSON if not installed)

2. **Bioinformatics** (for local processing)
   - `biopython`: Sequence manipulation
   - `pysam`: BAM/SAM file handling

## Minimal Install (Development/Testing)

If you just want to explore the codebase structure:

```bash
# Install only the core dependencies
pip install cryptography PyNaCl structlog numpy torch
```

## Verification

After installation, verify everything is working:

```bash
# Run the test suite
pytest tests/

# Check imports
python -c "import cryptography, nacl, structlog, numpy, torch; print('All core dependencies installed!')"
```

## Common Issues

### Missing System Libraries

Some dependencies require system libraries:

```bash
# On Ubuntu/Debian
sudo apt-get install python3-dev libssl-dev libffi-dev

# On macOS
brew install openssl libffi

# On CentOS/RHEL/Fedora
sudo yum install python3-devel openssl-devel libffi-devel
```

### PyTorch Installation

For GPU support, install the appropriate PyTorch version:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Docker Alternative

For a consistent environment:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "genomevault"]
```

## Important Note

**DO NOT** run GenomeVault without the required security dependencies (cryptography, PyNaCl). These are essential for protecting genomic data and maintaining privacy guarantees.
