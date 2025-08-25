# GenomeVault Reproducible Benchmark System

## Overview

GenomeVault provides a fully reproducible, deterministic, and cryptographically verifiable benchmark system. Every benchmark run produces signed, timestamped results with complete environment snapshots.

## Quick Start

### One-Command Execution

```bash
# Run with Docker (fully reproducible)
./run_reproducible_benchmark.sh docker

# Run locally (system-dependent)  
./run_reproducible_benchmark.sh local
```

### Docker Compose

```bash
# Run benchmarks and auto-verify
docker-compose -f docker-compose.benchmark.yml up

# Run with custom seed
GENOMEVAULT_SEED=12345 docker-compose -f docker-compose.benchmark.yml up
```

## Features

### 🔒 Deterministic Execution
- Fixed random seeds across all libraries (NumPy, PyTorch, Python)
- Deterministic CUDA operations when GPU available
- Environment variable control (`GENOMEVAULT_SEED`)
- Reproducible data generation

### 📦 Portable Containerization
- Docker container with pinned dependencies
- No absolute paths in code
- Platform-agnostic execution
- Resource limits enforced

### ✅ Cryptographic Verification
- RSA-2048 digital signatures (PSS padding, SHA-256)
- Tamper-evident results
- Standalone verification script
- Public key included with results

### 📊 Comprehensive Output

Each run produces:
```
results/
├── {git_sha}/
│   └── {timestamp}/
│       ├── results.json         # Signed benchmark results
│       ├── environment.json     # Complete environment snapshot
│       ├── sbom.json           # Software Bill of Materials
│       ├── signature.sig       # Digital signature
│       ├── public_key.pem      # Public verification key
│       ├── verify.py           # Standalone verifier
│       └── {benchmark_name}/
│           ├── result.json    # Individual benchmark result
│           └── execution.log  # Captured stdout/stderr
```

## Result Structure

### results.json
```json
{
  "run_id": "abc123_2025-08-25T01-00-00",
  "git_sha": "abc123def456...",
  "timestamp": "2025-08-25T01:00:00.000000",
  "seed": 42,
  "environment": { ... },
  "benchmarks": [
    {
      "name": "HDC Encoding",
      "status": "success",
      "execution_time": 1.234,
      "results": { ... }
    }
  ]
}
```

### environment.json
- Platform details (OS, architecture, Python version)
- Hardware capabilities (CUDA, Metal, CPU)
- Environment variables
- Git commit SHA
- Timestamp and seed

### sbom.json
- All Python dependencies with versions
- System libraries
- Package checksums (when available)

## Verification

### Standalone Verification
```bash
# Verify any result directory
python results/{git_sha}/{timestamp}/verify.py results/{git_sha}/{timestamp}
```

### Programmatic Verification
```python
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

def verify_results(results_dir: Path) -> bool:
    """Verify benchmark results signature"""
    # Load results and signature
    with open(results_dir / "results.json", 'r') as f:
        results = json.load(f)
    
    with open(results_dir / "signature.sig", 'rb') as f:
        signature = f.read()
    
    with open(results_dir / "public_key.pem", 'rb') as f:
        public_key = serialization.load_pem_public_key(f.read())
    
    # Verify
    try:
        public_key.verify(
            signature,
            json.dumps(results, sort_keys=True).encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except:
        return False
```

## Reproducibility Guarantees

### Deterministic Components
✅ Random number generation (seeded)
✅ Data generation patterns
✅ Algorithm initialization
✅ Timestamp ordering
✅ Hash computations

### Platform-Dependent Components
⚠️ Floating-point rounding (may vary by CPU)
⚠️ Parallel execution timing
⚠️ Memory allocation patterns
⚠️ Hardware acceleration (GPU/Metal)

## Adding New Benchmarks

```python
def benchmark_my_feature(seed: int = 42) -> Dict[str, Any]:
    """Custom benchmark with deterministic execution"""
    np.random.seed(seed)
    
    # Your benchmark code here
    results = {
        'metric1': value1,
        'metric2': value2
    }
    
    return results

# In main():
result = harness.run_benchmark(
    "My Feature",
    benchmark_my_feature,
    seed=harness.seed
)
```

## CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Reproducible Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run benchmarks
        run: ./run_reproducible_benchmark.sh docker
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results/
      
      - name: Verify signatures
        run: |
          find results -name verify.py -exec {} $(dirname {}) \;
```

## Security Considerations

### Key Management
- Private keys stored in `~/.genomevault/keys/`
- Never commit private keys
- Public keys included with results for verification
- Keys generated automatically on first run

### Signature Algorithm
- RSA 2048-bit keys
- PSS padding with SHA-256
- Maximum salt length
- Deterministic JSON serialization

### Trust Model
- Results signed by benchmark runner
- Verifier trusts public key included with results
- For production: use PKI or key registry

## Troubleshooting

### Docker Issues
```bash
# Permission denied
sudo usermod -aG docker $USER

# Out of space
docker system prune -a

# Build cache issues
docker build --no-cache -f Dockerfile.benchmark .
```

### Verification Failures
```bash
# Check file integrity
sha256sum results/*/results.json

# Verify manually
openssl dgst -sha256 -verify public_key.pem -signature signature.sig results.json
```

### Non-Deterministic Results
1. Check seed is set: `echo $GENOMEVAULT_SEED`
2. Verify Docker mode: `./run_reproducible_benchmark.sh docker`
3. Check for timing-dependent code
4. Review parallel execution order

## Best Practices

1. **Always use Docker mode for official benchmarks**
2. **Commit results to git for audit trail**
3. **Include git SHA in reports**
4. **Verify signatures before trusting results**
5. **Use consistent seeds across runs**
6. **Document any platform-specific behaviors**

## License

The reproducibility framework is part of GenomeVault and follows the same license terms.