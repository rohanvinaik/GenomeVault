# CLAUDE.md

Quick reference for Claude Code when working with the GenomeVault codebase.

## Project Overview

GenomeVault: Privacy-preserving genomic computing platform using hyperdimensional computing (HDC), Kolmogorov-Arnold Networks (KAN), zero-knowledge proofs, and federated learning. Achieves 50-100× compression with mathematical privacy guarantees.

## Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"  # or ".[full]" for all features

# Run API
uvicorn genomevault.api.main:app --reload --port 8000

# Run tests & checks
make test        # or: pytest
make lint        # or: ruff check . && ruff format .
make typecheck   # or: mypy genomevault

# Database setup
alembic upgrade head
python scripts/seed_data.py  # Load test data
```

## Core Architecture

```
genomevault/
├── api/              # FastAPI endpoints, OAuth2/OIDC auth
├── hypervector/      # HD encoding (10K-100K dimensions)
├── kan/              # KAN compression with splines
├── zk_proofs/        # Zero-knowledge proof circuits
├── federated/        # Federated learning
├── pir/              # Private information retrieval
├── clinical/         # Clinical evaluation & calibration
├── blockchain/       # Governance & audit trail
└── models/           # SQLAlchemy models (partitioned tables)
```

## Essential Commands

```bash
# Development
pytest tests/test_hypervector.py  # Run specific tests
ruff check --fix .                 # Auto-fix linting issues
python scripts/bench_hdc.py        # Performance benchmarks

# Database operations
alembic revision --autogenerate -m "description"
python scripts/backup_restore.sh --backup full

# Docker/K8s
docker-compose up -d
kubectl apply -f deployment/kubernetes/
```

## Key Features & Usage

### API Endpoints
- `POST /hv/encode` - Encode genomic data to hypervector
- `GET /health` - System health check with auth
- OAuth2 flows at `/auth/token`, `/auth/refresh`

### Accuracy Modes
- `OFF`: 90-95% (basic screening)
- `COMMON`: 95-98% (epidemiology)
- `CLINICAL`: 98-99.5% (diagnostics)
- `KAN-HD`: 99%+ (regulatory)

### Environment Variables
```bash
DATABASE_URL=postgresql://user:pass@localhost/genomevault
JWT_SECRET_KEY=your-secret-key
ENABLE_MFA=true
S3_BACKUP_BUCKET=genomevault-backups
```

## Common Tasks

### Fix Import Errors
```python
# Missing imports usually in:
genomevault/core/exceptions.py  # Add custom exceptions here
```

### Add New Feature
1. Create feature branch from `main`
2. Add tests in `tests/`
3. Run `make lint test` before commit
4. Ensure migrations if DB changes

### Deploy Updates
```bash
# Build and push Docker image
docker build -t genomevault/api:latest .
docker push genomevault/api:latest

# Update Kubernetes
kubectl set image deployment/genomevault-api api=genomevault/api:latest
```

## End-to-End Pipeline Testing

### Quick E2E Test
```bash
# Run comprehensive E2E test with full statistics
python run_e2e_test.py

# Or use the CLI for component testing
genomevault demo run --type full
```

### Manual E2E Pipeline
```bash
# 1. Encode genomic data
echo '[1.5, 2.3, 0.8, 3.2, 1.1]' > expression.json
genomevault hdc encode --json expression.json --dimension 1000 --out encoded.json

# 2. Compare vectors
genomevault hdc compare --v1 encoded1.json --v2 encoded2.json --metric all

# 3. Start PIR server
echo '["record1", "record2", "record3"]' > database.json
genomevault pir serve --data database.json --port 8001

# 4. Query PIR server
genomevault pir query --servers "http://localhost:8001" --index 1

# 5. Generate ZK proof (requires setup files)
genomevault zk build --circuit-type variant
genomevault zk prove --public pub.json --private priv.json
genomevault zk verify --proof proof.json --public pub.json
```

### Python E2E Example
```python
from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
from genomevault.core.constants import OmicsType
from genomevault.pir.servers import PIRServer
from genomevault.zk_proofs.prover import Prover
import numpy as np

# 1. HDC Encoding
config = HypervectorConfig(dimension=1000)
encoder = HypervectorEncoder(config=config)
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
encoded = encoder.encode(data, OmicsType.GENOMIC)

# 2. PIR Storage & Retrieval
records = [b"variant1", b"variant2", b"variant3"]
server = PIRServer(records)
mask = np.zeros(len(records), dtype=np.uint8)
mask[1] = 1  # Retrieve second record privately
result = server.answer(mask)

# 3. ZK Proof
prover = Prover()
public = {"threshold": 0.5}
private = {"actual": 0.75}
# proof = prover.prove_variant(public, private)  # Requires complete inputs

print(f"✅ E2E Pipeline Complete")
print(f"  HDC Vector: {encoded.shape if hasattr(encoded, 'shape') else len(encoded)} dimensions")
print(f"  PIR Result: {result.rstrip(b'\\0').decode()}")
```

### E2E Test Results Location
- Comprehensive results: `genomevault_e2e_results.json`
- CLI demo output: `/tmp/demo_output/`
- Test artifacts: `/tmp/encoded_vector.json`

## Performance Tips
- Use Hamming LUTs for 10-20× speedup
- Batch operations over individual calls
- Enable GPU with `pip install -e ".[gpu]"`

## Security Checklist
- [ ] No secrets in code (use `.env`)
- [ ] HD encoding for all genomic data
- [ ] Audit logs for PHI access
- [ ] Encrypted backups (AES-256)
- [ ] HIPAA compliance (7-year retention)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Check `genomevault/core/exceptions.py` |
| Slow HD ops | Enable LUTs, use batch operations |
| Auth failures | Verify JWT_SECRET_KEY, check token expiry |
| DB connection | Check DATABASE_URL, run migrations |
| Backup fails | Verify S3/GCS credentials, check disk space |

## Current Status
- Branch: `clean-slate` (PR target: `main`)
- Modified: `genomevault/zk_proofs/` files
- Pending: Some import fixes needed
