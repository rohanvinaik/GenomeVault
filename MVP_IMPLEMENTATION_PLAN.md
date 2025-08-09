# GenomeVault MVP Implementation Plan

## 🚨 Critical Blockers (Fix First)

### 1. Syntax Errors in Core Files
The audit identified these files with syntax errors that MUST be fixed:

```python
# Files with '_' placeholder issues:
- genomevault/local_processing/epigenetics.py (many _ placeholders)
- genomevault/local_processing/proteomics.py
- genomevault/local_processing/transcriptomics.py
- genomevault/pir/server/enhanced_pir_server.py
- genomevault/zk_proofs/prover.py

# Files with indentation issues:
- devtools/trace_import_failure.py (line 20)
- examples/minimal_verification.py (line 33)
```

## ✅ What's Already Working

Based on the recovery and analysis:

1. **HDC Module** ✅
   - `genomevault/hdc/core.py` - Encoding, bundling, similarity
   - Tests in `tests/smoke/test_hdc.py`

2. **Hypervector Module** ✅
   - Original `genomevault/hypervector/*` intact
   - Different from HDC, both can coexist

3. **Local Processing** (Partial)
   - Complex implementations exist but have syntax errors
   - Need to fix placeholder variables

4. **Federated Aggregation** ✅
   - `genomevault/federated/aggregate.py` implemented

## 🔧 What Needs Implementation

### Priority 1: Fix Syntax Errors (Blocker)
```bash
# Script to fix common issues:
1. Replace all standalone '_' with meaningful variable names
2. Fix f-string formatting issues
3. Fix indentation problems
4. Remove duplicate logger declarations
```

### Priority 2: Create Main API Application
```python
# genomevault/api/main.py
from fastapi import FastAPI
from genomevault.api.routers import health, encode, vectors, proofs, pir

app = FastAPI(title="GenomeVault", version="0.1.0")
app.include_router(health.router)
app.include_router(encode.router)
app.include_router(vectors.router)  # Need to create
app.include_router(proofs.router)   # Need to create
app.include_router(pir.router)      # Need to create
```

### Priority 3: Complete Missing Endpoints
```python
# genomevault/api/routers/vectors.py
@router.post("/vectors/bundle")
@router.post("/vectors/similarity")

# genomevault/api/routers/proofs.py
@router.post("/proofs/create")
@router.post("/proofs/verify")

# genomevault/api/routers/pir.py
@router.post("/pir/query")
@router.get("/pir/status")
```

### Priority 4: Wire Up Existing Modules
- Connect PIR client/server implementations
- Link ZK proof circuits to API
- Integrate blockchain contracts

### Priority 5: Complete Test Suite
```python
# tests/integration/test_full_pipeline.py
- Test data ingestion → encoding → proof generation
- Test PIR query flow
- Test federated aggregation

# tests/e2e/test_api_flow.py
- Test complete API workflow
- Test error handling
- Test authentication/authorization
```

## 📋 MVP Acceptance Criteria

### Functional Requirements
- [ ] Fix all syntax errors - code compiles
- [ ] All smoke tests pass
- [ ] API starts and serves /docs
- [ ] Can encode genomic data via API
- [ ] Can generate/verify ZK proofs
- [ ] Can perform PIR queries
- [ ] Federated aggregation works

### Non-Functional Requirements
- [ ] README with quickstart guide
- [ ] Docker compose for local development
- [ ] Basic CI/CD with GitHub Actions
- [ ] Linting passes (ruff)
- [ ] Type checking passes (mypy - relaxed)

## 🚀 Quick Implementation Script

```bash
#!/bin/bash
# fix_mvp.sh

# 1. Fix syntax errors
python fix_syntax_errors.py

# 2. Create main API app
cat > genomevault/api/main.py << 'EOF'
from fastapi import FastAPI
from genomevault.api.routers import health, encode

app = FastAPI(title="GenomeVault", version="0.1.0")
app.include_router(health.router)
app.include_router(encode.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# 3. Test the API
python genomevault/api/main.py &
sleep 2
curl http://localhost:8000/health
kill %1

# 4. Run tests
pytest tests/smoke -v

# 5. Create Docker setup
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
EOF

echo "MVP setup complete!"
```

## 📊 Effort Estimate

| Task | Priority | Hours | Complexity |
|------|----------|-------|------------|
| Fix syntax errors | P0 | 2-3 | Medium |
| Create main API | P0 | 1 | Low |
| Complete endpoints | P1 | 3-4 | Medium |
| Wire up modules | P1 | 4-6 | High |
| Complete tests | P2 | 3-4 | Medium |
| Documentation | P2 | 2 | Low |
| **Total** | | **15-20** | |

## 🎯 Definition of "Shippable MVP"

A shippable MVP for GenomeVault should:

1. **Run without errors** - All syntax fixed, imports work
2. **Serve core API** - FastAPI app with /docs
3. **Demonstrate key features**:
   - Encode genomic data to hypervectors
   - Generate mock ZK proofs
   - Perform basic PIR queries
   - Aggregate federated statistics
4. **Be deployable** - Docker container that starts
5. **Have basic tests** - Smoke tests pass
6. **Include documentation** - README with quickstart

This is achievable in 2-3 days of focused work.
