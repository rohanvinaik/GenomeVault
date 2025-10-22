# Backend System Migration Guide

Guide for migrating from legacy HDC encoding to the new hardware-accelerated backend system.

---

## Overview

GenomeVault now includes a unified hardware backend system that provides:
- **Automatic backend detection** (Metal > CUDA > CPU)
- **Configuration-driven backend selection** via YAML and environment variables
- **Performance optimization** for production, research, and edge deployments
- **Backward compatibility** with existing code

---

## Quick Start

### New Code (Recommended)

```python
from genomevault.hypervector_transform import create_backend_encoder

# Auto-detect best backend
encoder = create_backend_encoder(dimension=8192)

# Encode single sample
hypervector = encoder.encode_single(variants)

# Encode batch
hypervectors = encoder.encode_batch(variants_batch)
```

### Legacy Code (Still Supported)

```python
from genomevault.hypervector_transform import HypervectorEncoder, HypervectorConfig

# Old approach still works
config = HypervectorConfig(dimension=8192, use_metal=True)
encoder = HypervectorEncoder(config)
hypervector = encoder.encode(variants, OmicsType.GENOMIC)
```

---

## Migration Paths

### Path 1: Minimal Changes (Drop-in Replacement)

**Before**:
```python
from genomevault.hypervector_transform import HypervectorEncoder, HypervectorConfig

config = HypervectorConfig(dimension=8192)
encoder = HypervectorEncoder(config)

# Encoding
hv = encoder.encode(features, OmicsType.GENOMIC)
```

**After**:
```python
from genomevault.hypervector_transform import create_backend_encoder

encoder = create_backend_encoder(dimension=8192)

# Encoding (simpler API)
hv = encoder.encode_single(features)
```

**Changes**:
- ✅ Simpler API (no `OmicsType` required)
- ✅ Auto-detects best backend
- ✅ Returns numpy arrays (not torch tensors)

---

### Path 2: Configuration-Driven (Production)

**Before**:
```python
# Hardcoded Metal detection
config = HypervectorConfig(dimension=8192, use_metal=True)
encoder = HypervectorEncoder(config)
```

**After**:
```python
# Configuration from compute.yaml + environment variables
from genomevault.hypervector_transform import BackendOptimizedEncoder

encoder = BackendOptimizedEncoder()  # Reads config automatically
```

**Setup** `genomevault/config/compute.yaml`:
```yaml
compute:
  default_backend: "auto"  # or "cpu", "metal", "cuda"
```

**Environment Variables**:
```bash
export GENOMEVAULT_BACKEND=cpu  # Override for production
export GENOMEVAULT_PRESET=production_api
```

---

### Path 3: Explicit Backend Selection

**Before**:
```python
# Manual backend detection
if METAL_AVAILABLE:
    config = HypervectorConfig(use_metal=True)
else:
    config = HypervectorConfig(use_metal=False)
encoder = HypervectorEncoder(config)
```

**After**:
```python
from genomevault.hypervector_transform import create_backend_encoder

# Explicit backend selection
encoder = create_backend_encoder(dimension=8192, backend='metal')
# or backend='cpu', backend='cuda', backend='auto'
```

---

## Feature Comparison

| Feature | Legacy (HypervectorEncoder) | New (BackendOptimizedEncoder) |
|---------|----------------------------|-------------------------------|
| Metal Support | ✅ Via `use_metal=True` | ✅ Automatic detection |
| CUDA Support | ❌ Not supported | ✅ Full support |
| CPU Fallback | ✅ Manual | ✅ Automatic |
| Configuration | Code-based | YAML + env vars |
| Batch Encoding | ✅ Via `encode()` | ✅ Optimized `encode_batch()` |
| Similarity Search | ❌ External | ✅ Built-in with FAISS |
| HDC Operations | ❌ External | ✅ `bind_vectors()`, `bundle_vectors()` |
| Return Type | `torch.Tensor` | `np.ndarray` |
| API Simplicity | Moderate | Simple |

---

## Common Migration Scenarios

### Scenario 1: Production API (Latency-Optimized)

**Goal**: Predictable <10ms latency for real-time queries

**Before**:
```python
# Force CPU for production
config = HypervectorConfig(dimension=8192, use_metal=False)
encoder = HypervectorEncoder(config)
```

**After**:
```python
# Configuration-driven
encoder = create_backend_encoder(dimension=8192, backend='cpu')
# or set GENOMEVAULT_BACKEND=cpu environment variable
```

**Deployment**:
```bash
# Kubernetes
env:
  - name: GENOMEVAULT_BACKEND
    value: "cpu"
  - name: GENOMEVAULT_OPTIMIZE_LATENCY
    value: "true"
```

---

### Scenario 2: Research Batch Processing (Throughput-Optimized)

**Goal**: Maximum throughput for large-scale analysis

**Before**:
```python
# Try Metal, fallback to CPU
try:
    config = HypervectorConfig(dimension=8192, use_metal=True)
    encoder = HypervectorEncoder(config)
except:
    config = HypervectorConfig(dimension=8192, use_metal=False)
    encoder = HypervectorEncoder(config)

# Manual batch processing
results = [encoder.encode(v, OmicsType.GENOMIC) for v in variants_batch]
```

**After**:
```python
# Auto-detect GPU, optimized batch processing
encoder = create_backend_encoder(dimension=8192, backend='auto')

# Single batch operation (50× faster on GPU)
results = encoder.encode_batch(variants_batch)
```

---

### Scenario 3: Edge Deployment (Resource-Constrained)

**Goal**: Efficient operation on hospital servers

**Before**:
```python
# CPU-only with manual optimization
config = HypervectorConfig(dimension=8192, use_metal=False)
encoder = HypervectorEncoder(config)
```

**After**:
```python
# Use edge_deployment preset
encoder = BackendOptimizedEncoder()
# Preset enables aggressive FAISS usage for efficiency
```

**Configuration** (`compute.yaml`):
```yaml
presets:
  edge_deployment:
    default_backend: "cpu"
    hdc_encoding:
      enable_faiss: true
      faiss_threshold: 50000  # Lower threshold
```

---

## API Reference

### BackendOptimizedEncoder

```python
class BackendOptimizedEncoder:
    """Hardware-accelerated HDC encoder"""

    def __init__(self, config: Optional[BackendEncoderConfig] = None)

    def encode_single(
        self,
        variants: Union[np.ndarray, torch.Tensor],
        omics_type: Optional[OmicsType] = None
    ) -> np.ndarray:
        """Encode single sample to hypervector"""

    def encode_batch(
        self,
        variants_batch: list[Union[np.ndarray, torch.Tensor]],
        omics_type: Optional[OmicsType] = None
    ) -> np.ndarray:
        """Encode batch of samples to hypervectors"""

    def similarity_search(
        self,
        query: Union[np.ndarray, torch.Tensor],
        database: Union[np.ndarray, torch.Tensor],
        top_k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar hypervectors"""

    def bind_vectors(
        self,
        a: Union[np.ndarray, torch.Tensor],
        b: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """HDC binding operation (XOR)"""

    def bundle_vectors(
        self,
        vectors: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """HDC bundling operation (majority vote)"""

    @property
    def backend_name(self) -> str:
        """Get name of current backend"""

    @property
    def backend_type(self) -> ComputeBackend:
        """Get type of current backend"""
```

### Convenience Function

```python
def create_backend_encoder(
    dimension: int = 8192,
    backend: Optional[str] = None,  # 'cpu', 'metal', 'cuda', 'auto'
    **kwargs
) -> BackendOptimizedEncoder:
    """Create backend-optimized encoder"""
```

---

## Performance Expectations

### Single Sample Encoding

| Backend | Latency | Use Case |
|---------|---------|----------|
| CPU | 5-10ms ✓ | Production API |
| Metal | <1ms ⚡ | Development |
| CUDA | ~2ms ⚠️ | Not recommended (overhead) |

### Batch Encoding (1K samples)

| Backend | Time | Throughput | Speedup |
|---------|------|------------|---------|
| CPU | 5s | 200/sec | 1× |
| Metal | 0.1s | 10,000/sec | 50× |
| CUDA | 0.15s | 6,667/sec | 33× |

---

## Backward Compatibility

### Existing Code Continues to Work

The legacy `HypervectorEncoder` is **fully supported**:

```python
# This code still works exactly as before
from genomevault.hypervector_transform import HypervectorEncoder, HypervectorConfig

config = HypervectorConfig(dimension=8192, use_metal=True)
encoder = HypervectorEncoder(config)
hv = encoder.encode(features, OmicsType.GENOMIC)
```

### Gradual Migration

You can migrate incrementally:
1. ✅ Keep existing code running
2. ✅ Add new backend system for new features
3. ✅ Migrate critical paths when ready
4. ✅ No breaking changes

---

## Testing Migration

### Verify Backend Detection

```python
from genomevault.hypervector_transform import create_backend_encoder

encoder = create_backend_encoder()
print(f"Detected backend: {encoder.backend_name}")
# Expected: "Metal (Apple Silicon)" or "CPU" or "CUDA"
```

### Compare Performance

```python
import time
import numpy as np
from genomevault.hypervector_transform import (
    HypervectorEncoder,
    HypervectorConfig,
    create_backend_encoder,
)

# Test data
test_data = np.random.randn(100, 10).astype(np.float32)

# Legacy encoder
legacy_encoder = HypervectorEncoder(HypervectorConfig(dimension=8192))
start = time.perf_counter()
legacy_result = legacy_encoder.encode(test_data, OmicsType.GENOMIC)
legacy_time = time.perf_counter() - start

# New backend encoder
backend_encoder = create_backend_encoder(dimension=8192)
start = time.perf_counter()
backend_result = backend_encoder.encode_single(test_data)
backend_time = time.perf_counter() - start

print(f"Legacy: {legacy_time*1000:.2f}ms")
print(f"Backend: {backend_time*1000:.2f}ms")
print(f"Speedup: {legacy_time/backend_time:.1f}×")
```

### Run Integration Tests

```bash
# Run backend integration tests
pytest tests/test_backend_integration.py -v

# Run all HDC tests
pytest tests/test_hypervector*.py -v
```

---

## Troubleshooting

### Backend Not Detected

**Problem**: "Backend defaulting to CPU despite having GPU"

**Solutions**:
1. Check backend detection:
   ```python
   from genomevault.compute import get_backend
   backend = get_backend()
   print(f"Current backend: {backend.value}")
   ```

2. Verify GPU availability:
   ```bash
   # For Metal
   python -c "import mlx.core as mx; print(mx.metal.is_available())"

   # For CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Force specific backend:
   ```python
   encoder = create_backend_encoder(backend='metal')
   ```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'genomevault.compute'`

**Solution**: Ensure you have the latest version:
```bash
pip install -e . --upgrade
```

### Performance Regression

**Problem**: "New backend is slower than legacy"

**Solutions**:
1. Check if you're using the right backend:
   ```python
   print(encoder.backend_name)
   # For batch processing, ensure GPU is being used
   ```

2. For small batches (<100), use CPU explicitly:
   ```python
   encoder = create_backend_encoder(backend='cpu')
   ```

3. Review configuration:
   ```bash
   python -c "from genomevault.config.loader import get_config; get_config().print_config()"
   ```

---

## Best Practices

### ✅ DO

- Use `create_backend_encoder()` for new code
- Let backend auto-detect for development
- Force CPU for production APIs (predictable latency)
- Use GPU (`auto`) for batch processing
- Test backend selection in CI/CD
- Monitor performance per backend

### ❌ DON'T

- Use GPU for single-sample production endpoints
- Hardcode backend selection in application code
- Use CUDA for small batches (<100)
- Assume GPU is always faster (measure!)
- Mix legacy and new encoders for same data

---

## Getting Help

- **Documentation**: `docs/compute_backend_examples.md`
- **Deployment Guide**: `docs/deployment_scenarios.md`
- **Configuration Reference**: `genomevault/config/compute.yaml`
- **Tests**: `tests/test_backend_integration.py`

For questions or issues with migration, check the examples or open an issue.
