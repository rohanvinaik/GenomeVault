# Backend Selection Strategies

GenomeVault supports two backend selection strategies: **Configuration-Based** (default) and **Intelligent** (experimental).

## Overview

| Feature | Config Mode | Intelligent Mode |
|---------|-------------|------------------|
| Selection Method | Static rules from `compute.yaml` | Dynamic data analysis |
| Performance | Predictable, consistent | Adaptive, potentially optimal |
| Complexity | Simple, explicit | Automatic, transparent |
| Default | ✅ Yes | ❌ No (opt-in) |
| Production Ready | ✅ Yes | ⚠️ Experimental |

## Configuration-Based Selection (Default)

### How It Works

Backend selection is determined by static rules in `genomevault/config/compute.yaml`:

1. **Component-Specific Rules**: Different components use different backends
   - HDC encoding: CPU for single samples, GPU for large batches (>100)
   - Similarity search: CPU for small databases (<10K), GPU for large
   - ZK proofs: Always CPU (algorithm requirement)
   - PIR queries: Always CPU (network-bound)

2. **Threshold-Based Decisions**: Fixed thresholds determine backend selection
   ```yaml
   hdc_encoding:
     single_sample: "cpu"
     batch_threshold: 100
   ```

3. **Preset Configurations**: Pre-configured profiles for common scenarios
   - `production_api`: All CPU for predictable latency
   - `research_batch`: GPU for throughput
   - `edge_deployment`: CPU with aggressive FAISS

### Configuration Example

```yaml
# genomevault/config/compute.yaml
compute:
  default_backend: "auto"  # Auto-detect: Metal > CUDA > CPU

  hdc_encoding:
    single_sample: "cpu"
    batch_threshold: 100

  zk_proofs:
    backend: "cpu"
    allow_override: false  # Never use GPU for ZK
```

### Usage

```python
# Default config-based selection
from genomevault.hypervector_transform import create_backend_encoder

encoder = create_backend_encoder(dimension=8192)
# Uses configuration from compute.yaml
```

### When to Use Config Mode

✅ **Use configuration-based selection for:**
- Production deployments (predictable performance)
- Known workloads (pre-optimized thresholds)
- Debugging (eliminates selection variance)
- Reproducible benchmarks
- Safety-critical applications (explicit, auditable)

## Intelligent Selection (Experimental)

### How It Works

Intelligent mode analyzes data characteristics and operation requirements to select the optimal backend dynamically:

1. **Data Profiling**: Analyzes input data
   - Size: Number of samples/records
   - Dimensionality: Feature dimensions
   - Sparsity: Percentage of zeros
   - Memory footprint: Estimated RAM needed
   - Complexity score: Computational cost estimate

2. **Analysis Profiling**: Infers operation requirements
   - Operation type: Single/batch/streaming
   - Latency requirements: Real-time/interactive/batch
   - Throughput needs: Low/medium/high
   - Context: Interactive vs. background processing

3. **Performance Prediction**: Uses empirical models
   - Predicts CPU execution time
   - Predicts GPU execution time (including warmup)
   - Compares predicted performance
   - Accounts for GPU warmup overhead

4. **Decision Logic**: Selects optimal backend
   - Config overrides take precedence (highest priority)
   - ZK/PIR always use CPU (mandatory)
   - Small data (<100 samples) → CPU (avoid GPU warmup)
   - Real-time operations → CPU (predictable latency)
   - Large batches (>1000 samples) → GPU (maximize throughput)
   - Medium data → Compare predictions, prefer CPU if similar

### Configuration Example

```yaml
# genomevault/config/compute.yaml
intelligent_mode:
  enabled: true  # Enable intelligent selection

  thresholds:
    small_data_samples: 100
    large_data_samples: 1000
    gpu_warmup_cost_ms: 5.0
    interactive_latency_target_ms: 100.0

  performance_models:
    hdc_encoding:
      cpu_time_per_sample_ms: 5.0
      gpu_time_per_sample_ms: 0.5
      gpu_warmup_ms: 5.0
      batch_crossover_point: 100
```

### Usage

```python
from genomevault.compute.backend import ComputeBackendManager

# Enable intelligent mode
manager = ComputeBackendManager(intelligent_mode=True)

# Select backend based on data analysis
backend, reason = manager.get_backend_for_operation(
    operation='encode',
    data=my_variants,  # Actual data or size hint
    context={'interactive': True}
)

print(f"Selected {backend.value}: {reason}")
# Output: "Selected cpu: Real-time latency requirement met by CPU (45.2ms < 100.0ms)"
```

### Example Decision Flow

```python
# Example 1: Small interactive query
data = np.random.rand(50, 100)  # 50 samples
backend, reason = manager.get_backend_for_operation(
    operation='encode',
    data=data,
    context={'interactive': True}
)
# Result: CPU - "Small data (50 < 100) - CPU overhead-free"

# Example 2: Large batch job
data = np.random.rand(2000, 100)  # 2000 samples
backend, reason = manager.get_backend_for_operation(
    operation='encode_batch',
    data=data,
    context={'batch': True}
)
# Result: GPU - "Large data (2000 ≥ 1000) - GPU faster (105.0ms < 10000.0ms)"

# Example 3: ZK proof (always CPU)
backend, reason = manager.get_backend_for_operation(
    operation='prove',
    data=1000000  # Even very large
)
# Result: CPU - "zk_proof operations require CPU (algorithm design)"
```

### When to Use Intelligent Mode

✅ **Use intelligent selection for:**
- Exploratory data analysis (unknown data sizes)
- Mixed workloads (variable batch sizes)
- Research pipelines (automatic optimization)
- Development/prototyping (less manual tuning)
- Multi-tenant systems (different workload patterns)

❌ **Avoid intelligent mode for:**
- Production deployments (use config mode for predictability)
- Real-time critical systems (avoid selection overhead)
- Compliance/audit requirements (explicit config preferred)
- Debugging performance issues (eliminates variable)

## Performance Models

Intelligent mode uses linear performance models based on GenomeVault benchmarks:

### HDC Encoding

| Backend | Warmup | Per Sample | Crossover Point |
|---------|--------|------------|-----------------|
| CPU | 0ms | 5ms | - |
| Metal | 5ms | 0.5ms (10× faster) | >100 samples |
| CUDA | 10ms | 2ms (2.5× faster) | >150 samples |

**Decision Logic**:
- <100 samples: CPU (GPU warmup not worth it)
- 100-1000 samples: Compare predictions
- >1000 samples: GPU (better throughput)

### Similarity Search

| Backend | Warmup | Per 1K Records | Crossover Point |
|---------|--------|----------------|-----------------|
| CPU | 0ms | 2ms | - |
| Metal | 5ms | 0.2ms (10× faster) | >10K records |
| CUDA | 10ms | 0.5ms (4× faster) | >15K records |

**Decision Logic**:
- <10K database: CPU
- 10K-100K database: GPU if available
- >100K database: GPU + FAISS indexing

## Decision Transparency

Intelligent mode logs detailed reasoning for every selection:

```
INFO: Intelligent Backend Selection:
INFO:   Data: 500 samples, 5.2% sparse, 1.9MB
INFO:   Analysis: batch_encode, latency=batch
INFO:   Predicted: CPU 2500.0ms, GPU 255.0ms
INFO:   → GPU selected: Large data (500 ≥ 1000) - GPU faster (255.0ms < 2500.0ms)
```

## Migration Guide

### From Config Mode to Intelligent Mode

**Step 1**: Enable in configuration
```yaml
# compute.yaml
intelligent_mode:
  enabled: true
```

**Step 2**: Update code (if using backend manager directly)
```python
# Before (config mode)
manager = ComputeBackendManager()

# After (intelligent mode)
manager = ComputeBackendManager(intelligent_mode=True)
```

**Step 3**: Add operation context (optional, for better decisions)
```python
# Before
encoder.encode_single(data)

# After (with context)
backend, reason = manager.get_backend_for_operation(
    operation='encode',
    data=data,
    context={'interactive': True, 'latency_sensitive': True}
)
encoder.encode_single(data)
```

### Environment Variable Override

```bash
# Enable intelligent mode via environment
export GENOMEVAULT_INTELLIGENT_MODE=true

# Still respects backend override
export GENOMEVAULT_BACKEND=cpu  # Forces CPU even in intelligent mode
```

## Constraints and Guarantees

### Mandatory CPU Operations

These operations **ALWAYS** use CPU regardless of mode:

1. **ZK Proofs**: Algorithm is CPU-optimized, GPU provides <10% benefit
2. **PIR Queries**: Network-bound, not compute-bound

**Enforcement**: Config overrides are respected absolutely
```python
# Even with intelligent mode + large data, ZK uses CPU
backend, reason = manager.get_backend_for_operation(
    operation='prove',
    data=1000000
)
assert backend == ComputeBackend.CPU  # Always true
```

### Backward Compatibility

Intelligent mode is **100% backward compatible**:

- Default: `intelligent_mode=False` (config-based selection)
- Existing code works identically
- No API changes required
- Opt-in via parameter or config

## Troubleshooting

### Intelligent Mode Not Working

**Symptom**: Still using config-based selection despite `intelligent_mode=True`

**Causes**:
1. Missing configuration file
2. Import error for `IntelligentBackendSelector`
3. Fallback triggered by error

**Debug**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

manager = ComputeBackendManager(intelligent_mode=True)
# Check logs for initialization warnings
```

### Unexpected Backend Selection

**Symptom**: Intelligent mode selects different backend than expected

**Debug**:
```python
# Enable detailed logging
import logging
logging.getLogger('genomevault.compute.intelligent_selector').setLevel(logging.INFO)

backend, reason = manager.get_backend_for_operation(...)
print(f"Reason: {reason}")
# Examine prediction details in logs
```

### Performance Predictions Inaccurate

**Symptom**: Actual execution time differs significantly from predictions

**Solution**: Update performance models in `compute.yaml`:
```yaml
intelligent_mode:
  performance_models:
    hdc_encoding:
      cpu_time_per_sample_ms: 7.5  # Update based on your benchmarks
      gpu_time_per_sample_ms: 0.8
```

## Best Practices

### 1. Start with Config Mode

Begin with configuration-based selection for production:
- Explicit, predictable behavior
- Easier to debug and audit
- Well-tested in production

### 2. Test Intelligent Mode in Development

Enable intelligent mode in development/staging:
- Validate selection decisions
- Tune performance models
- Verify performance improvements

### 3. Provide Operation Context

Help intelligent selector make better decisions:
```python
# Good: Provides context
backend, reason = manager.get_backend_for_operation(
    operation='encode',
    data=data,
    context={
        'interactive': True,          # User waiting
        'latency_sensitive': True,    # <100ms target
    }
)

# OK: Minimal context
backend, reason = manager.get_backend_for_operation(
    operation='encode',
    data=data
)
```

### 4. Monitor and Tune

Review selection decisions and tune models:
```bash
# Examine selection reasoning
grep "Intelligent Backend Selection" logs/genomevault.log | tail -20

# Check for suboptimal selections
grep "Fallback" logs/genomevault.log
```

### 5. Respect Component Constraints

Never override mandatory CPU operations:
```python
# WRONG: Trying to force GPU for ZK
manager.get_backend_for_operation(
    operation='prove',
    data=data,
    config_override=ComputeBackend.CUDA  # Will be ignored!
)
```

## References

- [Compute Backend Configuration](../genomevault/config/compute.yaml)
- [Intelligent Selector Implementation](../genomevault/compute/intelligent_selector.py)
- [Performance Models](../genomevault/compute/performance_models.py)
- [Backend Integration Tests](../tests/test_intelligent_backend.py)
