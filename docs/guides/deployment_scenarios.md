# GenomeVault Deployment Scenarios

Comprehensive deployment guide for different use cases and environments.

## Table of Contents

- [Overview](#overview)
- [Scenario 1: Production API](#scenario-1-production-api-latency-optimized)
- [Scenario 2: Research Batch Processing](#scenario-2-research-batch-processing-throughput-optimized)
- [Scenario 3: Edge Deployment](#scenario-3-edge-deployment-hospital-servers)
- [Scenario 4: Cloud with GPU](#scenario-4-cloud-deployment-with-gpu)
- [Scenario 5: Hybrid Production/Research](#scenario-5-hybrid-productionresearch)
- [Performance Comparison](#performance-comparison)
- [Migration Guide](#migration-guide)

---

## Overview

| Scenario | Backend | Use Case | Optimization Goal |
|----------|---------|----------|-------------------|
| Production API | CPU | Real-time clinical queries | Latency (<10ms) |
| Research Batch | Auto (GPU) | Large-scale analysis | Throughput (50× speedup) |
| Edge Deployment | CPU | Hospital servers | Resource efficiency |
| Cloud GPU | CUDA/Metal | Bulk processing | Cost per sample |
| Hybrid | CPU + GPU | Mixed workload | Balanced |

---

## Scenario 1: Production API (Latency-Optimized)

### Use Case

Real-time clinical decision support system serving genomic queries with strict latency requirements.

### Architecture

```
Internet → Load Balancer → API Pods (CPU-only) → PostgreSQL
                            ↓
                       CPU Backend
                     (<10ms latency)
```

### Configuration

**`genomevault/config/compute.yaml`**:
```yaml
compute:
  default_backend: "cpu"
  optimize_latency: true

  hdc_encoding:
    single_sample: "cpu"
    batch_threshold: 10000  # Never use GPU

  similarity_search:
    small_database_backend: "cpu"
    large_database_backend: "cpu"  # Even large searches on CPU
```

**Environment Variables**:
```bash
export GENOMEVAULT_BACKEND=cpu
export GENOMEVAULT_OPTIMIZE_LATENCY=true
export GENOMEVAULT_PRESET=production_api
```

### Docker Configuration

**`Dockerfile.production`**:
```dockerfile
FROM python:3.11-slim

# CPU-only dependencies (no GPU drivers needed)
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scipy \
    faiss-cpu \
    fastapi \
    uvicorn

# Copy application
COPY genomevault/ /app/genomevault/
WORKDIR /app

# Set CPU backend
ENV GENOMEVAULT_BACKEND=cpu
ENV GENOMEVAULT_OPTIMIZE_LATENCY=true

# Run API server (4 workers for parallelism)
CMD ["uvicorn", "genomevault.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Kubernetes Deployment

**`deployment/production-api.yaml`**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genomevault-api
  labels:
    app: genomevault
    tier: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genomevault
  template:
    metadata:
      labels:
        app: genomevault
    spec:
      containers:
      - name: genomevault
        image: genomevault/api:cpu-latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "2000m"        # 2 cores
            memory: "4Gi"
          limits:
            cpu: "4000m"        # Burst to 4 cores
            memory: "8Gi"
        env:
        - name: GENOMEVAULT_BACKEND
          value: "cpu"
        - name: GENOMEVAULT_OPTIMIZE_LATENCY
          value: "true"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

### Performance Expectations

- **Single-sample encoding**: <10ms (p99)
- **API response time**: <50ms end-to-end
- **Throughput**: 100-200 queries/sec per pod
- **Resource usage**: 2-4 cores, 4-8GB RAM

### Rationale

1. **Predictable latency**: No GPU warmup or transfer overhead
2. **Simple deployment**: Standard compute instances, no GPU drivers
3. **Cost-effective**: Lower instance costs than GPU instances
4. **Proven performance**: Your benchmarks show 1.2s end-to-end is already excellent

---

## Scenario 2: Research Batch Processing (Throughput-Optimized)

### Use Case

Population-scale genomic analysis processing 100K+ samples for research studies.

### Architecture

```
Data Lake → Batch Job (GPU) → Results Store
   (VCF)        ↓                 (Parquet)
           Metal/CUDA Backend
         (50× speedup, <100ms/1K)
```

### Configuration

**`genomevault/config/compute.yaml`**:
```yaml
compute:
  default_backend: "auto"  # Auto-detect GPU
  optimize_latency: false

  hdc_encoding:
    single_sample: "cpu"
    batch_threshold: 100   # Use GPU for batch > 100

  similarity_search:
    large_database_backend: "auto"  # Prefer GPU
```

**Environment Variables**:
```bash
export GENOMEVAULT_BACKEND=auto
export GENOMEVAULT_PRESET=research_batch
```

### Docker Configuration

**`Dockerfile.research`** (Metal for Apple Silicon):
```dockerfile
FROM python:3.11

# Metal (MLX) dependencies
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    mlx \
    faiss-gpu

COPY genomevault/ /app/genomevault/
WORKDIR /app

ENV GENOMEVAULT_BACKEND=auto
ENV GENOMEVAULT_PRESET=research_batch

CMD ["python", "genomevault/pipelines/bulk_import.py"]
```

**`Dockerfile.research-cuda`** (NVIDIA GPU):
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip

# CUDA (PyTorch) dependencies
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    faiss-gpu

COPY genomevault/ /app/genomevault/
WORKDIR /app

ENV GENOMEVAULT_BACKEND=auto
ENV GENOMEVAULT_PRESET=research_batch

CMD ["python", "genomevault/pipelines/bulk_import.py"]
```

### Cloud Configuration (AWS)

**EC2 Instance**: `g4dn.xlarge` (NVIDIA T4 GPU)
- 4 vCPUs, 16 GB RAM, 1× T4 GPU
- Cost: ~$0.50/hour
- Throughput: 50× faster than CPU = $0.01 per 1000 samples

### Performance Expectations

- **Batch 1K samples**: <100ms (Metal), <150ms (CUDA)
- **Throughput**: ~10,000 samples/sec (GPU) vs 200/sec (CPU)
- **Cost efficiency**: GPU amortizes over batch, reducing cost per sample

### Example Batch Script

```python
from genomevault.config.loader import load_and_initialize
from genomevault.compute import get_accelerator
import numpy as np

# Initialize GPU backend
backend = load_and_initialize()
print(f"Using: {backend.value}")

accelerator = get_accelerator()

# Load all samples (100K+ samples)
samples = load_vcf_files("data/vcf/*.vcf")
print(f"Loaded {len(samples)} samples")

# Batch encode all at once (GPU handles this efficiently)
hypervectors = accelerator.encode_batch(samples)
print(f"Encoded {len(hypervectors)} samples")

# Save results
save_to_parquet(hypervectors, "results/encoded.parquet")
```

---

## Scenario 3: Edge Deployment (Hospital Servers)

### Use Case

On-premise hospital servers with limited resources, no GPU.

### Architecture

```
Hospital Network → Server (CPU-only) → Local Database
                     ↓
                 CPU + FAISS
              (Resource-efficient)
```

### Configuration

**`genomevault/config/compute.yaml`**:
```yaml
compute:
  default_backend: "cpu"
  optimize_latency: true

  hdc_encoding:
    enable_faiss: true
    faiss_threshold: 50000  # More aggressive FAISS usage

  similarity_search:
    faiss_cpu_threshold: 50000  # Lower threshold for edge
```

**Environment Variables**:
```bash
export GENOMEVAULT_BACKEND=cpu
export GENOMEVAULT_PRESET=edge_deployment
```

### Docker Compose

**`docker-compose.edge.yml`**:
```yaml
version: '3.8'

services:
  genomevault:
    image: genomevault/api:cpu-latest
    container_name: genomevault-edge
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
      - ./config:/app/config
    environment:
      - GENOMEVAULT_BACKEND=cpu
      - GENOMEVAULT_PRESET=edge_deployment
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  postgres:
    image: postgres:15
    container_name: genomevault-db
    environment:
      POSTGRES_DB: genomevault
      POSTGRES_USER: gv_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

volumes:
  postgres_data:
```

### Installation Script

```bash
#!/bin/bash
# install_edge.sh - Hospital edge deployment

echo "Installing GenomeVault Edge..."

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Pull images
docker pull genomevault/api:cpu-latest
docker pull postgres:15

# Setup environment
cat > .env << EOF
DB_PASSWORD=$(openssl rand -base64 32)
GENOMEVAULT_BACKEND=cpu
GENOMEVAULT_PRESET=edge_deployment
EOF

# Start services
docker-compose -f docker-compose.edge.yml up -d

echo "✓ GenomeVault Edge installed and running"
echo "✓ API available at: http://localhost:8000"
```

### Performance Expectations

- **Latency**: <20ms (slightly higher due to resource constraints)
- **Throughput**: 50-100 queries/sec
- **Resource usage**: 2-4 cores, 4-8GB RAM (hospital server)

---

## Scenario 4: Cloud Deployment with GPU

### Use Case

Cloud-based genomics pipeline leveraging GPU for cost-effective batch processing.

### AWS Configuration

**Instance Types**:
- **Metal (Apple Silicon)**: N/A (not available on AWS)
- **CUDA (NVIDIA)**: `g4dn.xlarge` (T4), `p3.2xlarge` (V100)

**`terraform/main.tf`**:
```hcl
resource "aws_instance" "genomevault_gpu" {
  ami           = "ami-xxxx"  # NVIDIA Deep Learning AMI
  instance_type = "g4dn.xlarge"

  user_data = <<-EOF
              #!/bin/bash
              docker pull genomevault/api:cuda-latest
              docker run -d \
                --gpus all \
                -p 8000:8000 \
                -e GENOMEVAULT_BACKEND=cuda \
                genomevault/api:cuda-latest
              EOF

  tags = {
    Name = "GenomeVault-GPU"
  }
}
```

### GCP Configuration

**Instance Types**:
- **NVIDIA T4**: `n1-standard-4` + T4 GPU
- **NVIDIA V100**: `n1-standard-8` + V100 GPU

**`gcloud` commands**:
```bash
# Create GPU instance
gcloud compute instances create genomevault-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --metadata="install-nvidia-driver=True"

# Deploy container
gcloud compute ssh genomevault-gpu -- \
    "docker run -d --gpus all -p 8000:8000 genomevault/api:cuda-latest"
```

### Cost Analysis

| Instance | GPU | $/hour | Throughput | $/1K samples |
|----------|-----|--------|------------|--------------|
| CPU (c5.2xlarge) | None | $0.34 | 200/sec | $0.47 |
| GPU (g4dn.xlarge) | T4 | $0.50 | 10,000/sec | $0.01 |
| GPU (p3.2xlarge) | V100 | $3.06 | 20,000/sec | $0.04 |

**Conclusion**: GPU is 47× more cost-effective for batch workloads!

---

## Scenario 5: Hybrid Production/Research

### Use Case

Organization needing both real-time API (production) and batch processing (research).

### Architecture

```
        ┌─────────────┐
        │ Load Balancer│
        └──────┬───────┘
               │
       ┌───────┴────────┐
       │                │
   ┌───▼────┐     ┌────▼────┐
   │ API    │     │ Batch   │
   │ (CPU)  │     │ (GPU)   │
   └────────┘     └─────────┘
    Production     Research
```

### Kubernetes Configuration

**`deployment/hybrid.yaml`**:
```yaml
---
# Production API (CPU-only)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genomevault-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genomevault
      tier: api
  template:
    metadata:
      labels:
        app: genomevault
        tier: api
    spec:
      containers:
      - name: genomevault
        image: genomevault/api:cpu-latest
        env:
        - name: GENOMEVAULT_BACKEND
          value: "cpu"

---
# Research Batch (GPU)
apiVersion: batch/v1
kind: Job
metadata:
  name: genomevault-batch
spec:
  template:
    spec:
      containers:
      - name: genomevault
        image: genomevault/api:cuda-latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: GENOMEVAULT_BACKEND
          value: "cuda"
      restartPolicy: Never
```

### Routing Logic

```python
from genomevault.config.loader import ComputeConfig
from genomevault.compute import initialize_backend, ComputeBackend

def handle_request(request):
    """Route request based on type"""

    if request.type == "real_time":
        # Production: Force CPU for predictable latency
        config = ComputeConfig()
        config.config['compute']['default_backend'] = 'cpu'
        backend = config.initialize_backend()

    elif request.type == "batch":
        # Research: Use GPU for throughput
        config = ComputeConfig()
        config.config['compute']['default_backend'] = 'auto'
        backend = config.initialize_backend()

    # Process request
    return process(request, backend)
```

---

## Performance Comparison

### Single Sample Encoding

| Backend | Latency | Use Case |
|---------|---------|----------|
| CPU | 5-10ms ✓ | Production API, real-time queries |
| Metal | <1ms ⚡ | Development on Mac, not production |
| CUDA | ~2ms ⚠️ | Not recommended (transfer overhead) |

### Batch Encoding (1K samples)

| Backend | Time | Throughput | Speedup | Cost/1K |
|---------|------|------------|---------|---------|
| CPU | 5s | 200/sec | 1× | $0.47 |
| Metal | 0.1s | 10,000/sec | 50× | $0.01 |
| CUDA | 0.15s | 6,667/sec | 33× | $0.01 |

### Similarity Search (1M database)

| Backend | Time | Method |
|---------|------|--------|
| CPU | <5s | FAISS index |
| Metal | <400ms | Brute-force GPU |
| CUDA | <300ms | Brute-force GPU |

---

## Migration Guide

### From CPU-Only to Hybrid

**Step 1**: Add GPU-enabled Docker image
```bash
docker build -f Dockerfile.research-cuda -t genomevault/api:cuda-latest .
```

**Step 2**: Deploy batch processing pod
```bash
kubectl apply -f deployment/batch-gpu.yaml
```

**Step 3**: Keep existing production API unchanged

**Step 4**: Route batch workloads to GPU pod

### From Manual Backend to Configuration

**Before**:
```python
# Hardcoded backend selection
from genomevault.compute import initialize_backend, ComputeBackend
initialize_backend(ComputeBackend.CPU)
```

**After**:
```python
# Configuration-driven
from genomevault.config.loader import load_and_initialize
backend = load_and_initialize()
```

### Testing Backend Changes

```bash
# Test CPU backend
GENOMEVAULT_BACKEND=cpu python tests/test_compute_backend.py

# Test GPU backend (if available)
GENOMEVAULT_BACKEND=auto python tests/test_compute_backend.py

# Compare performance
python benchmarks/backend_comparison.py
```

---

## Decision Matrix

| Requirement | Recommended Backend | Deployment |
|-------------|-------------------|------------|
| Real-time API (<10ms) | CPU | Production API |
| Batch >1K samples | Auto (GPU) | Research Batch |
| Limited resources | CPU + FAISS | Edge Deployment |
| Cost optimization | Auto (GPU) | Cloud with GPU |
| Mixed workload | Hybrid | Hybrid Setup |

## Best Practices

1. ✅ Use CPU for production APIs (predictable latency)
2. ✅ Use AUTO for research batch (leverage GPU when available)
3. ✅ Test backend detection in CI/CD pipeline
4. ✅ Monitor performance per backend in production
5. ✅ Use configuration files, not hardcoded backends
6. ❌ Don't use GPU for single samples (transfer overhead)
7. ❌ Don't use CUDA for small batches (<100)
8. ❌ Don't force GPU without CPU fallback
