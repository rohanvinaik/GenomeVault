# GenomeVault Analysis API Usage Guide

## Overview

The GenomeVault Analysis API provides a simple, privacy-preserving interface for genomic analysis. Upload your genome files and receive privacy-verified results with cryptographic guarantees.

## Table of Contents

- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Analysis Types](#analysis-types)
- [File Formats](#file-formats)
- [Privacy Guarantees](#privacy-guarantees)
- [Examples](#examples)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Quick Start

### Start the API Server

```bash
# Install dependencies
pip install -e ".[dev]"

# Start server
uvicorn genomevault.api.app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, access:
- **Swagger UI**: `http://localhost:8000/api/docs`
- **ReDoc**: `http://localhost:8000/api/redoc`

## Authentication

**Current Status**: Authentication is optional in development mode. For production deployment, enable OAuth2/API key authentication:

```bash
# Set API key (production)
export GENOMEVAULT_API_KEY="your-secret-key"
```

Include in requests:
```bash
curl -H "X-API-Key: your-secret-key" ...
```

## Endpoints

### 1. Submit Analysis

**POST** `/api/v1/analysis/submit`

Submit a genome file for privacy-preserving analysis.

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | Genomic data file (VCF, FASTQ, BAM) |
| `file_r2` | File | No | - | Paired-end FASTQ R2 file |
| `analysis_type` | String | Yes | - | Type of analysis (see [Analysis Types](#analysis-types)) |
| `reference_genome` | String | No | `GRCh38` | Reference genome assembly |
| `k_anonymity` | Integer | No | `3` | k-anonymity level (≥2) |
| `dimension` | Integer | No | `10000` | Hypervector dimension (1024-100000) |
| `enable_zk_proof` | Boolean | No | `true` | Generate zero-knowledge proof |
| `enable_blockchain` | Boolean | No | `false` | Record on blockchain |
| `enable_pir` | Boolean | No | `false` | Enable PIR query |
| `analysis_params` | JSON String | No | `{}` | Analysis-specific parameters |

**Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/submit" \
  -F "file=@patient_genome.vcf.gz" \
  -F "analysis_type=whole_genome" \
  -F "reference_genome=GRCh38" \
  -F "k_anonymity=3" \
  -F "dimension=10000" \
  -F "enable_zk_proof=true" \
  -F "enable_blockchain=false" \
  -F "analysis_params={}"
```

**Response:**

```json
{
  "analysis_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "queued",
  "message": "Analysis queued successfully. Use GET /api/v1/analysis/{analysis_id}/status to check progress."
}
```

### 2. Check Analysis Status

**GET** `/api/v1/analysis/{analysis_id}/status`

Get the current status of an analysis job.

**Example:**

```bash
curl "http://localhost:8000/api/v1/analysis/123e4567-e89b-12d3-a456-426614174000/status"
```

**Response:**

```json
{
  "analysis_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "processing",
  "progress_percent": 60.0,
  "current_stage": "hdc_encoding",
  "estimated_completion_seconds": 15.2
}
```

**Status Values:**
- `queued`: Analysis is waiting to be processed
- `processing`: Analysis is currently running
- `completed`: Analysis finished successfully
- `failed`: Analysis encountered an error

### 3. Get Analysis Results

**GET** `/api/v1/analysis/{analysis_id}/results`

Retrieve the results of a completed analysis.

**Example:**

```bash
curl "http://localhost:8000/api/v1/analysis/123e4567-e89b-12d3-a456-426614174000/results"
```

**Response:**

```json
{
  "analysis_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "success",
  "stages": [
    {
      "stage_name": "differential_encoding",
      "success": true,
      "duration_ms": 1360.5,
      "output": {
        "compression_ratio": 11.0,
        "num_differences": 292,
        "k_anonymity": 3
      }
    },
    {
      "stage_name": "hdc_encoding",
      "success": true,
      "duration_ms": 0.5,
      "output": {
        "dimension": 10000,
        "hypervector_id": "hv_123e4567-e89b-12d3-a456-426614174000"
      }
    },
    {
      "stage_name": "zk_proof",
      "success": true,
      "duration_ms": 740.2,
      "output": {
        "proof_id": "proof_123e4567-e89b-12d3-a456-426614174000",
        "verification_status": true
      }
    }
  ],
  "total_duration_ms": 2110.8,
  "compression_ratio": 264.0,
  "variants_analyzed": 120,
  "hypervector_id": "hv_123e4567-e89b-12d3-a456-426614174000",
  "hypervector_dimension": 10000,
  "zk_proof_id": "proof_123e4567-e89b-12d3-a456-426614174000",
  "zk_verification_status": true,
  "pir_query_result": null,
  "blockchain_tx_hash": null,
  "attestation_id": null,
  "analysis_results": {
    "analysis_type": "whole_genome",
    "note": "Analysis-specific results will be computed here based on analysis type"
  },
  "warnings": [],
  "recommendations": []
}
```

## Analysis Types

GenomeVault supports multiple types of genomic analysis:

| Analysis Type | Description | Use Case |
|---------------|-------------|----------|
| `whole_genome` | Complete genome sequencing analysis | Comprehensive genetic screening |
| `exome` | Exome sequencing (coding regions) | Disease gene identification |
| `targeted_panel` | Specific gene panels | Focused genetic testing |
| `pharmacogenomics` | Drug-gene interactions | Personalized medicine |
| `ancestry` | Population genetics | Ancestry and ethnicity analysis |
| `risk_assessment` | Disease risk scoring | Predictive health analytics |
| `carrier_screening` | Carrier status for genetic conditions | Family planning |
| `variant_pathogenicity` | Variant impact assessment | Clinical interpretation |

## File Formats

### Supported Formats

| Format | Extension | Description | Requirements |
|--------|-----------|-------------|--------------|
| **VCF** | `.vcf`, `.vcf.gz` | Variant Call Format | Direct encoding |
| **FASTQ** | `.fastq`, `.fastq.gz`, `.fq`, `.fq.gz` | Raw sequencing reads | Requires alignment tools |
| **BAM** | `.bam` | Binary alignment format | Requires variant calling |
| **SAM** | `.sam` | Sequence alignment format | Requires variant calling |

### FASTQ Processing Requirements

For FASTQ input, install bioinformatics tools:

```bash
conda install -c bioconda minimap2 samtools bcftools
```

### Paired-End FASTQ

For paired-end sequencing, provide both R1 and R2 files:

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/submit" \
  -F "file=@sample_R1.fastq.gz" \
  -F "file_r2=@sample_R2.fastq.gz" \
  -F "analysis_type=whole_genome"
```

### File Size Limits

- **Maximum file size**: 10 GB per file
- **Paired-end**: Both files combined must be ≤ 20 GB

## Privacy Guarantees

GenomeVault provides multiple layers of privacy protection:

### 1. Differential Encoding (k-anonymity)

**Default**: k=3

Your genome is encoded as differences from a pool of k reference genomes, ensuring k-anonymity. This means your data is indistinguishable from at least k-1 other individuals.

**Configuration:**
```bash
-F "k_anonymity=3"  # Minimum is 2, higher = more privacy
```

### 2. Compression (264× architectural, 38.4× empirical)

**Architectural (lossless):**
- Differential encoding: 11× compression
- HDC encoding: 24× additional compression
- **Total**: 264× architectural compression

**Empirical (space savings):**
- Raw VCF: 1,500 KB → Encoded: 39.06 KB
- **Actual savings**: 38.4× (97.4% reduction)

### 3. Zero-Knowledge Proofs

**Technology**: Groth16 (production-ready)

Proves properties of your genome (e.g., k-anonymity) without revealing the actual data.

**Proof characteristics:**
- Size: ~743 bytes
- Generation time: ~0.74s
- Verification: <100ms
- Constraints: 117,143

**Enable:**
```bash
-F "enable_zk_proof=true"
```

### 4. Private Information Retrieval (PIR)

**Protocol**: IT-PIR (Information-Theoretic)

Query a genomic database without revealing which record you're accessing.

**Security:**
- Information-theoretic privacy
- No computational assumptions
- Breach probability: 0.25%

**Enable:**
```bash
-F "enable_pir=true" \
-F "pir_database=clinvar"
```

### 5. Blockchain Attestation (Optional)

**Purpose**: Immutable audit trail

Records cryptographic hashes (not actual data) of analysis inputs/outputs on blockchain.

**What's recorded:**
- ✅ SHA-256 hash of input data
- ✅ SHA-256 hash of output data
- ✅ Timestamp and metadata
- ❌ **NOT** actual genomic variants
- ❌ **NOT** patient identifiers

**Enable:**
```bash
-F "enable_blockchain=true"
```

**Cost**: <$0.01 per attestation (Polygon) with batch mode

## Examples

### Example 1: Basic VCF Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/submit" \
  -F "file=@sample.vcf.gz" \
  -F "analysis_type=whole_genome" \
  -F "k_anonymity=3" \
  -F "dimension=10000"
```

### Example 2: Pharmacogenomics with ZK Proof

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/submit" \
  -F "file=@patient.vcf.gz" \
  -F "analysis_type=pharmacogenomics" \
  -F "enable_zk_proof=true" \
  -F "analysis_params={\"drugs\": [\"warfarin\", \"clopidogrel\"]}"
```

### Example 3: Paired-End FASTQ

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/submit" \
  -F "file=@sample_R1.fastq.gz" \
  -F "file_r2=@sample_R2.fastq.gz" \
  -F "analysis_type=exome" \
  -F "k_anonymity=5"
```

### Example 4: Complete Privacy Stack

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/submit" \
  -F "file=@sensitive_genome.vcf.gz" \
  -F "analysis_type=risk_assessment" \
  -F "k_anonymity=10" \
  -F "enable_zk_proof=true" \
  -F "enable_blockchain=true" \
  -F "enable_pir=true"
```

### Example 5: Polling for Results

```bash
#!/bin/bash

# Submit analysis
RESPONSE=$(curl -X POST "http://localhost:8000/api/v1/analysis/submit" \
  -F "file=@genome.vcf" \
  -F "analysis_type=whole_genome")

# Extract analysis ID
ANALYSIS_ID=$(echo $RESPONSE | jq -r '.analysis_id')
echo "Analysis ID: $ANALYSIS_ID"

# Poll status until completed
while true; do
  STATUS=$(curl "http://localhost:8000/api/v1/analysis/$ANALYSIS_ID/status" | jq -r '.status')
  echo "Status: $STATUS"

  if [ "$STATUS" == "completed" ] || [ "$STATUS" == "failed" ]; then
    break
  fi

  sleep 2
done

# Get results
curl "http://localhost:8000/api/v1/analysis/$ANALYSIS_ID/results" | jq '.'
```

### Example 6: Python Client

```python
import requests
import time
import json

API_BASE = "http://localhost:8000"

def submit_analysis(vcf_path: str, analysis_type: str = "whole_genome"):
    """Submit VCF file for analysis."""
    with open(vcf_path, "rb") as f:
        files = {"file": (vcf_path, f)}
        data = {
            "analysis_type": analysis_type,
            "k_anonymity": 3,
            "dimension": 10000,
            "enable_zk_proof": True,
        }

        response = requests.post(
            f"{API_BASE}/api/v1/analysis/submit",
            files=files,
            data=data
        )
        response.raise_for_status()
        return response.json()["analysis_id"]

def wait_for_completion(analysis_id: str, timeout: int = 300):
    """Wait for analysis to complete."""
    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(
            f"{API_BASE}/api/v1/analysis/{analysis_id}/status"
        )
        response.raise_for_status()
        status = response.json()

        print(f"Progress: {status['progress_percent']:.1f}% - {status['current_stage']}")

        if status["status"] == "completed":
            return True
        elif status["status"] == "failed":
            raise Exception("Analysis failed")

        time.sleep(2)

    raise TimeoutError("Analysis timed out")

def get_results(analysis_id: str):
    """Retrieve analysis results."""
    response = requests.get(
        f"{API_BASE}/api/v1/analysis/{analysis_id}/results"
    )
    response.raise_for_status()
    return response.json()

# Example usage
if __name__ == "__main__":
    # Submit analysis
    analysis_id = submit_analysis("patient_genome.vcf.gz", "whole_genome")
    print(f"Submitted analysis: {analysis_id}")

    # Wait for completion
    wait_for_completion(analysis_id)

    # Get results
    results = get_results(analysis_id)
    print(json.dumps(results, indent=2))

    # Extract key metrics
    print(f"\nCompression: {results['compression_ratio']:.1f}×")
    print(f"ZK Proof: {'Verified ✓' if results['zk_verification_status'] else 'Failed ✗'}")
    print(f"Duration: {results['total_duration_ms'] / 1000:.2f}s")
```

## Error Handling

### HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Analysis submitted successfully |
| 202 | Accepted | Analysis still processing |
| 400 | Bad Request | Invalid file format or parameters |
| 404 | Not Found | Analysis ID not found |
| 413 | Payload Too Large | File exceeds 10 GB limit |
| 422 | Validation Error | Invalid parameter values (e.g., k_anonymity < 2) |
| 500 | Server Error | Internal processing error |

### Error Response Format

```json
{
  "detail": "File type not supported. Allowed: .vcf, .vcf.gz, .fastq, .fastq.gz, .bam, .sam"
}
```

### Common Errors

**Invalid File Format:**
```json
{
  "detail": "File type not supported. Allowed: .vcf, .vcf.gz, .fastq, .fastq.gz, .fq, .fq.gz, .bam, .sam"
}
```

**Invalid Analysis Type:**
```json
{
  "detail": "Invalid analysis_type. Must be one of: whole_genome, exome, targeted_panel, pharmacogenomics, ancestry, risk_assessment, carrier_screening, variant_pathogenicity"
}
```

**Invalid JSON Parameters:**
```json
{
  "detail": "Invalid analysis_params JSON"
}
```

**Validation Error (k-anonymity too low):**
```json
{
  "detail": [
    {
      "loc": ["body", "k_anonymity"],
      "msg": "ensure this value is greater than or equal to 2",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

## Best Practices

### 1. File Upload

- **Compress files**: Use `.gz` for VCF/FASTQ to reduce upload time
- **Check file size**: Ensure files are under 10 GB limit
- **Validate format**: Use standard tools (bcftools, samtools) to validate before upload

### 2. Privacy Configuration

- **k-anonymity**: Use k=3 (default) for most applications, k=5+ for sensitive data
- **ZK proofs**: Enable for regulatory compliance and third-party verification
- **Blockchain**: Use only when immutable audit trail is required

### 3. Performance Optimization

- **Dimension**: Use 10,000 (default) for balanced accuracy/speed
  - 1,000: Faster but less accurate
  - 100,000: Slower but highest accuracy
- **Disable optional features**: Turn off ZK/PIR/blockchain for development testing

### 4. Production Deployment

```bash
# Set production environment variables
export GENOMEVAULT_BACKEND=auto           # Enable GPU if available
export GENOMEVAULT_API_KEY="secret-key"   # Enable authentication

# Use production ASGI server
gunicorn genomevault.api.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 600
```

### 5. Resource Management

**In-memory job storage** (current implementation) is suitable for development only. For production:

```python
# Use Redis for job tracking
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Store results in database or S3
# Store hypervectors in vector database (FAISS/Milvus)
```

### 6. Monitoring

Track key metrics:
- Analysis submission rate
- Average processing time per stage
- Success/failure rates
- Queue depth
- Storage usage

## Rate Limits

**Current**: No rate limits in development mode

**Production recommendation**:
- 100 requests/hour per API key
- 10 concurrent analyses per user
- 1 TB storage per account

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/yourusername/genomevault/issues
- **Documentation**: See `CLAUDE.md` for complete reference
- **Academic Paper**: `docs/GenomeVault_Academic_Paper.pdf`

## Changelog

### v1.0.0 (October 2025)
- Initial release of Analysis API
- Support for VCF, FASTQ, BAM, SAM formats
- Differential encoding with k-anonymity
- HDC encoding (264× compression)
- ZK proof generation (Groth16)
- PIR query support (IT-PIR)
- Blockchain attestation (optional)
- Complete privacy-preserving pipeline

---

**Last Updated**: October 2025
**API Version**: v1.0.0
