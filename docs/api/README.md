# GenomeVault API Documentation

Welcome to the GenomeVault API documentation. This guide provides comprehensive information about using our privacy-preserving genomic computing platform.

## Overview

GenomeVault is a privacy-preserving genomic computing platform that uses:

- **Hyperdimensional Computing (HDC)**: High-dimensional vectors for privacy-preserving genomic encoding
- **Private Information Retrieval (PIR)**: Query databases without revealing what you're looking for
- **Zero-Knowledge Proofs**: Verify computations without revealing the underlying data
- **Differential Privacy**: Mathematical bounds on information leakage
- **Federated Learning**: Distributed machine learning with privacy guarantees

## Quick Start

### 1. Authentication

Get your API key from the [GenomeVault Console](https://console.genomevault.io):

```bash
curl -H "X-API-Key: your-api-key" https://api.genomevault.io/v1/health
```

### 2. Basic Example

Encode genomic variants into privacy-preserving hypervectors:

```bash
curl -X POST https://api.genomevault.io/v1/hv/encode \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "variants": [
      {
        "chrom": "1",
        "pos": 1234567,
        "ref": "A",
        "alt": "T",
        "impact": "missense"
      }
    ],
    "dim": 8192,
    "binary": false
  }'
```

### 3. Response

```json
{
  "dim": 8192,
  "binary": false,
  "vector": [0.12, -0.34, 0.56, ...],
  "privacy_level": "k-anonymous",
  "compression_ratio": 87.3
}
```

## API Reference

### Base URL
- Production: `https://api.genomevault.io/v1`
- Staging: `https://staging-api.genomevault.io/v1`

### Authentication

#### API Key Authentication
Include your API key in the request header:
```http
X-API-Key: your-api-key
```

#### OAuth2 Authentication
Use OAuth2 with PKCE for web applications:
```http
Authorization: Bearer your-oauth-token
```

**Scopes:**
- `genomic:read` - Read genomic data and perform basic encoding
- `pir:query` - Execute private information retrieval queries
- `zk:prove` - Generate and verify zero-knowledge proofs
- `clinical:analyze` - Perform clinical genomic analysis
- `admin:manage` - Administrative operations

### Rate Limits

All API requests are subject to rate limiting:

| Tier | Requests/Hour | Use Case |
|------|---------------|----------|
| Standard | 1,000 | Development & testing |
| Clinical | 10,000 | Clinical applications |
| Research | 50,000 | Large-scale research |

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Reset time (Unix timestamp)

### Endpoints

#### Health Check
```http
GET /v1/health
```

Check system health and service availability.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "pir_engine": "healthy",
    "zk_prover": "healthy"
  }
}
```

#### Hypervector Encoding
```http
POST /v1/hv/encode
```

Encode genomic data into privacy-preserving hypervectors.

**Request Body:**
```json
{
  "variants": [
    {
      "chrom": "1",
      "pos": 1234567,
      "ref": "A",
      "alt": "T",
      "impact": "missense",
      "quality": 99.5
    }
  ],
  "dim": 8192,
  "binary": false
}
```

**Parameters:**
- `variants` (array): Genomic variants to encode
- `numeric` (array): Alternative to variants - numeric feature array
- `dim` (integer): Hypervector dimension (1024-100000, default: 8192)
- `binary` (boolean): Return binary (-1/+1) or continuous values

**Response:**
```json
{
  "dim": 8192,
  "binary": false,
  "vector": [0.12, -0.34, 0.56, 0.78],
  "privacy_level": "k-anonymous",
  "compression_ratio": 87.3
}
```

#### Private Information Retrieval
```http
POST /v1/pir/query
```

Execute PIR query without revealing the query index to the server.

**Request Body:**
```json
{
  "index": 42,
  "query_id": "unique-query-identifier",
  "timeout_seconds": 30
}
```

**Response:**
```json
{
  "index": 42,
  "item_base64": "YWxwaGE=",
  "privacy_proof": "zk_proof_hash_example",
  "query_time_ms": 125
}
```

#### Zero-Knowledge Proofs
```http
POST /v1/zk/prove
```

Generate cryptographic proofs of computation validity.

**Request Body:**
```json
{
  "proof_type": "genomic",
  "public_inputs": {
    "population": "EUR",
    "analysis_type": "gwas"
  },
  "private_inputs_hash": "sha256_hash_of_private_data"
}
```

**Response:**
```json
{
  "proof_id": "proof_12345",
  "proof_data": "zk_snark_proof_hex",
  "verification_key": "verification_key_hex",
  "public_signals": ["signal1", "signal2"],
  "validity_period_hours": 24
}
```

#### Clinical Analysis
```http
POST /v1/clinical/analyze
```

Perform HIPAA-compliant clinical genomic analysis.

**Request Body:**
```json
{
  "patient_id_hash": "sha256_hash",
  "variants": [
    {
      "gene": "BRCA1",
      "variant": "c.68_69delAG",
      "classification": "pathogenic",
      "evidence_level": "A"
    }
  ],
  "analysis_type": "risk_assessment",
  "population_reference": "gnomAD"
}
```

**Response:**
```json
{
  "analysis_id": "analysis_67890",
  "risk_score": 0.85,
  "confidence_interval": [0.78, 0.92],
  "recommendations": [
    "Consider genetic counseling",
    "Regular screening recommended"
  ],
  "audit_trail_hash": "audit_hash_example"
}
```

## Error Handling

All errors return a standardized format with PHI-safe messages:

```json
{
  "type": "ValidationError",
  "code": "GV_INVALID_INPUT",
  "message": "Invalid genomic coordinate format",
  "details": {
    "request_id": "req_1234567890",
    "field": "variants[0].chrom",
    "allowed_values": ["1", "2", "3", "...", "22", "X", "Y", "M"]
  },
  "errors": [
    {
      "field": "variants[0].chrom",
      "message": "Invalid genomic coordinate format",
      "code": "GV_VALIDATION_ERROR"
    }
  ],
  "request_id": "req_1234567890",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `GV_AUTHENTICATION_ERROR` | Invalid API key | 401 |
| `GV_AUTHORIZATION_ERROR` | Insufficient permissions | 403 |
| `GV_VALIDATION_ERROR` | Request validation failed | 422 |
| `GV_RATE_LIMITED` | Rate limit exceeded | 429 |
| `GV_INVALID_GENOMIC_COORDINATE` | Invalid genomic coordinate | 400 |
| `GV_PIR_QUERY_FAILED` | PIR query execution failed | 400 |
| `GV_PHI_DETECTED` | Protected health information detected | 400 |
| `GV_SERVICE_UNAVAILABLE` | Service temporarily unavailable | 503 |

## Privacy Model

### Mathematical Guarantees

GenomeVault provides privacy through mathematical properties rather than traditional encryption:

1. **k-Anonymity**: Hypervectors ensure each data point is indistinguishable from k-1 others
2. **Differential Privacy**: Mathematical bounds on information leakage (ε-differential privacy)
3. **Information-Theoretic Security**: PIR provides unconditional privacy guarantees

### Data Processing

- **No Data Storage**: Input genomic data is processed in-memory and never stored
- **Audit Trails**: All operations are cryptographically logged for compliance
- **Zero-Knowledge**: Proofs verify computations without revealing data
- **Federated Learning**: Models are trained without centralizing data

### Compliance

- **HIPAA**: All PHI processing meets HIPAA requirements
- **GDPR**: Full compliance with EU privacy regulations
- **SOC 2 Type II**: Annual security audits and certifications
- **ISO 27001**: Information security management standards

## SDKs and Tools

### Python SDK
```bash
pip install genomevault-sdk
```

```python
from genomevault_sdk import GenomeVaultClient

client = GenomeVaultClient(api_key="your-api-key")

# Encode variants
variants = [{"chrom": "1", "pos": 1234567, "ref": "A", "alt": "T"}]
result = await client.encode_variants(variants)
print(f"Encoded to {result.dim}-dimensional vector")
```

### JavaScript/TypeScript SDK
```bash
npm install @genomevault/sdk
```

```typescript
import { GenomeVaultClient } from '@genomevault/sdk';

const client = new GenomeVaultClient({
  apiKey: 'your-api-key'
});

// Encode variants
const variants = [{ chrom: '1', pos: 1234567, ref: 'A', alt: 'T' }];
const result = await client.encodeVariants(variants);
console.log(`Encoded to ${result.dim}-dimensional vector`);
```

### CLI Tool
```bash
pip install genomevault-cli

# Configure API key
gv config set-api-key your-api-key

# Check health
gv health

# Encode variants from VCF
gv encode variants variants.vcf --output encoded.json

# Execute PIR query
gv pir query 42 --output retrieved.data
```

## Examples

See the [examples directory](./examples/) for complete working examples:

- [Basic encoding](./examples/basic-encoding.py)
- [PIR queries](./examples/pir-queries.js)
- [Clinical analysis](./examples/clinical-analysis.py)
- [Zero-knowledge proofs](./examples/zk-proofs.py)
- [Batch processing](./examples/batch-processing.py)

## Support

- **Documentation**: [https://docs.genomevault.io](https://docs.genomevault.io)
- **API Status**: [https://status.genomevault.io](https://status.genomevault.io)
- **Support**: [support@genomevault.io](mailto:support@genomevault.io)
- **Issues**: [GitHub Issues](https://github.com/genomevault/genomevault/issues)
- **Community**: [Discord Server](https://discord.gg/genomevault)

## Changelog

### v1.0.0 (2024-01-15)
- Initial public release
- Hypervector encoding API
- PIR query functionality
- Zero-knowledge proof generation
- Clinical analysis endpoints
- Python and JavaScript SDKs
- CLI tool
