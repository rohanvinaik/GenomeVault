# GenomeVault API Gateway

A comprehensive FastAPI gateway implementing OpenAPI specifications for the GenomeVault privacy-preserving genomic computing platform.

## Overview

The API Gateway provides unified access to GenomeVault's core services with comprehensive security, monitoring, and real-time capabilities.

## Architecture

```
genomevault/api/gateway/
├── __init__.py              # Main gateway application export
├── main.py                  # FastAPI application with middleware stack
├── README.md               # This documentation
├── models/                 # Pydantic models for all endpoints
│   ├── __init__.py
│   ├── base.py            # Common base models and utilities
│   ├── algorithms.py      # Algorithm marketplace models
│   ├── health.py          # Health check models
│   ├── models.py          # Federated learning model models
│   ├── pipelines.py       # Pipeline management models
│   ├── proofs.py          # Zero-knowledge proof models
│   ├── queries.py         # Query and PIR models
│   ├── specialized.py     # Section 5.2.4 specialized endpoints
│   ├── vectors.py         # Vector operation models
│   └── websockets.py      # WebSocket communication models
├── routes/                # FastAPI route handlers
│   ├── __init__.py
│   ├── algorithms.py      # Algorithm marketplace endpoints
│   ├── health.py          # Health monitoring endpoints
│   ├── models.py          # Federated learning endpoints
│   ├── pipelines.py       # Pipeline management endpoints
│   ├── proofs.py          # Zero-knowledge proof endpoints
│   ├── queries.py         # Query and PIR endpoints
│   ├── specialized.py     # Specialized endpoints (Section 5.2.4)
│   └── vectors.py         # Vector operation endpoints
├── middleware/            # Custom middleware components
│   ├── __init__.py
│   ├── authentication.py # Authentication and authorization
│   ├── error_handling.py  # Comprehensive error handling
│   ├── logging.py         # Request/response logging
│   ├── rate_limiting.py   # Token bucket rate limiting
│   └── security.py        # Security headers and protections
└── websockets/           # Real-time WebSocket handlers
    ├── __init__.py
    ├── connection_manager.py # WebSocket connection management
    └── main.py           # WebSocket routing and message handling
```

## Key Features

### 🔐 Security & Authentication
- **Multi-method Authentication**: API keys, OAuth2, JWT tokens
- **Fine-grained Authorization**: Role-based access control
- **Rate Limiting**: Token bucket algorithm with user-tier support
- **Security Headers**: CSP, HSTS, XSS protection
- **Input Sanitization**: XSS, SQL injection, path traversal protection

### 📊 Comprehensive API Endpoints

#### Main Route Categories
- **`/health`** - System health monitoring and readiness checks
- **`/pipelines`** - Processing pipeline management
- **`/vectors`** - Hypervector operations (encode, compare, search, store)
- **`/proofs`** - Zero-knowledge proof generation and verification
- **`/queries`** - PIR queries and database access
- **`/models`** - Federated learning model management
- **`/algorithms`** - Algorithm marketplace operations

#### Specialized Endpoints (Section 5.2.4)
- **`POST /specialized/topology`** → `{nearestLNs: [...], tsNodes: [...]}`
- **`POST /specialized/credit/vault/redeem`** → `{invoiceId, creditsBurned}`
- **`POST /specialized/audit/challenge`** → `{challenger, target, epoch, resultHash}`

### 🚀 Real-time Communication
- **WebSocket Support**: Real-time updates and notifications
- **Connection Management**: Automatic cleanup and health monitoring
- **Subscription System**: Subscribe to pipeline status, training metrics, etc.
- **Message Types**: Connect, subscribe, data updates, alerts, errors

### 📈 Monitoring & Observability
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Performance Metrics**: Request timing, throughput, error rates
- **Health Checks**: Kubernetes liveness/readiness probes
- **Error Tracking**: Comprehensive error categorization and reporting

### 🔒 Privacy Features
- **Differential Privacy**: Configurable noise injection
- **K-Anonymity**: High-dimensional vector encoding
- **Zero-Knowledge Proofs**: Cryptographic verification without data exposure
- **PIR Queries**: Private database access without query pattern leakage

## API Documentation

### Health Monitoring

```bash
# Basic health check
GET /health

# Detailed health with metrics
GET /health/detailed

# Kubernetes probes
GET /health/liveness
GET /health/readiness
```

### Vector Operations

```bash
# Encode genomic data to hypervector
POST /vectors/encode
{
  "variants": [{"chrom": "1", "pos": 1234567, "ref": "A", "alt": "T"}],
  "dimension": 8192,
  "encoding_type": "unified"
}

# Compare two vectors
POST /vectors/compare
{
  "vector1_id": "vec_123",
  "vector2": [1, -1, 1, 1, -1],
  "metrics": ["hamming", "cosine"]
}

# Search similar vectors
POST /vectors/search
{
  "query_vector": [1, -1, 1, 1, -1],
  "top_k": 10,
  "similarity_threshold": 0.7
}
```

### Specialized Operations

```bash
# Network topology discovery
POST /specialized/topology
{
  "client_location": {"lat": 37.7749, "lng": -122.4194},
  "max_nodes": 10,
  "optimize_for": "latency"
}

# Credit redemption
POST /specialized/credit/vault/redeem
{
  "vault_id": "vault_123",
  "credit_type": "compute",
  "amount": 100,
  "vault_signature": "0x..."
}

# Audit challenge
POST /specialized/audit/challenge
{
  "challenge_type": "proof_verification",
  "target_node": "node_123",
  "challenger_signature": "0x...",
  "epoch": 1005
}
```

### WebSocket Communication

```javascript
// Connect to WebSocket
const ws = new WebSocket('wss://api.genomevault.io/gateway/v1/ws/');

// Subscribe to pipeline status
ws.send(JSON.stringify({
  message_id: "msg_123",
  message_type: "subscribe",
  timestamp: new Date().toISOString(),
  data: {
    subscription_type: "pipeline_status",
    resource_id: "pipeline_abc123"
  }
}));
```

## Security Configuration

### Authentication Methods

1. **API Key Authentication**
   ```bash
   curl -H "X-API-Key: gv_your_api_key" https://api.genomevault.io/gateway/v1/health
   ```

2. **OAuth2 Bearer Token**
   ```bash
   curl -H "Authorization: Bearer your_token" https://api.genomevault.io/gateway/v1/vectors/encode
   ```

### Rate Limiting Tiers

| Tier | Requests/Hour | Use Cases |
|------|--------------|-----------|
| Anonymous | 100 | Public health checks |
| Standard | 1,000 | Basic research |
| Clinical | 10,000 | Clinical applications |
| Research | 50,000 | Large-scale research |

## Error Handling

All errors return standardized JSON responses:

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
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Deployment

### Docker
```bash
docker build -t genomevault/api-gateway .
docker run -p 8000:8000 genomevault/api-gateway
```

### Kubernetes
```bash
kubectl apply -f deployment/kubernetes/genomevault-gateway.yaml
```

### Development
```bash
cd /Users/rohanvinaik/genomevault
uvicorn genomevault.api.gateway.main:app --reload --port 8000
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/genomevault

# Authentication
JWT_SECRET_KEY=your-secret-key
ENABLE_MFA=true

# CORS
GENOMEVAULT_CORS_ORIGINS=http://localhost:3000,https://app.genomevault.io

# Rate Limiting
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
STRUCTURED_LOGGING=true
```

## Monitoring

### Metrics Available
- Request count and rate
- Response time percentiles
- Error rates by endpoint
- Active WebSocket connections
- Authentication success/failure rates
- Rate limit violations

### Health Checks
- Database connectivity
- External service availability
- Memory and CPU usage
- WebSocket connection health

## Privacy Guarantees

### Mathematical Privacy
- **K-Anonymity**: Hypervector encoding provides k≥10 anonymity
- **Differential Privacy**: ε-differential privacy with configurable ε
- **Zero-Knowledge**: Cryptographic proofs reveal no additional information

### Data Handling
- Input data is not logged or stored
- Vector representations cannot be reverse-engineered
- PIR queries hide access patterns
- Audit trails use cryptographic hashing

## Development Status

✅ **Completed Features:**
- FastAPI application structure
- Comprehensive Pydantic models
- Authentication and authorization middleware
- Rate limiting with token bucket algorithm
- Error handling and logging middleware
- Security middleware (headers, input sanitization)
- Health monitoring endpoints
- Vector operations (encode, compare, search, store)
- Specialized endpoints (topology, credit redemption, audit challenges)
- WebSocket connection management
- OpenAPI documentation generation

🚧 **Implementation Needed:**
- Route handler business logic (currently placeholder implementations)
- Database integration for persistent storage
- Redis integration for caching and rate limiting
- External service integrations (PIR engines, ZK provers)
- WebSocket subscription management
- Background task processing
- Comprehensive test coverage

## Usage Examples

See the main GenomeVault documentation and the `examples/` directory for comprehensive usage examples demonstrating:

- End-to-end genomic data processing workflows
- Privacy-preserving analysis pipelines
- Federated learning model training
- Algorithm marketplace integration
- Real-time monitoring and alerts

## Support

For questions, issues, or contributions:
- 📧 Email: support@genomevault.io
- 🐛 Issues: GitHub Issues
- 📖 Docs: https://docs.genomevault.io
- 💬 Community: GenomeVault Discord

## License

Apache 2.0 License - see LICENSE file for details.
