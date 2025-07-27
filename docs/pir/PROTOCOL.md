# PIR Protocol Specification

## Overview

GenomeVault implements a 2-server Information-Theoretic Private Information Retrieval (IT-PIR) protocol based on XOR secret sharing. This protocol provides perfect information-theoretic privacy as long as the servers do not collude.

## Mathematical Foundation

### Protocol Description

Given a database D of n items, where each item is of size l bits:

1. **Client Query Generation**:
   - To retrieve item at index j, client generates:
   - Random vector r ∈ {0,1}^n
   - Query vectors: q₁ = r, q₂ = r ⊕ eⱼ
   - Where eⱼ is the unit vector with 1 at position j

2. **Server Response**:
   - Server i computes: aᵢ = Σₖ qᵢ[k] · D[k]
   - Returns the sum of database items where query bit is 1

3. **Client Reconstruction**:
   - Client computes: D[j] = a₁ ⊕ a₂

### Security Analysis

**Theorem**: The protocol provides perfect information-theoretic privacy against any single server.

**Proof**:
- Each server sees only one query vector qᵢ
- Since r is uniformly random, qᵢ is uniformly distributed
- No information about j can be inferred from a single query vector

## Adversary Model

### Assumptions
1. **Non-collusion**: The two servers do not share query information
2. **Honest-but-curious**: Servers follow protocol but try to learn query patterns
3. **Network adversary**: May observe network traffic patterns

### Threat Mitigations
- **Padding**: All queries/responses use fixed sizes
- **Timing**: Constant-time operations prevent timing attacks
- **Traffic analysis**: Dummy queries maintain constant traffic patterns

## Implementation Details

### Message Formats

**Query Message**:
```json
{
  "query_id": "uuid",
  "query_vector": [0, 1, 0, ...],  // Binary vector
  "vector_size": 1000000,
  "timestamp": 1234567890,
  "protocol_version": "1.0"
}
```

**Response Message**:
```json
{
  "query_id": "uuid",
  "server_id": "server_1",
  "response_data": "base64_encoded_xor_result",
  "computation_time_ms": 42.5,
  "timestamp": 1234567891
}
```

### Database Encoding

- Genomic data is chunked into fixed-size blocks (default: 1KB)
- Each block is indexed by genomic position
- Sparse regions use zero-padding for consistency

### Performance Optimizations

1. **Batching**: Multiple queries can be combined
2. **Caching**: Frequently accessed regions cached client-side
3. **Compression**: XOR results are compressed before transmission

## Security Parameters

- **Database size (n)**: 10^6 to 10^9 items
- **Item size (l)**: 1024 bits (1KB blocks)
- **Query vector sparsity**: ~10 non-zero entries for efficiency
- **Padding overhead**: 20% additional bandwidth

## Privacy Guarantees

Given honesty probability q for each server:
- Privacy failure probability: P_fail = (1-q)²
- For q = 0.98 (HIPAA TS nodes): P_fail = 0.0004
- For q = 0.95 (generic nodes): P_fail = 0.0025

## Protocol Extensions

### Multi-server (k-out-of-n)
- Extend to k servers using polynomial secret sharing
- Tolerates up to k-1 colluding servers

### Batch Queries
- Amortize communication costs
- Single round for multiple retrievals

## Compliance

- HIPAA-compliant when using certified TS nodes
- Audit logs maintain query metadata (not content)
- No persistent storage of query vectors
