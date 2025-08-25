# GenomeVault Cost Analysis

## Executive Summary

**Bottom Line Costs (10K queries/day = 300K/month)**:
- **PIR**: $35-2,262/month (monolithic) or $910/month (sharded 10M)
- **ZK Proofs**: $67-3,968/month (depending on backend and complexity)
- **Combined**: $163-3,439/month for typical deployments
- **Note**: Sharding significantly reduces costs for large databases

## Private Information Retrieval (PIR) Costs

### PIR Trust Models
- **CPIR (Computational PIR)**: Single-server setup relying on computational hardness assumptions (e.g., LWE, homomorphic encryption). Server sees encrypted queries but cannot decrypt them.
- **IT-PIR (Information-Theoretic PIR)**: Multi-server setup providing unconditional privacy as long as servers don't collude. Typically requires 2-3 non-colluding servers.

### Pricing Methodology
- AWS us-east-1 pricing (2024)
- vCPU-hour: $0.042 (t3.large), $0.096 (m5.xlarge), $0.192 (r5.xlarge)
- Network egress: $0.09/GB
- Instance costs are fixed monthly
- Variable costs = compute + network per query

### Cost Breakdown

| Database Size | Scheme | Trust Model | Latency¹ | Instance³ | Pricing⁴ | Variable/Query | Fixed/Month | Total/Month (10K/day) |
|--------------|--------|-------------|----------|-----------|----------|----------------|-------------|----------------------|
| **100K rows** | CPIR | Computational | 0.59s | t3.medium | On-demand $0.042/hr | $0.000016 | $30 | **$35/mo** |
| **1M rows** | CPIR | Computational | 0.92s | t3.large | On-demand $0.042/hr | $0.00010 | $61 | **$91/mo** |
| **10M rows** | CPIR | Computational | 113s | r5.xlarge | On-demand $0.192/hr | $0.00693 | $183 | **$2,262/mo** |
| **100K rows** | IT-PIR | 2+ honest servers | 6.4s | 3×t3.large | On-demand $0.042/hr | $0.00027 | $183 | **$264/mo** |
| **1M rows** | IT-PIR | 2+ honest servers | 8.1s | 3×m5.xlarge | On-demand $0.096/hr | $0.00113 | $415 | **$754/mo** |

³ CPU credits sustainable for t3 at these volumes (see analysis below)
⁴ AWS us-east-1 on-demand pricing, no spot discount applied

**Evidence Sources:**
1. Latency: `benchmark_results/pir_benchmark.json` → `latency_ms` field
2. Network: `benchmark_results/pir_benchmark.json` → `network_bytes` field

### Detailed Cost Calculation

#### 100K CPIR (Single-Server)
```
Variable per query:
- Compute: 0.59s × $0.042/3600 = $0.0000069
- Network: 0.0001GB × $0.09 = $0.0000090
- Total variable: $0.0000159 ≈ $0.000016
- Trust: Computational privacy (server sees encrypted query)

Monthly (300K queries):
- Variable: 300,000 × $0.000016 = $4.80
- Fixed (t3.medium): $30
- Total: $35/month
```

#### 1M CPIR (Single-Server)
```
Variable per query:
- Compute: 0.92s × $0.042/3600 = $0.0000107
- Network: 0.001GB × $0.09 = $0.0000900
- Total variable: $0.0001007 ≈ $0.00010
- Trust: Computational privacy (server sees encrypted query)

Monthly (300K queries):
- Variable: 300,000 × $0.00010 = $30
- Fixed (t3.large): $61
- Total: $91/month
```

#### 10M CPIR (Single-Server)
```
Variable per query:
- Compute: 113s × $0.192/3600 = $0.006027
- Network: 0.01GB × $0.09 = $0.000900
- Total variable: $0.006927 ≈ $0.00693
- Trust: Computational privacy (server sees encrypted query)

Monthly (300K queries):
- Variable: 300,000 × $0.00693 = $2,079
- Fixed (r5.xlarge): $183
- Total: $2,262/month
```

#### 100K IT-PIR (3-Server)
```
Variable per query (3 servers total):
- Compute: 6.4s × 3 × $0.042/3600 = $0.000224
- Network: 0.000538GB × $0.09 = $0.000048
- Total variable: $0.000272 ≈ $0.00027
- Trust: Information-theoretic (requires 2+ honest servers)

Monthly (300K queries):
- Variable: 300,000 × $0.00027 = $81
- Fixed (3× t3.large): $183
- Total: $264/month
```

#### 1M IT-PIR (3-Server)
```
Variable per query (3 servers total):
- Compute: 8.1s × 3 × $0.096/3600 = $0.000648
- Network: 0.0054GB × $0.09 = $0.000486
- Total variable: $0.001134 ≈ $0.00113
- Trust: Information-theoretic (requires 2+ honest servers)

Monthly (300K queries):
- Variable: 300,000 × $0.00113 = $339
- Fixed (3× m5.xlarge): $415
- Total: $754/month
```

### PIR Performance Characteristics

```yaml
100K Database (Recommended for Clinical):
  CPIR (Single-Server):
    Query Time: 590ms
    Peak RAM: 1.2GB
    Network: 100KB/query
    Monthly Cost: $35
    Trust: Computational privacy
    
  IT-PIR (3-Server):
    Query Time: 6.4s
    Peak RAM: 3.6GB total
    Network: 538KB/query
    Monthly Cost: $264
    Trust: Information-theoretic (non-collusion)
  
  Use Case: Patient lookups, variant checks

1M Database (Research Scale):
  CPIR (Single-Server):
    Query Time: 920ms
    Peak RAM: 2.8GB
    Network: 1MB/query
    Monthly Cost: $91
    Trust: Computational privacy
    
  IT-PIR (3-Server):
    Query Time: 8.1s
    Peak RAM: 8.4GB total
    Network: 5.4MB/query
    Monthly Cost: $754
    Trust: Information-theoretic (non-collusion)
  
  Use Case: Cohort studies, biobank queries

10M Database (Population Scale):
  CPIR (Single-Server):
    Query Time: 113s
    Peak RAM: 14GB
    Network: 10MB/query
    Monthly Cost: $2,262
    Trust: Computational privacy
    Note: Consider sharding for better performance
  
  Use Case: Population genomics
```

### PIR Sharding Strategy (10M+ Records)

For databases exceeding 10M rows, sharding improves performance:

```yaml
Hash-Based Sharding (Recommended):
  Configuration: 10 shards × 1M records each
  Routing: Hash(query_key) % 10 → single shard
  Query Distribution: 10K queries/day ÷ 10 shards = 1K/shard/day
  
  Performance:
    Latency: 920ms (single shard query)
    Throughput: 10× parallel capacity
    
  Cost Breakdown:
    Per Shard: $91/month (1M CPIR at 1K queries/day)
    Total: 10 × $91 = $910/month
    Note: Significantly cheaper than 10M monolithic ($2,262/month)
    
Range-Based Sharding:
  Configuration: Partition by genomic region/chromosome
  Routing: Binary search on range boundaries
  Best For: Known access patterns (e.g., by chromosome)
  
Broadcast Query (NOT Recommended):
  Configuration: Query all shards in parallel
  Latency: 920ms (parallel) but 10× network cost
  Monthly Cost: 10 × $91 = $910 base + 10× network = $1,810/month
  Use Case: Only for exhaustive searches
```

## Zero-Knowledge Proof Costs

### Pricing Methodology
- AWS us-east-1 pricing (2024)
- vCPU-hour: $0.042 (t3.large), $0.084 (t3.xlarge), $0.384 (r5.2xlarge), $0.768 (r5.4xlarge)
- Network egress: $0.09/GB
- Storage: $0.10/GB/month

### Cost Breakdown

| Backend | Constraints | Prove Time³ | Instance⁵ | Pricing⁶ | Proof Size⁴ | Variable/Proof | Fixed/Month | Total/Month⁷ |
|---------|------------|-------------|-----------|----------|--------------|----------------|-------------|--------------|
| **Groth16** | 15K | 1.15s | c5.large | On-demand $0.085/hr | 192B | $0.000021 | $61 | **$67/mo** |
| **PLONK** | 15K | 0.82s | c5.xlarge | On-demand $0.170/hr | 1KB | $0.000031 | $122 | **$131/mo** |
| **Halo2** | 15K | 0.60s | c5.xlarge | On-demand $0.170/hr | 5KB | $0.000034 | $122 | **$132/mo** |
| **Groth16** | 1M | 18.3s | c5.4xlarge | On-demand $0.680/hr | 192B | $0.003910 | $489 | **$1,662/mo** |
| **PLONK** | 1M | 14.7s | c5.9xlarge | On-demand $1.530/hr | 1KB | $0.012554 | $1,101 | **$4,867/mo** |
| **Halo2** | 1M | 11.2s | c5.9xlarge | On-demand $1.530/hr | 5KB | $0.009558 | $1,101 | **$3,968/mo** |

⁵ c5 instances chosen over t3 for consistent performance (no credit management)
⁶ AWS us-east-1 on-demand pricing, spot would reduce by ~70%
⁷ At 10K proofs/day = 300K/month volume

**Evidence Sources:**
3. Prove Time: `benchmark_results/zk_proof_real_benchmark.json` → `prove_time_p50`
4. Proof Size: `benchmark_results/zk_proof_real_benchmark.json` → `proof_size_bytes`

### CPU Credit Analysis for t3 vs c5

```yaml
t3.large Burst Analysis (2 vCPUs):
  Baseline: 30% CPU (0.6 vCPUs continuous)
  Credits earned: 36 credits/hour
  Groth16-15K workload:
    - 10K proofs/day = 417/hour
    - CPU needed: 417 × 1.15s = 480 CPU-seconds/hour
    - Credits consumed: 480/3600 × 60 = 8 credits/hour
    - Net: +28 credits/hour (SUSTAINABLE with t3.large)
    
t3.xlarge Burst Analysis (4 vCPUs):
  Baseline: 40% CPU (1.6 vCPUs continuous)
  Credits earned: 96 credits/hour
  Halo2-15K workload:
    - 10K proofs/day = 417/hour
    - CPU needed: 417 × 0.60s = 250 CPU-seconds/hour
    - Credits consumed: 250/3600 × 60 = 4.2 credits/hour
    - Net: +91.8 credits/hour (SUSTAINABLE with t3.xlarge)

Conclusion: t3 instances ARE sustainable for 15K constraints at 10K/day
but c5 provides consistent performance without credit management
```

### Detailed Cost Calculation

#### Groth16-15K
```
Variable per proof:
- Compute: 1.15s × $0.085/3600 = $0.0000272  # c5.large pricing
- Network: 0.000000192GB × $0.09 = $0.0000000
- Total variable: $0.0000272 ≈ $0.000027

Monthly (300K proofs):
- Variable: 300,000 × $0.000027 = $8.10
- Fixed (c5.large): $61
- Total: $69/month

Note: Can use t3.large ($65/month) as CPU credits sustainable
```

#### Halo2-1M
```
Variable per proof:
- Compute: 11.2s × $1.53/3600 = $0.004760  # c5.9xlarge pricing (36 vCPUs)
- Network: 0.0000051GB × $0.09 = $0.0000005
- Total variable: $0.004761 ≈ $0.00476

Monthly (300K proofs):
- Variable: 300,000 × $0.00476 = $1,428
- Fixed (c5.9xlarge): $1,101
- Total: $2,529/month
```

### Setup Costs (One-Time)

| Backend | Setup Type | Cost | Time | Trust Model |
|---------|-----------|------|------|-------------|
| **Groth16** | Ceremony | $10-50K | 2-4 weeks | 1-of-N honest |
| **PLONK** | Download SRS | $0 | 1 hour | Use existing |
| **Halo2** | None | $0 | 0 | Trustless |

### ZK Backend Comparison

```yaml
Simple Proofs (15K constraints):
  Groth16:
    Monthly: $65
    Proof Size: 192B (best for blockchain)
    Trust: Requires ceremony
    
  PLONK:
    Monthly: $128
    Proof Size: 1KB
    Trust: Universal setup
    
  Halo2 (Recommended):
    Monthly: $128
    Proof Size: 5KB
    Trust: ZERO (trustless)

Complex Proofs (1M constraints):
  Groth16:
    Monthly: $1,539
    Peak RAM: 28GB
    Use Case: On-chain verification
    
  PLONK:
    Monthly: $4,498
    Peak RAM: 42GB
    Use Case: Flexible circuits
    
  Halo2:
    Monthly: $3,600
    Peak RAM: 48GB
    Use Case: Regulatory compliance
```

## Combined Stack Costs

### Recommended Production Configurations

#### Small Clinical Practice (1K patients)
```yaml
Components:
  PIR: 100K database, CPIR (computational)
  ZK: Halo2, 15K constraints (trustless)
  
Performance:
  PIR Latency: 590ms
  ZK Latency: 600ms
  Total: ~1.2s per query
  
Resources:
  Peak RAM: 5.4GB total
  Storage: 10GB
  Instances: t3.medium + c5.xlarge
  
Trust Model:
  PIR: Computational (CPIR, single server)
  ZK: Trustless (Halo2, no ceremony)
  
Monthly Cost (10K queries/day, on-demand):
  PIR: $35 (t3.medium sustainable)
  ZK: $132 (c5.xlarge)
  Total: $167/month
  
Per Query: $0.000056
Pricing: AWS us-east-1 on-demand
```

#### Research Institution (100K samples)
```yaml
Components:
  PIR: 1M database, IT-PIR (3-server)
  ZK: Halo2, 15K constraints (trustless)
  
Performance:
  PIR Latency: 8.1s
  ZK Latency: 600ms
  Total: ~8.7s per query
  
Resources:
  Peak RAM: 8.4GB (PIR) + 4.2GB (ZK)
  Storage: 100GB
  Instances: 3×m5.xlarge + c5.xlarge
  
Trust Model:
  PIR: IT-PIR (2+ honest servers, unconditional)
  ZK: Trustless (Halo2, no ceremony)
  
Monthly Cost (10K queries/day, on-demand):
  PIR: $754 (3×m5.xlarge @ $0.096/hr)
  ZK: $132 (c5.xlarge @ $0.170/hr)
  Total: $886/month
  
Per Query: $0.00295
Pricing: AWS us-east-1 on-demand
```

#### Healthcare Network (10M records)
```yaml
Components:
  PIR: 10M sharded (10×1M, hash routing), CPIR
  ZK: Halo2, 1M constraints (trustless)
  
Performance:
  PIR Latency: 920ms (single shard query)
  ZK Latency: 11.2s
  Total: ~12.1s per query
  
Resources:
  Peak RAM: 2.8GB × 10 (PIR shards) + 48GB (ZK)
  Storage: 1TB
  
Trust Model:
  PIR: Computational hardness (LWE/HE)
  ZK: Trustless (no setup ceremony)
  
Monthly Cost (10K queries/day):
  PIR: $910 (10 shards × $91 each)
  ZK: $2,529 (c5.9xlarge)
  Total: $3,439/month
  
Per Query: $0.01146

Note: Hash-based sharding reduces PIR cost by 60% vs monolithic
```

## Cost Optimization Strategies

### 1. Caching (40% reduction)
```yaml
Implementation:
  - Redis cluster for hypervectors
  - Proof caching for common queries
  - TTL: 24 hours
  
Cost Impact:
  - Cache hit rate: 40%
  - Effective cost: 60% of baseline
  - ROI: 3-4 months
```

### 2. Batch Processing (60% reduction)
```yaml
Implementation:
  - Queue queries for batch processing
  - Process during off-peak hours
  - Use spot instances
  
Cost Impact:
  - Spot discount: 70%
  - Batch efficiency: 2x
  - Effective cost: 35% of baseline
```

### 3. Tiered Architecture
```yaml
Tier 1 (Hot - 10%):
  - In-memory PIR
  - Pre-computed proofs
  - Cost: $500/month
  
Tier 2 (Warm - 30%):
  - SSD-backed PIR
  - On-demand proofs
  - Cost: $200/month
  
Tier 3 (Cold - 60%):
  - S3-backed PIR
  - Batch proofs
  - Cost: $50/month
  
Total: $750/month (vs $2,243 flat)
```

## Break-Even Analysis

### PIR: CPIR vs IT-PIR

**Formula**: For break-even queries per day Q*:
```
F₁ + 30·Q·v₁ = F₂ + 30·Q·v₂
Q* = (F₂ - F₁) / (30·(v₁ - v₂))
```
Where F = fixed monthly cost, v = variable cost per query

#### 100K Database
```
CPIR: F₁ = $30, v₁ = $0.000016
IT-PIR: F₂ = $183, v₂ = $0.00027

Q* = (183 - 30) / (30 × (0.000016 - 0.00027))
Q* = 153 / (30 × -0.000254)
Q* = 153 / -0.00762
Q* = -20,079 queries/day

Since Q* is negative, IT-PIR is NEVER cheaper for 100K database
(CPIR always wins due to much lower fixed costs)
```

#### 1M Database
```
CPIR: F₁ = $61, v₁ = $0.00010
IT-PIR: F₂ = $415, v₂ = $0.00113

Q* = (415 - 61) / (30 × (0.00010 - 0.00113))
Q* = 354 / (30 × -0.00103)
Q* = 354 / -0.0309
Q* = -11,456 queries/day

Since Q* is negative, IT-PIR is NEVER cheaper for 1M database
(CPIR always wins for this configuration)
```

**Conclusion**: CPIR is always more cost-effective than IT-PIR at these scales, but IT-PIR provides unconditional privacy (no computational assumptions)

### ZK: Groth16 vs Halo2

#### 15K Constraints
```
Groth16: F₁ = $61, v₁ = $0.000013
Halo2: F₂ = $122, v₂ = $0.000021

Q* = (122 - 61) / (30 × (0.000013 - 0.000021))
Q* = 61 / (30 × -0.000008)
Q* = 61 / -0.00024
Q* = -254,167 queries/day

Since Q* is negative, Halo2 is NEVER cheaper for 15K constraints
(But Halo2 avoids $10-50K trusted setup ceremony)
```

#### 1M Constraints
```
Groth16: F₁ = $366, v₁ = $0.003910
Halo2: F₂ = $732, v₂ = $0.009558

Q* = (732 - 366) / (30 × (0.003910 - 0.009558))
Q* = 366 / (30 × -0.005648)
Q* = 366 / -0.16944
Q* = -2,160 queries/day

Since Q* is negative, Halo2 is NEVER cheaper for 1M constraints
(Groth16 wins on pure operational cost, but requires trusted setup)
```

**Conclusion**: Groth16 is operationally cheaper, but Halo2 eliminates trusted setup requirements worth $10-50K

## Cloud Provider Comparison

| Provider | PIR (1M DB) | ZK (Halo2-15K) | Combined | Notes |
|----------|-------------|----------------|----------|-------|
| **AWS** | $151/mo | $123/mo | $274/mo | Best spot pricing |
| **GCP** | $163/mo | $135/mo | $298/mo | Better networking |
| **Azure** | $158/mo | $129/mo | $287/mo | Hybrid friendly |
| **On-Prem** | $500 amortized | $400 amortized | $900/mo | 3-year TCO |

## Pricing Calculator

```python
def calculate_breakeven(f1: float, v1: float, f2: float, v2: float) -> float:
    """Calculate break-even point between two configurations.
    
    Returns queries/day where costs are equal, or negative if config 2 never wins.
    """
    if v1 == v2:
        return float('inf') if f1 < f2 else 0
    return (f2 - f1) / (30 * (v1 - v2))

def calculate_monthly_cost(
    queries_per_day: int,
    database_rows: int,
    zk_constraints: int,
    backend: str = "halo2"
) -> dict:
    
    # Variable costs per query
    pir_variable = {
        100_000: 0.000016,
        1_000_000: 0.00010,
        10_000_000: 0.00693
    }
    
    # Fixed monthly costs
    pir_fixed = {
        100_000: 30,
        1_000_000: 61,
        10_000_000: 183
    }
    
    # ZK variable costs per proof
    zk_variable = {
        ("groth16", 15_000): 0.000013,
        ("plonk", 15_000): 0.000019,
        ("halo2", 15_000): 0.000021,
        ("groth16", 1_000_000): 0.003910,
        ("plonk", 1_000_000): 0.012554,
        ("halo2", 1_000_000): 0.009558
    }
    
    # ZK fixed monthly costs
    zk_fixed = {
        ("groth16", 15_000): 61,
        ("plonk", 15_000): 122,
        ("halo2", 15_000): 122,
        ("groth16", 1_000_000): 366,
        ("plonk", 1_000_000): 732,
        ("halo2", 1_000_000): 732
    }
    
    # Calculate totals
    queries_per_month = queries_per_day * 30
    
    pir_var_month = pir_variable.get(database_rows, 0.00010) * queries_per_month
    pir_fix_month = pir_fixed.get(database_rows, 61)
    
    zk_var_month = zk_variable.get((backend, zk_constraints), 0.000021) * queries_per_month
    zk_fix_month = zk_fixed.get((backend, zk_constraints), 122)
    
    total_monthly = pir_var_month + pir_fix_month + zk_var_month + zk_fix_month
    cost_per_query = (pir_var_month + zk_var_month) / queries_per_month
    
    return {
        "pir_variable_monthly": pir_var_month,
        "pir_fixed_monthly": pir_fix_month,
        "zk_variable_monthly": zk_var_month,
        "zk_fixed_monthly": zk_fix_month,
        "total_monthly": total_monthly,
        "cost_per_query_variable": cost_per_query
    }

# Example: Research institution
cost = calculate_monthly_cost(
    queries_per_day=10_000,
    database_rows=1_000_000,
    zk_constraints=15_000,
    backend="halo2"
)
print(f"Monthly cost: ${cost['total_monthly']:.2f}")
print(f"Variable per query: ${cost['cost_per_query_variable']:.6f}")

# Break-even analysis
print("\n=== Break-Even Analysis ===")

# PIR: CPIR vs IT-PIR for 1M database
cpir_fixed, cpir_var = 61, 0.00010
itpir_fixed, itpir_var = 415, 0.00113
breakeven = calculate_breakeven(cpir_fixed, cpir_var, itpir_fixed, itpir_var)
if breakeven < 0:
    print(f"PIR (1M DB): CPIR always cheaper than IT-PIR")
else:
    print(f"PIR (1M DB): Break-even at {breakeven:.0f} queries/day")

# ZK: Groth16 vs Halo2 for 15K constraints
groth_fixed, groth_var = 61, 0.000013
halo_fixed, halo_var = 122, 0.000021
breakeven = calculate_breakeven(groth_fixed, groth_var, halo_fixed, halo_var)
if breakeven < 0:
    print(f"ZK (15K): Halo2 never cheaper (but avoids trusted setup)")
else:
    print(f"ZK (15K): Break-even at {breakeven:.0f} queries/day")
```

## Key Takeaways

1. **PIR costs grow rapidly** with database size ($35 → $2,262/month for 100K → 10M rows)
2. **ZK proof costs scale with complexity** ($128 → $3,600/month for 15K → 1M constraints)
3. **Fixed costs dominate at low volume**, variable costs dominate at high volume
4. **Halo2 recommended** for trustless setup despite higher costs for complex proofs
5. **Batch processing and caching** can reduce costs by 40-60%
6. **Small practices** can operate for <$200/month, enterprises need $5K+/month

---

*Prices based on AWS us-east-1 as of 2024. Assumes 10K queries/day (300K/month). Add 20% for multi-region deployment.*