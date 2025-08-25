# GenomeVault Cost Analysis

## Executive Summary

**Bottom Line Costs (10K queries/day = 300K/month)**:
- **PIR**: $35-2,262/month depending on database size
- **ZK Proofs**: $65-3,600/month depending on backend and complexity
- **Combined**: $163-5,862/month for full privacy stack

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

| Database Size | Scheme | Trust Model | Latency | vCPU-sec | Network | Variable/Query | Fixed/Month | Total/Month (10K/day) |
|--------------|--------|-------------|---------|----------|---------|----------------|-------------|----------------------|
| **100K rows** | CPIR | Computational | 0.59s | 0.59s | 100KB | $0.000016 | $30 | **$35/mo** |
| **1M rows** | CPIR | Computational | 0.92s | 0.92s | 1MB | $0.00010 | $61 | **$91/mo** |
| **10M rows** | CPIR | Computational | 113s | 113s | 10MB | $0.00693 | $183 | **$2,262/mo** |
| **100K rows** | IT-PIR (3-server) | Information-theoretic* | 6.4s | 19.2s | 538KB | $0.00027 | $183 | **$264/mo** |
| **1M rows** | IT-PIR (3-server) | Information-theoretic* | 8.1s | 24.3s | 5.4MB | $0.00113 | $415 | **$754/mo** |

*Assumes non-collusion among servers

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

## Zero-Knowledge Proof Costs

### Pricing Methodology
- AWS us-east-1 pricing (2024)
- vCPU-hour: $0.042 (t3.large), $0.084 (t3.xlarge), $0.384 (r5.2xlarge), $0.768 (r5.4xlarge)
- Network egress: $0.09/GB
- Storage: $0.10/GB/month

### Cost Breakdown

| Backend | Constraints | Prove Time | vCPU-sec | Proof Size | Variable/Proof | Fixed/Month | Total/Month (10K/day) |
|---------|------------|------------|----------|------------|----------------|-------------|----------------------|
| **Groth16** | 15K | 1.15s | 1.15s | 192B | $0.000013 | $61 | **$65/mo** |
| **PLONK** | 15K | 0.82s | 0.82s | 1KB | $0.000019 | $122 | **$128/mo** |
| **Halo2** | 15K | 0.60s | 0.60s | 5KB | $0.000021 | $122 | **$128/mo** |
| **Groth16** | 1M | 18.3s | 36.6s* | 192B | $0.003910 | $366 | **$1,539/mo** |
| **PLONK** | 1M | 14.7s | 58.8s** | 1KB | $0.012554 | $732 | **$4,498/mo** |
| **Halo2** | 1M | 11.2s | 44.8s** | 5KB | $0.009558 | $732 | **$3,600/mo** |

*2 vCPUs, **4 vCPUs for complex proofs

### Detailed Cost Calculation

#### Groth16-15K
```
Variable per proof:
- Compute: 1.15s × $0.042/3600 = $0.0000134
- Network: 0.000000192GB × $0.09 = $0.0000000
- Total variable: $0.0000134 ≈ $0.000013

Monthly (300K proofs):
- Variable: 300,000 × $0.000013 = $3.90
- Fixed (t3.large): $61
- Total: $65/month
```

#### Halo2-1M
```
Variable per proof:
- Compute: 11.2s × 4 × $0.768/3600 = $0.009557
- Network: 0.0000051GB × $0.09 = $0.0000005
- Total variable: $0.009558 ≈ $0.00956

Monthly (300K proofs):
- Variable: 300,000 × $0.00956 = $2,868
- Fixed (r5.4xlarge): $732
- Total: $3,600/month
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
  PIR: 100K database, CPIR (computational privacy)
  ZK: Halo2, 15K constraints (trustless)
  
Performance:
  PIR Latency: 590ms
  ZK Latency: 600ms
  Total: ~1.2s per query
  
Resources:
  Peak RAM: 5.4GB total
  Storage: 10GB
  
Trust Model:
  PIR: Computational hardness (LWE/HE)
  ZK: Trustless (no setup ceremony)
  
Monthly Cost (10K queries/day):
  PIR: $35
  ZK: $128
  Total: $163/month
  
Per Query: $0.000054
```

#### Research Institution (100K samples)
```yaml
Components:
  PIR: 1M database, IT-PIR (3-server, information-theoretic)
  ZK: Halo2, 15K constraints (trustless)
  
Performance:
  PIR Latency: 8.1s
  ZK Latency: 600ms
  Total: ~8.7s per query
  
Resources:
  Peak RAM: 8.4GB (PIR) + 4.2GB (ZK)
  Storage: 100GB
  
Trust Model:
  PIR: Information-theoretic (2+ honest servers)
  ZK: Trustless (no setup ceremony)
  
Monthly Cost (10K queries/day):
  PIR: $754
  ZK: $128
  Total: $882/month
  
Per Query: $0.00294
```

#### Healthcare Network (10M records)
```yaml
Components:
  PIR: 10M database, CPIR (computational privacy)
  ZK: Halo2, 1M constraints (trustless)
  
Performance:
  PIR Latency: 113s
  ZK Latency: 11.2s
  Total: ~124s per query
  
Resources:
  Peak RAM: 14GB (PIR) + 48GB (ZK)
  Storage: 1TB
  
Trust Model:
  PIR: Computational hardness (LWE/HE)
  ZK: Trustless (no setup ceremony)
  
Monthly Cost (10K queries/day):
  PIR: $2,262
  ZK: $3,600
  Total: $5,862/month
  
Per Query: $0.01954
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