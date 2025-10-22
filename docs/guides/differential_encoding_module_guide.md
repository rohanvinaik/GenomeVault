# Differential Encoding Module

## Overview

This module implements cryptographically secure differential encoding of genomic data. The system stores experimental genomes as cryptographically verified differences from randomly selected reference genome sections.

## Architecture

```
differential_encoding/
├── crypto_primitives.py      # Cryptographic RNG and hashing
├── reference_management.py   # Reference genome pool management
├── chunking.py               # (Coming: Cryptographic chunking)
├── differential_encoder.py   # (Coming: Differential encoding)
└── hypervector_binding.py    # (Coming: HV integration)
```

## Implemented Components

### ✅ Section 2: Cryptographic Primitives (Complete)

#### 1. CryptoRNG Class
**File**: `crypto_primitives.py`

Cryptographically secure random number generator with:
- HKDF-based deterministic seed derivation (HMAC-SHA256)
- Context-aware derivation with counter
- Deterministic random integer generation
- Cryptographic random selection from lists
- Full reproducibility guarantees

**Key Methods**:
- `derive_seed(context: bytes) -> bytes` - HKDF derivation
- `random_int(low: int, high: int, seed: bytes) -> int` - Deterministic random int
- `random_choice(items: List[T], seed: bytes) -> T` - Cryptographic selection

**Security Properties**:
- Pseudorandomness (indistinguishable from uniform random)
- Unpredictability (cannot predict future seeds)
- Independence (different contexts → independent seeds)
- Determinism (same inputs → same outputs)

#### 2. Chunk Identifier Generation
**Function**: `compute_chunk_id(chunk, master_seed) -> bytes`

Generates collision-resistant identifiers for genomic chunks using:
- SHA-256 hashing
- Genomic coordinates (chromosome, start, end)
- Variant content hash
- Cryptographic binding to master seed

**Properties**:
- Deterministic (same chunk → same ID)
- Collision-resistant (different chunks → different IDs)
- Unpredictable (cannot guess IDs without master seed)
- Integrity verification (detects modifications)

#### 3. Reference Genome Hashing
**Function**: `compute_reference_hash(reference) -> str`

Computes SHA-256 hash of reference genomes for:
- Integrity verification
- Version tracking
- Tamper detection

**Features**:
- Deterministic ordering (sorted chromosomes and variants)
- Includes assembly information
- Includes variant genotypes
- 64-character hex digest (256-bit security)

#### 4. Chunk-Reference Binding
**Function**: `compute_chunk_reference_binding(chunk_id, reference_id) -> bytes`

Creates HMAC-based cryptographic binding ensuring:
- Cannot swap references without detection
- Cannot forge bindings without chunk_id
- Cryptographic proof of chunk-reference association

**Security**:
- HMAC-SHA256 based
- Unforgeability
- Collision resistance
- Verification support

## Test Coverage

**File**: `tests/differential_encoding/test_crypto_primitives.py`

### Test Suites (40 tests total):

1. **TestCryptoRNG** (16 tests)
   - Initialization (default/custom/invalid seeds)
   - Deterministic seed derivation
   - Counter increment and management
   - HMAC structure verification
   - Random integer generation (deterministic, range, edge cases)
   - Random choice (deterministic, distribution, edge cases)
   - Cross-instance reproducibility

2. **TestComputeChunkID** (6 tests)
   - Deterministic ID generation
   - Different chromosomes/positions/variants
   - Variant order independence
   - Master seed sensitivity

3. **TestComputeReferenceHash** (6 tests)
   - Deterministic hashing
   - Different assemblies/variants/genotypes
   - Chromosome and variant order independence

4. **TestComputeChunkReferenceBinding** (5 tests)
   - Deterministic binding
   - Different chunk IDs and reference IDs
   - HMAC structure verification
   - Complete verification workflow

5. **TestSecurityProperties** (3 tests)
   - Collision resistance for chunk IDs
   - Integrity detection for reference hashes
   - Seed derivation independence

6. **TestEdgeCases** (4 tests)
   - Empty variant lists
   - Empty references
   - Large variant lists
   - Special chromosome names

### Test Results

```
======================== 40 passed in 0.15s ========================
```

All tests pass successfully with:
- ✅ Deterministic behavior verified
- ✅ Security properties confirmed
- ✅ Edge cases handled
- ✅ HMAC/SHA-256 implementations correct

## Usage Examples

### Basic CryptoRNG Usage

```python
from genomevault.differential_encoding import CryptoRNG

# Initialize with random seed
rng = CryptoRNG()

# Or with custom seed for reproducibility
rng = CryptoRNG(master_seed=b"\x00" * 32)

# Derive deterministic seeds
seed1 = rng.derive_seed(b"chunk_1")
seed2 = rng.derive_seed(b"reference_selection")

# Generate random integers
value = rng.random_int(0, 100, seed1)  # Deterministic

# Select random items
references = ["GRCh38", "GRCh37", "CHM13"]
selected = rng.random_choice(references, seed2)
```

### Chunk ID Generation

```python
from genomevault.differential_encoding import compute_chunk_id
from dataclasses import dataclass
from typing import List

@dataclass
class Variant:
    position: int
    ref: str
    alt: str

@dataclass
class GenomeChunk:
    chromosome: str
    start_position: int
    end_position: int
    variants: List[Variant]

# Create chunk
chunk = GenomeChunk(
    chromosome="chr1",
    start_position=100000,
    end_position=200000,
    variants=[
        Variant(position=150000, ref="A", alt="G")
    ]
)

# Generate chunk ID
master_seed = rng.derive_seed(b"master_context")
chunk_id = compute_chunk_id(chunk, master_seed)
# Returns: 32-byte identifier
```

### Reference Genome Hashing

```python
from genomevault.differential_encoding import compute_reference_hash

@dataclass
class ReferenceGenome:
    assembly: str
    variants: Dict[str, List[Variant]]

reference = ReferenceGenome(
    assembly="GRCh38",
    variants={
        "chr1": [Variant(position=100, ref="A", alt="G", genotype="0/1")],
        "chr2": [Variant(position=200, ref="C", alt="T", genotype="0/1")]
    }
)

# Compute hash
ref_hash = compute_reference_hash(reference)
# Returns: "a3f5c89..." (64-character hex string)

# Later: verify integrity
new_hash = compute_reference_hash(reference)
assert new_hash == ref_hash  # Verify no tampering
```

### Chunk-Reference Binding

```python
from genomevault.differential_encoding import compute_chunk_reference_binding

# Create binding
chunk_id = compute_chunk_id(chunk, master_seed)
reference_id = "GRCh38"
binding = compute_chunk_reference_binding(chunk_id, reference_id)

# Store binding with encoded chunk
# ...

# Later: verify chunk-reference association
claimed_reference = "GRCh38"
computed_binding = compute_chunk_reference_binding(chunk_id, claimed_reference)
assert computed_binding == binding  # Verification passes

# Attack detection
wrong_reference = "GRCh37"
forged_binding = compute_chunk_reference_binding(chunk_id, wrong_reference)
assert forged_binding != binding  # Attack detected!
```

## Mathematical Foundation

### HKDF Seed Derivation

```
derived_seed = HMAC-SHA256(master_seed, context || counter)

where:
  HMAC(K, M) = H((K ⊕ opad) || H((K ⊕ ipad) || M))
  H = SHA-256
  K = master_seed (key)
  M = context || counter (message)
```

### Chunk ID Computation

```
chunk_id = SHA-256(master_seed || chr || start || end || variant_hash)

where:
  variant_hash = SHA-256(sorted_variants)
  sorted_variants = sort by position
```

### Reference Hash Computation

```
reference_hash = SHA-256(assembly || ⊕[chr_hashes])

where:
  chr_hash_i = SHA-256(chr_name || sorted_variants_i)
```

### Chunk-Reference Binding

```
binding = HMAC-SHA256(chunk_id, reference_id)
```

## Security Guarantees

### Cryptographic Properties

1. **Pseudorandomness**: All derived seeds indistinguishable from uniform random
2. **Unpredictability**: Cannot predict future outputs from past outputs
3. **Collision Resistance**: SHA-256 provides 2^128 collision resistance
4. **Preimage Resistance**: SHA-256 provides 2^256 preimage resistance
5. **Binding Security**: HMAC prevents forgery and reference swapping

### Information-Theoretic Bounds

- **Seed Space**: 2^256 possible seeds (32 bytes)
- **Chunk ID Space**: 2^256 possible IDs
- **Collision Probability**: < 2^-128 for different chunks
- **Brute Force Resistance**: 2^256 operations to invert hash

## Next Steps

This README covers the completed sections of the Differential Encoding pipeline.

### Implementation Status:

1. ✅ **Section 2**: Cryptographic Primitives (COMPLETE)
   - CryptoRNG with HKDF-based derivation
   - Chunk ID generation
   - Reference genome hashing
   - Chunk-reference binding

2. ✅ **Section 3**: Reference Genome Management (COMPLETE)
   - VCF parsing and loading
   - Position indexing with IntervalTree
   - Secure random reference selection
   - Reference verification

3. ✅ **Section 4**: Cryptographic Chunking (COMPLETE)
   - 7 analysis-type-specific strategies
   - Sliding window chunking
   - Feature-based chunking
   - Deterministic randomization

4. ⏳ **Section 5**: Differential Encoding (UPCOMING)
   - Variant difference computation
   - Differential encoding pipeline

5. ⏳ **Section 6**: Hypervector Encoding with Binding (UPCOMING)
   - Feature vector construction
   - Hypervector projection
   - Cryptographic binding integration

See detailed documentation in `docs/DIFFERENTIAL_ENCODING_SECTION*_COMPLETE.md` files.

## Dependencies

- Python 3.11+
- `hashlib` (standard library)
- `hmac` (standard library)
- `secrets` (standard library)
- `random` (standard library)

## References

- NIST SP 800-108: Recommendation for Key Derivation Functions
- RFC 2104: HMAC - Keyed-Hashing for Message Authentication
- FIPS 180-4: Secure Hash Standard (SHA-256)
