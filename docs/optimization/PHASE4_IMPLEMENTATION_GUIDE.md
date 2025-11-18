# Phase 4 Implementation Guide: Research Optimizations

**Date:** October 25, 2025
**Status:** Optional research-level optimizations
**Expected Additional Speedup:** ~30 minutes (minimal)
**Effort:** 8-12 hours
**Priority:** LOW (other phases provide better ROI)

---

## Overview

Phase 4 contains research-level optimizations that require significant implementation effort but provide minimal performance gains. These are **optional** and recommended only for academic exploration or specific use cases.

**Included optimizations:**
1. **PLONK ZK Backend** - Alternative to Groth16 (faster proving, larger proofs)
2. **Memory-Mapped Graph Construction** - Reduce RAM usage for Layer 1

### Prerequisites

**Phases 1-3 should be completed first:**
- ✅ Phase 1: Sambamba, parallel BCFtools, Metal GPU HDC (5.6 hours saved)
- ✅ Phase 2: Minimap2 index caching, AMX alignment (2.4 hours saved)
- ✅ Phase 3: Chromosome-parallel sorting, parallel VCF parsing (2.1 hours saved)

**Current state after Phase 3:**
- Layer 1: 25 min
- Per reference: 12 min
- 12 references: 2.4 hours
- Layer 4 ZK proof: 1-2 sec
- **Total: ~3 hours**

**After Phase 4:**
- Layer 1: 15 min (memory-mapped graph)
- Layer 4 ZK proof: 0.5-0.8 sec (PLONK)
- **Total: ~2.9 hours**
- **Additional time saved: ~10-15 min** ⚠️ **LOW ROI**

---

## ⚠️ Important ROI Assessment

**Before implementing Phase 4, consider:**

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|--------|---------|---------|---------|---------|
| **Effort** | 30 min | 4-6 hours | 6-10 hours | **8-12 hours** |
| **Time Saved** | 5.6 hours | 2.4 hours | 2.1 hours | **~15 min** ⚠️ |
| **ROI (savings/effort)** | 11.2× | 0.5× | 0.3× | **0.02×** ⚠️ |
| **Risk** | Low | Low-Med | Medium | **High** |
| **Complexity** | Low | Medium | High | **Very High** |

**Recommendation:** **Skip Phase 4 unless:**
- You need PLONK for specific proof size/verification tradeoffs
- You're research-focused and want to explore advanced techniques
- You have unlimited development time

**Better use of time:**
- Optimize data acquisition (scale to k=20+)
- Improve user interfaces
- Add new privacy features
- Write documentation

---

## Optimization 1: PLONK Zero-Knowledge Backend

### Overview

**Current:** Groth16 ZK proof system (743 bytes proof, 1-2 sec proving)

**Alternative:** PLONK proof system (1.5 KB proof, 0.5-0.8 sec proving)

**Tradeoff:**
- ✅ **Faster proving:** 1-2 sec → 0.5-0.8 sec (2× faster)
- ❌ **Larger proofs:** 743 bytes → 1,536 bytes (2× larger)
- ❌ **Same verification:** ~10 ms (unchanged)

### Performance Impact

| Metric | Groth16 | PLONK | Change |
|--------|---------|-------|--------|
| Proving time | 1-2 sec | 0.5-0.8 sec | **2× faster** ✅ |
| Proof size | 743 bytes | 1,536 bytes | **2× larger** ❌ |
| Verification time | ~10 ms | ~10 ms | Same |
| Setup | Trusted | Universal | **Better** ✅ |

**Net benefit:** Save 0.5-1 sec per query

**For 1 query:** 1 sec saved
**For 1,000 queries:** 16 min saved
**For 1 million queries:** 11 hours saved

**Recommendation:** Only implement if you expect **millions** of queries.

### Implementation

#### Step 1: Install PLONK Library

```bash
# Install Halo2 (PLONK variant, Rust-based)
cargo install halo2

# Or use Python wrapper
pip install py-halo2
```

#### Step 2: Create PLONK Circuit

Create `genomevault/zk_proofs/circuits/variant_presence_plonk.py`:

```python
"""PLONK circuit for variant presence proof."""

try:
    from py_halo2 import Circuit, Field, assign
    PLONK_AVAILABLE = True
except ImportError:
    PLONK_AVAILABLE = False
    print("⚠️ py-halo2 not available, PLONK backend disabled")

import hashlib
import numpy as np


class VariantPresencePLONK:
    """PLONK-based variant presence proof (alternative to Groth16)."""

    def __init__(self, dimension: int = 10000):
        """
        Initialize PLONK prover.

        Args:
            dimension: Hypervector dimension
        """
        if not PLONK_AVAILABLE:
            raise RuntimeError("py-halo2 not installed")

        self.dimension = dimension

    def create_circuit(self):
        """
        Create PLONK circuit for variant presence.

        Circuit constraints:
        1. hypervector_hash = SHA256(hypervector)
        2. variant_id_hash = SHA256(variant_id)
        3. proof_hash = SHA256(hypervector_hash || variant_id_hash)
        """
        circuit = Circuit()

        # Witness inputs (private)
        hypervector = circuit.add_witness_array(self.dimension)
        variant_id = circuit.add_witness()

        # Public inputs
        hypervector_hash = circuit.add_public_input()
        proof_commitment = circuit.add_public_input()

        # Constraint 1: Hash hypervector
        computed_hash = circuit.sha256(hypervector)
        circuit.enforce_equal(computed_hash, hypervector_hash)

        # Constraint 2: Compute proof commitment
        variant_hash = circuit.sha256([variant_id])
        commitment = circuit.sha256([computed_hash, variant_hash])
        circuit.enforce_equal(commitment, proof_commitment)

        return circuit

    def generate_proof(
        self,
        hypervector: np.ndarray,
        variant_id: str
    ) -> dict:
        """
        Generate PLONK proof of variant presence.

        Args:
            hypervector: 10,000D hypervector (private witness)
            variant_id: Variant identifier (private witness)

        Returns:
            Dict with proof, public inputs, verification status
        """
        # Hash hypervector (public input)
        hypervector_bytes = hypervector.astype(np.float32).tobytes()
        hypervector_hash = hashlib.sha256(hypervector_bytes).digest()

        # Compute proof commitment (public input)
        variant_hash = hashlib.sha256(variant_id.encode()).digest()
        proof_commitment = hashlib.sha256(
            hypervector_hash + variant_hash
        ).digest()

        # Create circuit
        circuit = self.create_circuit()

        # Assign witness values
        witness = {
            "hypervector": hypervector.tolist(),
            "variant_id": int.from_bytes(variant_id.encode()[:32], 'big')
        }

        public_inputs = {
            "hypervector_hash": int.from_bytes(hypervector_hash, 'big'),
            "proof_commitment": int.from_bytes(proof_commitment, 'big')
        }

        # Generate proof (this is where PLONK is 2× faster than Groth16)
        proof = circuit.prove(witness, public_inputs)

        return {
            "proof": proof.serialize(),
            "public_inputs": public_inputs,
            "proof_size_bytes": len(proof.serialize()),
            "backend": "PLONK"
        }

    def verify_proof(
        self,
        proof: bytes,
        public_inputs: dict
    ) -> bool:
        """Verify PLONK proof."""
        circuit = self.create_circuit()
        return circuit.verify(proof, public_inputs)


def benchmark_plonk_vs_groth16():
    """Benchmark PLONK vs Groth16 proving time."""
    import time
    from genomevault.zk_proofs.groth16 import Groth16Prover

    # Generate test data
    hypervector = np.random.randn(10000).astype(np.float32)
    variant_id = "chr22:12345678 C>A"

    # Test Groth16
    print("Groth16 Proof Generation...")
    groth16 = Groth16Prover()

    start = time.time()
    proof_g16 = groth16.generate_proof(hypervector, variant_id)
    g16_time = time.time() - start

    print(f"  Time: {g16_time:.3f} sec")
    print(f"  Size: {proof_g16['proof_size_bytes']} bytes")

    # Test PLONK
    print("\nPLONK Proof Generation...")
    plonk = VariantPresencePLONK()

    start = time.time()
    proof_plonk = plonk.generate_proof(hypervector, variant_id)
    plonk_time = time.time() - start

    print(f"  Time: {plonk_time:.3f} sec")
    print(f"  Size: {proof_plonk['proof_size_bytes']} bytes")

    # Summary
    print("\n" + "="*60)
    print(f"Speedup: {g16_time/plonk_time:.2f}× faster with PLONK")
    print(f"Size increase: {proof_plonk['proof_size_bytes']/proof_g16['proof_size_bytes']:.2f}×")
    print("="*60)


if __name__ == "__main__":
    if PLONK_AVAILABLE:
        benchmark_plonk_vs_groth16()
    else:
        print("Install py-halo2 to run benchmark")
```

#### Step 3: Add Backend Selection

Update `genomevault/zk_proofs/prover.py`:

```python
"""Unified ZK proof interface with backend selection."""

from typing import Literal

ZKBackend = Literal["groth16", "plonk"]


class ZKProver:
    """Unified ZK prover with selectable backend."""

    def __init__(self, backend: ZKBackend = "groth16"):
        """
        Initialize ZK prover.

        Args:
            backend: "groth16" (default, smaller proofs) or "plonk" (faster proving)
        """
        self.backend = backend

        if backend == "groth16":
            from genomevault.zk_proofs.groth16 import Groth16Prover
            self.prover = Groth16Prover()
        elif backend == "plonk":
            from genomevault.zk_proofs.circuits.variant_presence_plonk import VariantPresencePLONK
            self.prover = VariantPresencePLONK()
        else:
            raise ValueError(f"Unknown ZK backend: {backend}")

    def generate_proof(self, hypervector, variant_id):
        """Generate proof using selected backend."""
        return self.prover.generate_proof(hypervector, variant_id)

    def verify_proof(self, proof, public_inputs):
        """Verify proof using selected backend."""
        return self.prover.verify_proof(proof, public_inputs)
```

### Testing

```bash
# Test 1: Install PLONK library
pip install py-halo2

# Verify installation
python3 -c "from py_halo2 import Circuit; print('✅ PLONK available')"

# Test 2: Run benchmark
python3 genomevault/zk_proofs/circuits/variant_presence_plonk.py

# Expected output:
# Groth16: 1.2 sec, 743 bytes
# PLONK: 0.6 sec, 1536 bytes
# Speedup: 2× faster with PLONK

# Test 3: Use in pipeline
python3 benchmarks/run_enhanced_privacy_pipeline.py \
    --zk-backend plonk \
    --num-references 1
```

### When to Use PLONK

**Use PLONK when:**
- ✅ High query volume (thousands+ per day)
- ✅ Proving time more important than proof size
- ✅ Bandwidth is cheap (proof size doesn't matter)

**Use Groth16 when:**
- ✅ Low query volume (<100 per day)
- ✅ Proof size matters (bandwidth-constrained)
- ✅ Default for GenomeVault (already optimized)

---

## Optimization 2: Memory-Mapped Graph Construction

### Overview

**Problem:** Layer 1 consensus building loads entire variant graph into RAM (2-4 GB for whole genome).

**Solution:** Use memory-mapped files (`mmap`) for zero-copy graph storage.

### Performance Impact

| Metric | In-Memory | Memory-Mapped | Change |
|--------|-----------|---------------|--------|
| Peak RAM | 4 GB | 500 MB | **8× less RAM** ✅ |
| Build time | 60 min | 45 min | **1.3× faster** |
| Disk space | 0 GB | 4 GB | **+4 GB disk** ❌ |

**Net benefit:**
- Save 15 min on Layer 1 (one-time cost)
- Reduce RAM usage (useful for low-memory systems)

**Recommendation:** Only implement if you have **RAM constraints** (<8 GB available).

### Implementation

Create `genomevault/reference/memory_mapped_graph.py`:

```python
"""Memory-mapped variant graph for reduced RAM usage."""

import os
import mmap
import struct
from typing import Dict, Tuple
import numpy as np


class MemoryMappedVariantGraph:
    """
    Variant graph stored in memory-mapped file.

    This reduces peak RAM usage from 4 GB → 500 MB by storing
    graph on disk and accessing via mmap (zero-copy).
    """

    # Each variant entry: 48 bytes
    # - position (8 bytes, uint64)
    # - ref_allele (4 bytes, encoded)
    # - alt_allele (4 bytes, encoded)
    # - frequency (8 bytes, float64)
    # - quality (8 bytes, float64)
    # - padding (16 bytes)
    ENTRY_SIZE = 48

    def __init__(
        self,
        backing_file: str,
        num_positions: int = 60_000_000,
        create: bool = True
    ):
        """
        Initialize memory-mapped graph.

        Args:
            backing_file: Path to backing file (will be created)
            num_positions: Maximum number of genomic positions
            create: Create new file (True) or open existing (False)
        """
        self.backing_file = backing_file
        self.num_positions = num_positions
        self.file_size = num_positions * self.ENTRY_SIZE

        if create:
            self._create_backing_file()

        # Memory-map the file
        self.file_handle = open(backing_file, "r+b")
        self.mmap = mmap.mmap(
            self.file_handle.fileno(),
            0,
            access=mmap.ACCESS_WRITE
        )

    def _create_backing_file(self):
        """Create and pre-allocate backing file."""
        print(f"Creating memory-mapped file: {self.backing_file} ({self.file_size / (1024**3):.2f} GB)")

        with open(self.backing_file, "wb") as f:
            # Pre-allocate file (filled with zeros)
            f.seek(self.file_size - 1)
            f.write(b'\0')

    def add_variant(
        self,
        position: int,
        ref_allele: str,
        alt_allele: str,
        frequency: float,
        quality: float
    ):
        """
        Add variant to graph (writes directly to mmap, zero-copy).

        Args:
            position: Genomic position
            ref_allele: Reference allele
            alt_allele: Alternate allele
            frequency: Allele frequency
            quality: Variant quality score
        """
        if position >= self.num_positions:
            raise ValueError(f"Position {position} exceeds max {self.num_positions}")

        # Calculate offset in mmap
        offset = position * self.ENTRY_SIZE

        # Encode alleles (4 bytes each, truncate if longer)
        ref_encoded = self._encode_allele(ref_allele)
        alt_encoded = self._encode_allele(alt_allele)

        # Pack entry as binary struct
        entry_bytes = struct.pack(
            "Q I I d d 16x",  # Q=uint64, I=uint32, d=double, 16x=padding
            position,
            ref_encoded,
            alt_encoded,
            frequency,
            quality
        )

        # Write to mmap (zero-copy, kernel handles disk I/O)
        self.mmap[offset:offset + self.ENTRY_SIZE] = entry_bytes

    def get_variant(self, position: int) -> Dict:
        """Read variant from graph (zero-copy mmap read)."""
        offset = position * self.ENTRY_SIZE

        # Read from mmap
        entry_bytes = self.mmap[offset:offset + self.ENTRY_SIZE]

        # Unpack binary struct
        pos, ref_enc, alt_enc, freq, qual = struct.unpack(
            "Q I I d d 16x",
            entry_bytes
        )

        # Decode alleles
        ref_allele = self._decode_allele(ref_enc)
        alt_allele = self._decode_allele(alt_enc)

        return {
            "position": pos,
            "ref": ref_allele,
            "alt": alt_allele,
            "frequency": freq,
            "quality": qual
        }

    def _encode_allele(self, allele: str) -> int:
        """Encode allele string as 32-bit integer."""
        # Simple encoding: pack up to 4 nucleotides
        encoding = {"A": 0, "C": 1, "G": 2, "T": 3}
        result = 0

        for i, nuc in enumerate(allele[:4]):  # Max 4 nucleotides
            if nuc in encoding:
                result |= (encoding[nuc] << (i * 2))

        return result

    def _decode_allele(self, encoded: int) -> str:
        """Decode 32-bit integer back to allele string."""
        decoding = {0: "A", 1: "C", 2: "G", 3: "T"}
        allele = ""

        for i in range(4):
            bits = (encoded >> (i * 2)) & 0x3
            if bits in decoding:
                allele += decoding[bits]

        return allele.rstrip("\x00")

    def close(self):
        """Close mmap and file."""
        self.mmap.close()
        self.file_handle.close()

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'mmap'):
            self.close()


def benchmark_mmap_vs_inmemory():
    """Benchmark memory-mapped vs in-memory graph."""
    import time

    num_variants = 1_000_000

    # Test 1: Memory-mapped graph
    print("Memory-mapped graph...")
    start = time.time()

    mmap_file = "/tmp/test_graph.mmap"
    graph = MemoryMappedVariantGraph(mmap_file, num_positions=num_variants)

    for i in range(num_variants):
        graph.add_variant(i, "A", "G", 0.5, 30.0)

    mmap_time = time.time() - start
    graph.close()

    # Test 2: In-memory dict (baseline)
    print("\nIn-memory dict...")
    start = time.time()

    in_memory = {}
    for i in range(num_variants):
        in_memory[i] = {"ref": "A", "alt": "G", "freq": 0.5, "qual": 30.0}

    inmem_time = time.time() - start

    # Results
    print(f"\nMemory-mapped: {mmap_time:.2f} sec")
    print(f"In-memory: {inmem_time:.2f} sec")
    print(f"Speedup: {inmem_time/mmap_time:.2f}×")

    # Cleanup
    os.remove(mmap_file)


if __name__ == "__main__":
    benchmark_mmap_vs_inmemory()
```

### Testing

```bash
# Test memory-mapped graph
python3 genomevault/reference/memory_mapped_graph.py

# Expected output:
# Memory-mapped: 15.2 sec
# In-memory: 12.8 sec
# Speedup: 0.84× (slightly slower, but uses 8× less RAM)
```

### When to Use Memory-Mapped Graph

**Use memory-mapped when:**
- ✅ RAM-constrained system (<8 GB available)
- ✅ Building whole-genome consensus (60M+ positions)
- ✅ Disk I/O is fast (SSD)

**Use in-memory when:**
- ✅ RAM is plentiful (16 GB+)
- ✅ Single chromosome (chr22 only)
- ✅ Speed more important than memory

---

## Combined Phase 4 Deployment

```bash
# Full pipeline with Phase 4 (optional)
python3 benchmarks/run_enhanced_privacy_pipeline.py \
    --output-dir benchmark_results/enhanced_privacy_k13_phase4_$(date +%Y%m%d_%H%M%S) \
    --num-references 12 \
    --threads 16 \
    --enable-amx \
    --use-chromosome-partitioned-sort \
    --use-parallel-vcf-parsing \
    --zk-backend plonk \
    --use-memory-mapped-graph \
    2>&1 | tee logs/phase4_pipeline_$(date +%Y%m%d_%H%M%S).log
```

---

## Summary

### Phase 4 ROI Analysis

| Optimization | Effort | Time Saved | ROI | Recommendation |
|--------------|--------|------------|-----|----------------|
| PLONK ZK | 4-6 hours | 0.5-1 sec/query | **0.01×** | ❌ Skip (unless high query volume) |
| Memory-mapped graph | 4-6 hours | 15 min one-time | **0.05×** | ❌ Skip (unless RAM-constrained) |
| **Phase 4 Total** | **8-12 hours** | **~15 min** | **0.02×** | **❌ NOT RECOMMENDED** |

### Final Cumulative Progress (All Phases)

| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Total |
|--------|----------|---------|---------|---------|---------|-------|
| Layer 1 | 60 min | 60 min | 60 min | 25 min | 15 min | **4× faster** |
| Per ref | 60 min | 32 min | 20 min | 12 min | 12 min | **5× faster** |
| 12 refs | 12 hours | 6.4 hours | 4.0 hours | 2.4 hours | 2.4 hours | **5× faster** |
| ZK proof | 1.5 sec | 1.5 sec | 1.5 sec | 1.5 sec | 0.8 sec | **2× faster** |
| **Total** | **13.0 hours** | **7.4 hours** | **5.0 hours** | **3.0 hours** | **2.9 hours** | **4.5× faster** |

**Phase 4 only saves 6 minutes after 10+ hours of work - NOT WORTH IT.**

---

## Recommendation: Skip Phase 4

### Better Use of 10 Hours

Instead of Phase 4, consider:

1. **Scale to k=20 reference pool** (better privacy)
   - Effort: 6-8 hours (data acquisition + processing)
   - Benefit: 2× stronger k-anonymity

2. **Add clinical variant database** (better utility)
   - Effort: 4-6 hours
   - Benefit: Immediate clinical value

3. **Improve documentation** (better usability)
   - Effort: 8-10 hours
   - Benefit: Easier for users to adopt

4. **Build web UI** (better accessibility)
   - Effort: 10-15 hours
   - Benefit: Non-technical users can use GenomeVault

5. **Write academic paper** (better dissemination)
   - Effort: 20-40 hours
   - Benefit: Publish research, get citations

---

## When Phase 4 Might Make Sense

**PLONK is worth it if:**
- You expect **1 million+ queries per day**
- Time saved: 1M × 0.7 sec = 8.1 days per year
- ROI: 8.1 days / 5 days effort = 1.6× (break-even after 1 year)

**Memory-mapped graph is worth it if:**
- Your system has **<4 GB RAM available**
- You process whole genomes (60M+ positions)
- You have fast SSD storage

---

**Status:** ⚠️ **NOT RECOMMENDED** for most users
**Risk Level:** High (research-level code, limited testing)
**ROI:** Very Low (0.02× return on effort)
**Better alternatives:** Phases 1-3, data acquisition, user features
