# CLAUDE.md

Quick reference for Claude Code when working with the GenomeVault codebase.

---

## CRITICAL: Data Protection

**NEVER delete `data/guide_strands/`** - Contains 338 GB of irreplaceable guide strand data (30+ hours to regenerate).

---

## Core Architecture: 7-Layer Privacy Pipeline

GenomeVault implements a privacy-preserving genomic encoding system where **experimental data never touches public references directly**.

```
Layer 1: Byzantine Consensus    → Build reference from hg38+hg19+chm13
Layer 2: Guide Strands          → k=12 real genomic samples (blind middleman)
Layer 3: Experimental Alignment → Align patient data to guides (NOT consensus!)
Layer 4: GDiff Encoding         → Differential encoding vs guide pool
Layer 5: HDC Encoding           → Adaptive hypervectors (99.2% accuracy)
Layer 6: ZK Proofs              → Groth16 proofs (128-bit security)
Layer 7: PIR Queries            → Information-theoretic retrieval (0 bits leaked)
```

**The Iron Law:** Any contact between experimental data and public reference invalidates all privacy guarantees.

---

## Key Files

| Component | Location |
|-----------|----------|
| **Unified Pipeline** | `genomevault/pipelines/unified_pipeline.py` |
| **CLI** | `genomevault/cli/unified.py` |
| **HDC Encoder** | `genomevault/hypervector_transform/adaptive_encoder.py` |
| **GDiff Encoder** | `genomevault/differential_encoding/gdiff/encoder.py` |
| **ZK Prover** | `genomevault/zk_proofs/prover.py` |
| **PIR Protocol** | `genomevault/pir/it_pir_protocol.py` |
| **Archive** | `archive/` (legacy code, do not import) |

---

## Essential Commands

```bash
# Setup
pip install -e ".[dev]"

# Run unified pipeline
python -m genomevault.cli.main unified status    # Check component readiness
python -m genomevault.cli.main unified run \
    --fastq-r1 sample_R1.fq.gz \
    --fastq-r2 sample_R2.fq.gz \
    --guides data/guide_strands \
    --output pipeline_output

# Run tests
pytest tests/
```

---

## Performance

| Stage | Time | Output |
|-------|------|--------|
| GDiff Encoding | 1.36s | Differential variants |
| HDC Encoding | 0.5ms | 4096D hypervector |
| ZK Proof | 0.74s | 743 bytes, 128-bit security |
| PIR Query | 4.33ms | 0 bits leaked |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Check module `__init__.py` exports |
| GPU not detected | `GENOMEVAULT_BACKEND=auto` |
| ZK setup fails | `./benchmarks/setup_groth16_enhanced.sh` |

---

**Version:** 1.3.0 | **Updated:** January 2025
