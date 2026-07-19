# GenomeVault

**Compute on a genome without ever holding it.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Your genome is a password you can never change — and it is not only yours. It carries your siblings, your parents, your children; leak it once and it is exposed for three generations, permanently. Yet genomic research runs on shared data, so every institution meets the same trade nobody should have to make: keep the genome private, or make it useful.

| Approach | Privacy | Speed | Utility |
|----------|---------|-------|---------|
| Raw data sharing | None | Fast | Full |
| Homomorphic encryption | Strong | Hours/query | Limited |
| Differential privacy | Statistical | Fast | Degraded |

Homomorphic encryption buys privacy at hours per query; differential privacy buys speed by degrading the answer; raw sharing buys everything except the one thing that matters. The trade looks like a law of nature. It is an unsolved engineering problem, and GenomeVault solves it: **cryptographic privacy, sub-second queries, and full analytical utility, at once.** A 150 GB genome becomes a 1–10 KB encrypted query; a zero-knowledge proof answers the question without revealing the genome that answered it; private information retrieval fetches the answer without revealing the question. Zero bits leaked — and the whole pipeline runs in 2.11 seconds on real whole-genome data.

```
Raw Genome (150 GB)
       |
       v
[7-Layer Privacy Pipeline]
       |
       v
Encrypted Query (1-10 KB) -----> GenomeVault Network
       |                              |
       v                              v
Zero-Knowledge Proof          Private Information Retrieval
(proves without revealing)     (retrieves without exposing query)
```

## Seven layers, each closing one way to leak

Privacy here is not a single trick; it is seven, composed so that no layer ever sees what the layer before it protected. The genome is never aligned to a public reference — it is aligned to a pool of real genomes acting as blind middlemen, encoded as a difference against that pool, projected into a hypervector, and answered only through a proof and a retrieval that each reveal nothing.

| Layer | Function | Output |
|-------|----------|--------|
| 1. Byzantine Consensus | Merge public references (hg38, hg19, chm13) | Consensus genome |
| 2. Guide Strands | k=12 real genomes as blind middlemen | Privacy indirection |
| 3. Alignment | Align to guides (never to public refs) | BAM file |
| 4. GDiff Encoding | Differential encoding vs guide pool | ~15 MB encrypted |
| 5. HDC Encoding | Hyperdimensional vector projection | 1-10 KB per query |
| 6. ZK Proofs | Groth16 proofs (128-bit security) | 743-byte proof |
| 7. PIR Queries | Information-theoretic retrieval | 0 bits leaked |

## What it costs

Measured on real whole-genome data — not a model, not a projection:

| Operation | Time | Security |
|-----------|------|----------|
| Full pipeline | 2.11s | k=12 anonymity |
| HDC encoding | 0.5ms | 99.2% accuracy |
| ZK proof generation | 0.74s | 128-bit |
| PIR query | 4.33ms | 0 bits leaked |

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Check system status
python -m genomevault.cli.main unified status

# Run pipeline
python -m genomevault.cli.main unified run \
    --fastq-r1 sample_R1.fq.gz \
    --fastq-r2 sample_R2.fq.gz \
    --guides data/guide_strands \
    --output output/
```

## Where it goes

- **Clinical genomics** — query a patient's variants without exposing the raw genome.
- **Multi-site studies** — collaborate across institutions with no data transfer at all.
- **Rare disease research** — pool cohorts while every individual stays private.
- **Pharmacogenomics** — predict drug response under cryptographic guarantee.

## Real, not simulated

The distinction the field usually blurs, stated plainly:

- **Adaptive HDC encoding** — 99.2% accuracy, Numba JIT-accelerated.
- **Real ZK proofs** — production Groth16 via Circom, not a simulation of one.
- **IT-PIR protocol** — information-theoretic security, so it is quantum-resistant by construction, not by key length.
- **GPU acceleration** — Metal (Apple Silicon) and CUDA.

## Architecture

```
genomevault/
├── pipelines/               # Unified production pipeline
├── hypervector_transform/   # Adaptive HDC encoder
├── differential_encoding/   # GDiff format implementation
│   └── gdiff/               # Encoder, validator, schemas
├── zk_proofs/               # Groth16 ZK proof system
├── pir/                     # IT-PIR protocol
└── cli/                     # Command-line interface
```

## Documentation

- [API Usage Guide](docs/API_USAGE_GUIDE.md)
- [GDiff Format Rationale](docs/GDIFF_RATIONALE.md)
- [Privacy Architecture](docs/SECURE_GUIDE_REFERENCE_SYSTEM.md)
- [Academic Paper](docs/GenomeVault_Paper_Current/GenomeVault_Academic_Paper.pdf)

## Citation

```bibtex
@software{genomevault2025,
  title={GenomeVault: Privacy-Preserving Genomic Computing},
  author={Vinaik, Rohan},
  year={2025},
  url={https://github.com/rohanvinaik/GenomeVault}
}
```

## License

AGPL-3.0 — see [LICENSE](LICENSE).

---

The privacy-versus-utility trade was never fundamental to genomics. It was the cost of not having built this yet.
