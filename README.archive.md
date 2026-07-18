# GenomeVault

**Privacy-Preserving Genomic Computing with Cryptographic Guarantees**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](benchmark_results/FINAL_VALIDATION_SUMMARY.md)

---

## The Problem

Genomic data is uniquely sensitive - it identifies individuals, reveals disease risks, and exposes family relationships. Yet genomic research requires data sharing across institutions.

Current solutions force an impossible trade-off:

| Approach | Privacy | Speed | Utility |
|----------|---------|-------|---------|
| Raw data sharing | None | Fast | Full |
| Homomorphic encryption | Strong | Hours/query | Limited |
| Differential privacy | Statistical | Fast | Degraded |

**Result:** Research stalls because sharing data means destroying privacy.

---

## The Solution

GenomeVault enables **cryptographic privacy** with **sub-second queries** and **full analytical utility**.

### How It Works

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

### The 7 Layers

| Layer | Function | Output |
|-------|----------|--------|
| 1. Byzantine Consensus | Merge public references (hg38, hg19, chm13) | Consensus genome |
| 2. Guide Strands | k=12 real genomes as blind middlemen | Privacy indirection |
| 3. Alignment | Align to guides (never to public refs) | BAM file |
| 4. GDiff Encoding | Differential encoding vs guide pool | ~15 MB encrypted |
| 5. HDC Encoding | Hyperdimensional vector projection | 1-10 KB per query |
| 6. ZK Proofs | Groth16 proofs (128-bit security) | 743-byte proof |
| 7. PIR Queries | Information-theoretic retrieval | 0 bits leaked |

---

## Performance

**Validated on real whole-genome data:**

| Operation | Time | Security |
|-----------|------|----------|
| Full pipeline | 2.11s | k=12 anonymity |
| HDC encoding | 0.5ms | 99.2% accuracy |
| ZK proof generation | 0.74s | 128-bit |
| PIR query | 4.33ms | 0 bits leaked |

---

## Quick Start

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

---

## Use Cases

- **Clinical Genomics**: Query patient variants without exposing raw data
- **Multi-Site Studies**: Collaborate across institutions without data transfer
- **Rare Disease Research**: Pool data while preserving individual privacy
- **Pharmacogenomics**: Drug response prediction with cryptographic guarantees

---

## Key Features

- **Adaptive HDC Encoding**: 99.2% accuracy with Numba JIT acceleration
- **Real ZK Proofs**: Production Groth16 via Circom (not simulation)
- **IT-PIR Protocol**: Information-theoretic security (quantum-resistant)
- **GPU Acceleration**: Metal (Apple Silicon) and CUDA support

---

## Documentation

- [API Usage Guide](docs/API_USAGE_GUIDE.md)
- [GDiff Format Rationale](docs/GDIFF_RATIONALE.md)
- [Privacy Architecture](docs/SECURE_GUIDE_REFERENCE_SYSTEM.md)
- [Academic Paper](docs/GenomeVault_Paper_Current/GenomeVault_Academic_Paper.pdf)

---

## Architecture

```
genomevault/
├── pipelines/           # Unified production pipeline
├── hypervector_transform/  # Adaptive HDC encoder
├── differential_encoding/  # GDiff format implementation
│   └── gdiff/             # Encoder, validator, schemas
├── zk_proofs/            # Groth16 ZK proof system
├── pir/                  # IT-PIR protocol
└── cli/                  # Command-line interface
```

---

## Citation

If you use GenomeVault in research, please cite:

```bibtex
@software{genomevault2025,
  title={GenomeVault: Privacy-Preserving Genomic Computing},
  author={Vinaik, Rohan},
  year={2025},
  url={https://github.com/rohanvinaik/GenomeVault}
}
```

---

## License

AGPL-3.0 - See [LICENSE](LICENSE) for details.

---

**GenomeVault**: Because genomic privacy shouldn't require sacrificing research utility.
