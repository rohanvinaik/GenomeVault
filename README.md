# GenomeVault

**Compute on a genome without ever holding it — security, privacy, and utility at once, on a laptop.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

`99.98% accuracy · full-resolution genome at ~2.5 GB · microsecond queries · 256-bit · zero training · runs on a laptop`

Precision medicine was supposed to be here by now. It stalled — but not on the walls anyone planned for. Sequencing a genome costs about $100. The biology moves fast, CRISPR is more powerful than anyone imagined, and the compute infrastructure is fine. What actually stops it is a wall that looks like a law of nature: you can have **security**, or **privacy**, or **utility** — the ability to move, share, and compute on the data — but not all three. More security, less utility. More privacy, less utility. As a two-way trade, that wall is real, and it is *definitionally* unsolvable.

So GenomeVault does not try to. It treats the three as what they actually are — independent axes, not ends of one dial. Security keeps people out; privacy makes the data useless even to someone who gets in; utility lets you compute on it anyway. Nothing says all three cannot be high at once. GenomeVault is the point (1, 1, 1), pushed as far as it goes.

## The key: DNA is not base-4

Treating A/T/G/C as four arbitrary letters is the thing that makes the wall look permanent. They are not arbitrary. Each base is two independent yes/no facts — **purine or pyrimidine** (two rings or one) and **amino or keto** — so the alphabet is Z₂ × Z₂, which is exactly balanced ternary: `−1, 0, +1`. And the zeros are not nothing; they carry structure.

Encode a genome that way and the trade quietly comes apart:

- **Full nucleotide resolution below the naive information bound** — the ternary zeros are structural, not random, so they cost less than a bit. A whole genome, queryable, at ~2.5 GB; distributable as a ~30 MB lossless difference.
- **Microsecond queries** — an answer is a dot product, so lookups run in ~5.81 µs on a laptop, 1,000–1,000,000× faster than reading a BAM.
- **99.98% accuracy** — two independent lenses on the same data (purine/pyrimidine and amino/keto) vote, and a committee settles the hard sites.
- **Information-theoretic privacy** — the genome is only ever encoded as a difference against a pool of other genomes acting as blind middlemen, and a query is answered where the data lives, so the raw sequence never moves. The guarantee is mathematical, not key-length — quantum-resistant by construction. (Distribution rides a packaged ZK/PIR layer; the answer travels, the genome does not.)

## The part that wasn't planned

When the two lenses disagree — the sites the encoding calls "errors" — those positions turn out to be biophysically loaded: **46× enriched for DNase hypersensitivity, p < 10⁻¹⁵**. The encoding's mistakes are functional genomic sites.

And the reconstruction is not a trick. Against T2T ground truth the decode fixes real sequencing errors at a **4,400 : 1** ratio — and it does so the way biology already does. DNA polymerase misincorporates at roughly one base in a few thousand, yet the cell holds the net error rate below one in a million, through redundancy and proofreading rather than a better enzyme. The committee corrects the same way. That hyperdimensional encoding lands on exactly the error-correction mechanism life already uses is the strongest sign that HDC is not an arbitrary representation for a genome — it is the native one.

## How it works

Four phases. No genome is ever aligned to a public reference; it is encoded as a difference against a blind pool, projected into ternary, packed, and answered only through a query that reveals nothing.

```
Phase 0 — Privacy encoding      Differential encoding vs a pooled set of public references.
                                Random reference selection makes reconstruction infeasible.
                                → GDiff file (~30 MB) + reference-pool mapping

Phase 1 — Two-bank OTP encoding 512 bp chunks → Z₂×Z₂ decomposition → Bank 1 (purine/pyrimidine),
                                Bank 2 (amino/keto) → Sparse-Hadamard → ternary {−1,0,+1}
                                → int8 ternary vectors, ~2,700 chunks/s (Numba JIT)

Phase 2 — Compression & storage ~48 GB raw vectors → interleaved 4-bit packing → zstd
                                → ~2.5 GB baseline · ~10 s to decompress to query-ready RAM

Phase 3 — Tiered query          Tier 0: two-bank similarity voting, O(1) array access, 99.19%
                                Tier 1: 4-lens committee (99.91%, HIGH confidence when 3+ agree)
                                Tier 2: review queue for genuine committee splits
```

## Quick start

```bash
pip install -e ".[dev]"

# check system status
python -m genomevault.cli.main unified status

# run the pipeline
python -m genomevault.cli.main unified run \
    --fastq-r1 sample_R1.fq.gz \
    --fastq-r2 sample_R2.fq.gz \
    --guides data/guide_strands \
    --output output/
```

## Where it goes

- **Population-level lookups in seconds** — cohort queries without a data-transfer step.
- **Compliance built into the architecture** — the raw genome never leaves, so there is nothing to leak.
- **Rare-disease research without consent bottlenecks** — pool individuals while each stays private.
- **Biophysical insight in real time** — functional sites fall out of the encoding, on edge devices.

## What this is — honestly

Clever iteration on existing tools. Differential encoding, ternary projection, committee voting, private retrieval — none of it is truly novel, and this document will not pretend otherwise. It is a meaningful advance on current implementations, and its one genuinely new move is small and load-bearing: reading DNA as Z₂ × Z₂ instead of base-4, and letting that make the three axes stop fighting.

## Architecture

```
genomevault/
├── differential_encoding/   # GDiff: the privacy-preserving difference format
│   └── gdiff/
├── hypervector/             # the ternary two-bank encoder
├── hdv_validation/          # committee voting, lens calibration, error correction
├── zk/  ·  zk_proofs/       # zero-knowledge distribution layer
├── pir/                     # private information retrieval
└── cli/                     # unified command-line interface
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

The privacy-versus-utility trade was never a law. It was an artifact of reading DNA as four letters instead of two bits.
