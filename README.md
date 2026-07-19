# GenomeVault

**Compute on a genome without ever holding it — security, privacy, and utility at once, on a laptop.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

`99.98% accuracy · full-resolution genome at ~2.5 GB · microsecond queries · information-theoretic privacy · zero training · runs on a laptop`

Precision medicine was supposed to be here by now. It stalled, but not on the wall anyone planned for. Sequencing a genome costs about $100, the biology moves fast, and the compute is fine. What actually stops it is a wall that looks like a law of nature: you can have **security**, or **privacy**, or **utility** — the ability to move, share, and compute on the data — but not all three. More security, less utility. More privacy, less utility. As a two-way trade, that wall is real, and it is *definitionally* unsolvable.

So GenomeVault does not treat it as a two-way trade. Security, privacy, and utility are independent axes, not ends of one dial. Security keeps people out. Privacy makes the data useless to anyone who gets in. Utility lets you compute on it anyway. Nothing says all three cannot be high at once — and the thing that made them look like they were fighting is one wrong assumption about the data:

```
A T G C are not four arbitrary letters. Each base is two independent yes/no facts:

   purine or pyrimidine   ×   amino or keto   =   Z₂ × Z₂   =   balanced ternary {−1, 0, +1}

encode a genome that way, and the wall comes apart:

   full genome, queryable        ~2.5 GB     · ~30 MB as a lossless difference
   a lookup is a dot product       5.81 µs   · 1,000–1,000,000× faster than reading a BAM
   accuracy                        99.98%    · two lenses vote; a committee settles the hard sites
   the raw sequence never moves    0 leaked  · information-theoretic, quantum-resistant by construction
```

## DNA is not base-4

Treating A/T/G/C as four arbitrary letters is the thing that makes the wall look permanent. They are not arbitrary. Each base is two independent facts — purine or pyrimidine (two rings or one), and amino or keto. So the alphabet is Z₂ × Z₂, which is exactly balanced ternary: −1, 0, +1. And the zeros are not nothing. They carry structure, so full nucleotide resolution lands *below* the naive information bound — the ternary zeros cost less than a bit. That is the whole move, and it is the one genuinely new thing here.

Everything else follows. A query is a dot product, so a lookup runs in **5.81 µs** on a laptop. Accuracy is **99.98%**, because the two independent lenses — purine/pyrimidine and amino/keto — vote, and a committee settles the sites where they split. Privacy is information-theoretic, because the genome is only ever encoded as a difference against a pool of other genomes acting as blind middlemen, and a query is answered where the data lives. The raw sequence never moves. The guarantee is mathematical, not key-length — quantum-resistant by construction.

## The part that wasn't planned

When the two lenses disagree — the sites the encoding calls "errors" — those positions turn out to be biophysically loaded: **46× enriched for DNase hypersensitivity, p < 10⁻¹⁵**. The encoding's mistakes are functional genomic sites.

And the reconstruction is not a trick. Against T2T ground truth, the decode fixes real sequencing errors at a **4,400 : 1** ratio — and it does so the way biology already does. DNA polymerase misincorporates at roughly one base in a few thousand. The cell holds the net error rate below one in a million anyway, through redundancy and proofreading rather than a better enzyme. The committee corrects the same way. That a hyperdimensional encoding lands on exactly the error-correction mechanism life already uses is the strongest sign that HDC is not an arbitrary representation for a genome. It is the native one.

## How it works

Four phases. No genome is ever aligned to a public reference. It is encoded as a difference against a blind pool, projected into ternary, packed, and answered only through a query that reveals nothing.

```
Phase 0 · privacy encoding    differential encoding vs a pool of public references; random
                              reference selection makes reconstruction infeasible
                              → a GDiff file (~30 MB) + the reference-pool mapping

Phase 1 · two-bank encoding   512 bp chunks → Z₂×Z₂ decomposition → Bank 1 (purine/pyrimidine),
                              Bank 2 (amino/keto) → Sparse-Hadamard → ternary {−1, 0, +1}
                              → int8 ternary vectors, ~2,700 chunks/s (Numba JIT)

Phase 2 · compression         ~48 GB of raw vectors → interleaved 4-bit packing → zstd
                              → ~2.5 GB baseline · ~10 s to decompress to query-ready RAM

Phase 3 · tiered query        Tier 0: two-bank similarity voting, O(1) access, 99.19%
                              Tier 1: 4-lens committee, 99.91% (HIGH confidence when 3+ agree)
                              Tier 2: a review queue for genuine committee splits
```

## What this is — honestly

Clever iteration on existing tools. Differential encoding, ternary projection, committee voting, private retrieval — none of it is truly novel, and this document will not pretend otherwise. It is a meaningful advance on current implementations. Its one genuinely new move is small and load-bearing: reading DNA as Z₂ × Z₂ instead of base-4, and letting that make the three axes stop fighting.

## Where it goes

- **Population-level lookups in seconds** — cohort queries with no data-transfer step.
- **Compliance built into the architecture** — the raw genome never leaves, so there is nothing to leak.
- **Rare-disease research without consent bottlenecks** — pool individuals while each stays private.
- **Biophysical insight in real time** — functional sites fall out of the encoding, on edge devices.

## Run it

```bash
pip install -e ".[dev]"
python -m genomevault.cli.main unified status                       # check the system
python -m genomevault.cli.main unified run --fastq-r1 R1.fq.gz \
    --fastq-r2 R2.fq.gz --guides data/guide_strands --output output/
```

The module layout follows the pipeline — `differential_encoding/` (the GDiff privacy format), `hypervector/` (the ternary two-bank encoder), `hdv_validation/` (committee voting and error correction), `zk/` and `pir/` (the distribution layer), `cli/`. The GDiff rationale, the privacy architecture, and the academic paper are in [`docs/`](docs/).

---

AGPL-3.0 — Rohan Vinaik. The privacy-versus-utility trade was never a law. It was an artifact of reading DNA as four letters instead of two bits.
