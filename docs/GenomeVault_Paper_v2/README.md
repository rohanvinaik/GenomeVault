# GenomeVault Academic Paper (Version 2)

**Complete rewrite based on new architecture and comprehensive benchmark results**

## Overview

This directory contains the LaTeX source for the GenomeVault academic paper, targeting tier-1 venues like:
- **Nature Biotechnology** (systems biology, genomics)
- **USENIX Security** (cryptography, privacy)
- **IEEE Transactions on Biomedical Engineering** (medical applications)

The paper presents GenomeVault as a cryptographically private, hyperdimensional computing platform for secure genomic computation, achieving 38.4× empirical compression and 2.15-second end-to-end latency with $2^{516}$-bit combined security.

## File Structure

```
GenomeVault_Paper_v2/
├── GenomeVault_Paper.tex          # Main paper (20-25 pages)
├── genomevault_refs.bib           # Bibliography (BibTeX format)
├── figures/                       # Figure directory
│   ├── FIGURES_README.md         # Figure specifications
│   └── *.pdf                     # Generated figures (to be created)
├── README.md                      # This file
└── compiled/                      # Output directory (generated)
    ├── GenomeVault_Paper.pdf     # Compiled PDF
    └── *.aux, *.log, *.out       # LaTeX auxiliary files
```

## Paper Structure

### Sections (with page estimates)

1. **Abstract** (1 page)
   - 200 words summarizing system, metrics, and contributions

2. **Introduction** (2 pages)
   - Genomic privacy paradox
   - Limitations of existing approaches
   - GenomeVault's architectural approach
   - 7 key contributions

3. **System Architecture** (4 pages)
   - 7-stage pipeline overview
   - 4-layer privacy stack (alignment, reference pools, differential, HDC)
   - ZK proof verification
   - Blockchain attestation
   - IT-PIR queries
   - KAN-HD analytics
   - Federated learning

4. **Mathematical Foundations** (3 pages)
   - SHA-256² dual-barrier security
   - Probabilistic alignment entropy
   - Tunable privacy-accuracy spectrum
   - HDC collision analysis
   - IT-PIR formal privacy proofs
   - ZK proof soundness

5. **Performance Evaluation** (4 pages)
   - Experimental setup (ERR3239334 dataset)
   - End-to-end pipeline benchmarks
   - Compression analysis (264× architectural, 38.4× empirical)
   - Accuracy validation (99.9% at 5 runs)
   - Security evaluation
   - Scalability analysis
   - Clinical query performance

6. **Blockchain Integration and Economics** (2 pages)
   - Attestation architecture
   - Institutional integration
   - Economic model (\$0.17/genome/year vs \$1.40 VCF)
   - Total addressable market (\$53.6M/year)

7. **Related Work** (2 pages)
   - Genomic privacy techniques (HE, DP, SMPC, TEEs)
   - Compression and encoding (CRAM, variant graphs, ML)
   - Blockchain and data sovereignty
   - Hyperdimensional computing

8. **Discussion** (2 pages)
   - Privacy as architecture
   - Accuracy-privacy spectrum
   - Interpretability and selective decryption
   - Economic viability
   - Limitations and future work
   - Broader impact

9. **Conclusion** (1 page)
   - Summary of contributions
   - Impact on genomic medicine

10. **Appendices** (2 pages)
    - Cryptographic circuit definitions
    - Differential encoding proofs
    - Experimental scripts

**Total:** ~23-25 pages (conference/journal full-length paper)

## Key Metrics and Results

### Performance
- **Total latency:** 2.15 seconds (end-to-end)
- **Differential encoding:** 1.36s, 11× compression
- **HDC encoding:** 0.50ms, 24× compression
- **ZK proof generation:** 0.74s, 743-byte proof
- **PIR query:** 4.33ms, 0-bit leakage

### Compression
- **Architectural:** 264× (11× diff × 24× HDC)
- **Empirical:** 38.4× (VCF 1,500 KB → 39.06 KB)
- **From FASTQ:** 1,500× (150 GB → 100 MB theoretical)

### Security
- **Combined entropy:** $2^{516}$ bits (SHA-256² dual-barrier)
- **Alignment randomization:** 260 bits (information-theoretic)
- **HDC inversion:** $2^{800,000}$ search space
- **ZK soundness:** $2^{-128}$ error probability
- **IT-PIR leakage:** 0 bits (perfect privacy) + <7 bits side-channel

### Accuracy
- **1 run:** 95.0% (2.15s)
- **5 runs:** 99.9% (10.75s, clinical threshold)
- **7 runs:** 99.99% (15.05s, research grade)
- **BRCA1/2 detection:** 100%
- **KAN-HD selective decode:** 99.7%

### Economics
- **Per-genome cost:** \$0.17/year (vs \$1.40 VCF)
- **100M genomes:** \$17M/year (vs \$140M VCF)
- **Annual savings:** \$123M for 100M genomes
- **Blockchain attestation:** \$0.01/genome (one-time)

## Compilation Instructions

### Method 1: Local LaTeX Installation

#### Prerequisites
- **LaTeX distribution:**
  - macOS: MacTeX (`brew install --cask mactex`)
  - Linux: TeX Live (`sudo apt install texlive-full`)
  - Windows: MiKTeX
- **Required packages:** (included in full distributions)
  - amsmath, amssymb, amsthm
  - graphicx, tikz
  - algorithm, algorithmic
  - booktabs, multirow
  - hyperref

#### Compile

```bash
cd docs/GenomeVault_Paper_v2

# Create output directory
mkdir -p compiled

# Compile (run 3 times for references/citations)
pdflatex -output-directory=compiled GenomeVault_Paper.tex
bibtex compiled/GenomeVault_Paper
pdflatex -output-directory=compiled GenomeVault_Paper.tex
pdflatex -output-directory=compiled GenomeVault_Paper.tex

# View PDF
open compiled/GenomeVault_Paper.pdf  # macOS
xdg-open compiled/GenomeVault_Paper.pdf  # Linux
```

#### One-Line Compilation

```bash
latexmk -pdf -output-directory=compiled GenomeVault_Paper.tex
```

### Method 2: Overleaf (Recommended)

1. **Create Overleaf project:**
   - Go to https://www.overleaf.com
   - Create new project → Upload Project
   - Upload ZIP of `GenomeVault_Paper_v2/`

2. **Project settings:**
   - Compiler: pdfLaTeX
   - TeX Live version: 2023 or later
   - Main document: `GenomeVault_Paper.tex`

3. **Compile:**
   - Click "Recompile" button
   - Auto-compile on save (enable in settings)

4. **Collaborate:**
   - Share link with co-authors
   - Track changes and comments
   - Version history automatically saved

### Method 3: Docker (Reproducible)

```bash
# Run LaTeX in Docker container
docker run --rm -v $(pwd):/workspace texlive/texlive:latest \
  bash -c "cd /workspace && \
  pdflatex GenomeVault_Paper.tex && \
  bibtex GenomeVault_Paper && \
  pdflatex GenomeVault_Paper.tex && \
  pdflatex GenomeVault_Paper.tex"
```

## Generating Figures

Figures must be created before compilation. See `figures/FIGURES_README.md` for specifications.

### Quick Figure Generation

```bash
# From project root
cd /Users/rohanvinaik/genomevault

# Install dependencies
pip install matplotlib seaborn numpy pandas

# Generate all figures (when script is created)
python analysis/generate_paper_figures.py \
  --output docs/GenomeVault_Paper_v2/figures/

# Individual figures
python analysis/plot_pipeline_breakdown.py
python analysis/plot_consensus_accuracy.py
python analysis/plot_storage_comparison.py
```

### Required Figures

| # | Filename | Type | Status |
|---|----------|------|--------|
| 1 | `pipeline_overview.pdf` | Schematic | TODO |
| 2 | `dual_barrier.pdf` | Schematic | TODO |
| 3 | `multirun_consensus.pdf` | Plot | TODO |
| 4 | `hdc_collision.pdf` | Plot | TODO |
| 5 | `pipeline_breakdown.pdf` | Bar chart | TODO |
| 6 | `scaling_variants.pdf` | Line plot | TODO |
| 7 | `storage_comparison.pdf` | Bar chart | TODO |
| 8 | `economic_scaling.pdf` | Line plot | TODO |
| 9 | `zk_proof_flow.pdf` | Schematic | TODO |
| 10 | `pir_flow.pdf` | Schematic | TODO |
| 11 | `blockchain_architecture.pdf` | Schematic | TODO |
| 12 | `kan_hd_pipeline.pdf` | Hybrid | TODO |

**Temporary Workaround:** To compile without figures, comment out `\includegraphics` lines in `.tex` file.

## Troubleshooting

### Missing Packages

**Error:** `LaTeX Error: File 'package.sty' not found`

**Solution:**
```bash
# macOS
sudo tlmgr update --self
sudo tlmgr install <package-name>

# Linux
sudo apt install texlive-<package-collection>

# Or install full distribution
sudo apt install texlive-full  # ~5 GB
```

### Bibliography Not Showing

**Issue:** References show as `[?]` in PDF

**Solution:** Run compilation sequence (pdflatex → bibtex → pdflatex × 2)
```bash
pdflatex GenomeVault_Paper.tex
bibtex GenomeVault_Paper  # No .tex extension
pdflatex GenomeVault_Paper.tex
pdflatex GenomeVault_Paper.tex
```

### Figure Not Found

**Error:** `LaTeX Error: File 'figures/pipeline_overview.pdf' not found`

**Solution:**
- Generate figures using scripts in `analysis/`
- Or temporarily comment out figure includes

### Overleaf Timeout

**Issue:** Compilation times out on Overleaf (free tier: 1 min limit)

**Solution:**
- Upgrade to Overleaf Premium (4 min timeout)
- Or compile locally using Method 1

### PDF Won't Open

**Issue:** PDF is corrupted or won't open

**Solution:**
```bash
# Clean auxiliary files and recompile
rm compiled/*.aux compiled/*.log compiled/*.out
pdflatex -output-directory=compiled GenomeVault_Paper.tex
```

## Citation

If using this paper structure or results, cite as:

```bibtex
@article{Vinaik2025GenomeVault,
  title={GenomeVault: A Cryptographically Private, Hyperdimensional Platform for Secure, Queryable Genomic Computation},
  author={Vinaik, Rohan and Pal, Parama},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Version History

### Version 2 (October 2025) - Current
- Complete rewrite based on new outline
- Incorporates actual benchmark results (ERR3239334 dataset)
- SHA-256² dual-barrier architecture
- Tunable privacy-accuracy spectrum (multi-run consensus)
- KAN-HD integration
- Comprehensive economic analysis
- Blockchain attestation (Phase 1 + 2)
- ~25 pages, 12 figures, 40+ citations

### Version 1 (Archived)
- Original academic-style paper
- See `docs/archive_paper_versions/version_20251024/`
- Based on earlier architecture and synthetic data

## Target Venues

### Tier 1 - Primary Targets

1. **Nature Biotechnology**
   - Impact factor: 68.2
   - Focus: Genomics, computational biology
   - Format: Article (3,000-5,000 words + figures)
   - Emphasis: Clinical impact, transformative technology

2. **USENIX Security Symposium**
   - Tier: A* security conference
   - Focus: Cryptography, privacy-preserving systems
   - Format: 18 pages (including references)
   - Emphasis: Security proofs, threat model, implementation

3. **IEEE Transactions on Biomedical Engineering**
   - Impact factor: 4.6
   - Focus: Medical applications, clinical validation
   - Format: Full paper (10-12 pages)
   - Emphasis: Performance, accuracy, clinical utility

### Tier 2 - Alternative Venues

4. **Proceedings of the National Academy of Sciences (PNAS)**
   - Broad scientific audience
   - 6 pages + SI

5. **ACM Conference on Computer and Communications Security (CCS)**
   - Top security conference
   - Similar to USENIX format

6. **Bioinformatics (Journal)**
   - Impact factor: 6.9
   - Computational focus

## Derivative Publications

This paper can be split into multiple focused publications:

### Paper 1: Systems (Current)
- **Title:** GenomeVault: A Practical Cryptographically Private Genomics Platform
- **Venue:** USENIX Security or IEEE Transactions
- **Focus:** Full system, performance, security proofs

### Paper 2: Cryptography
- **Title:** SHA-256² Dual-Barrier Architecture for Genomic Privacy
- **Venue:** ACM CCS or NDSS
- **Focus:** Cryptographic proofs, IT-PIR, ZK circuits

### Paper 3: AI/Interpretability
- **Title:** KAN-HD: Interpretable Hyperdimensional Genomic Analysis
- **Venue:** NeurIPS or ICML
- **Focus:** KAN architecture, selective decoding, phenotype prediction

### Paper 4: Economics/Policy
- **Title:** Economic and Policy Implications of Privacy-Preserving Genomics
- **Venue:** Science or Health Affairs
- **Focus:** Cost analysis, market sizing, regulatory compliance

## Contact and Contributions

For questions or contributions to the paper:

- **Lead Author:** Rohan Vinaik (rohan.vinaik@example.edu)
- **GitHub:** https://github.com/rohanvinaik/GenomeVault
- **Issues:** https://github.com/rohanvinaik/GenomeVault/issues

## License

The GenomeVault paper is released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

The associated code is released under **MIT License**.

## Acknowledgments

This paper builds on:
- Benchmark results from `docs/reports/COMPLETE_BENCHMARK_RESULTS.md`
- Security analysis from `docs/guides/HYPERVECTOR_SECURITY.md`
- Blockchain integration from `docs/reports/BLOCKCHAIN_INTEGRATION_COMPLETE.md`
- Economic analysis from `docs/reports/OPTIMIZATION_RESULTS_SUMMARY.md`

Special thanks to the 1000 Genomes Project and UK Biobank for public genomic datasets.

---

**Last Updated:** October 24, 2025
**Status:** Draft v2.0 - Ready for figure generation and compilation
**Target Submission:** December 2025 (USENIX Security) / January 2026 (Nature Biotech)
