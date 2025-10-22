# GenomeVault Academic Papers

**Location**: [`GenomeVault_Paper_Current/`](GenomeVault_Paper_Current/)
**Last Updated**: October 20, 2025, 10:10 UTC

## 📚 Current Versions (Dual-Track)

We maintain **TWO publication-ready versions** optimized for different audiences:

### Version 2.0: Church-Enhanced (Strategic)
**Target**: Funding proposals, George Church, strategic partnerships

**Quick Access**:
- **📄 PDF**: [GenomeVault_Academic_Paper.pdf](GenomeVault_Academic_Paper.pdf) (33 pages, 429 KB)
- **📝 LaTeX**: [GenomeVault_Academic_Paper.tex](GenomeVault_Academic_Paper.tex)

**Key Features**:
- Blockchain genomics section (Nebula/HLTH.network failure analysis)
- DNA↔HDC structural correspondence (biological justification)
- Research applications enabled (rare disease, epistasis, federated GWAS)
- Collaboration-focused limitations
- Strategic positioning for fundability

**Use For**: Grant applications, industry partnerships, Church Lab pitches

---

### Version 2.1: Journal-Ready (Neutral)
**Target**: Peer-reviewed journals (Bioinformatics, Nature Biotechnology)

**Quick Access**:
- **📄 PDF**: [GenomeVault_Academic_Paper_Journal_Ready.pdf](GenomeVault_Academic_Paper_Journal_Ready.pdf) (12 pages, 177 KB)
- **📝 LaTeX**: [GenomeVault_Academic_Paper_Journal_Ready.tex](GenomeVault_Academic_Paper_Journal_Ready.tex)

**Key Features**:
- Comprehensive limitations section (synthetic data, validation needs)
- Neutral academic tone throughout ("may enable", "suggests")
- Concise structure (12 pages vs 33)
- Appropriate hedging on all claims
- No promotional language

**Use For**: Journal submissions, academic conferences, rigorous peer review

---

## 📊 Shared Resources

- **📊 Figures**: [GenomeVault_Paper_Current/paper_figures/](GenomeVault_Paper_Current/paper_figures/)
- **📋 README**: [GenomeVault_Paper_Current/README.md](GenomeVault_Paper_Current/README.md)
- **📜 Version Changelog**: [GenomeVault_Paper_Current/VERSION_CHANGELOG.md](GenomeVault_Paper_Current/VERSION_CHANGELOG.md)

## 🔍 Version Comparison

| Aspect | Church-Enhanced (v2.0) | Journal-Ready (v2.1) |
|--------|------------------------|----------------------|
| **Pages** | 33 | 12 |
| **Tone** | Confident | Neutral |
| **Blockchain** | ✅ Full section | ❌ Omitted |
| **Limitations** | Collaboration-focused | Critical |
| **Target** | Funding/partnerships | Peer review |

## Archived Versions

Old paper versions are stored in [`archive_paper_versions/`](archive_paper_versions/).

## 🔨 Compilation

### Church-Enhanced Version (v2.0)
```bash
cd /Users/rohanvinaik/genomevault/docs
pdflatex GenomeVault_Academic_Paper.tex
pdflatex GenomeVault_Academic_Paper.tex  # Second pass
```

### Journal-Ready Version (v2.1)
```bash
cd /Users/rohanvinaik/genomevault/docs
pdflatex GenomeVault_Academic_Paper_Journal_Ready.tex
pdflatex GenomeVault_Academic_Paper_Journal_Ready.tex  # Second pass
```

### Compile Both + Deploy
```bash
cd /Users/rohanvinaik/genomevault/docs
# Compile both versions
pdflatex GenomeVault_Academic_Paper.tex && pdflatex GenomeVault_Academic_Paper.tex
pdflatex GenomeVault_Academic_Paper_Journal_Ready.tex && pdflatex GenomeVault_Academic_Paper_Journal_Ready.tex
# Copy to all locations
cp GenomeVault_Academic_Paper*.pdf GenomeVault_Paper_Current/
cp GenomeVault_Academic_Paper*.pdf ~/Desktop/
```

## 📋 Selection Guide

**Choose Church-Enhanced (v2.0) when:**
- 💰 Writing grant applications (NIH, NSF)
- 🤝 Pitching partnerships or collaborations
- 🎯 Highlighting market opportunity and strategic vision
- 📈 Demonstrating awareness of genomics startup landscape
- 💡 Positioning for George Church or Church Lab

**Choose Journal-Ready (v2.1) when:**
- 📄 Submitting to peer-reviewed journals
- 🔬 Presenting at academic conferences
- ⚖️ Requiring maximum objectivity and rigor
- 🛡️ Responding to reviewer concerns
- 📊 Academic job talks or fellowship applications

---

**For full details on differences, see**: [VERSION_CHANGELOG.md](GenomeVault_Paper_Current/VERSION_CHANGELOG.md)
