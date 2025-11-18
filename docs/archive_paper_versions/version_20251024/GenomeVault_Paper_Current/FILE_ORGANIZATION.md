# GenomeVault Paper - File Organization

**Last Reorganized**: October 20, 2025, 00:20 UTC
**Cleanup Version**: v1.0

---

## Current Version (v2.0) - Clean Structure

### Location: `/docs/GenomeVault_Paper_Current/`

This folder contains **all materials** for the current, publication-ready academic paper (Version 2.0).

#### Complete File List:

```
GenomeVault_Paper_Current/
├── README.md                           # Main documentation
├── CHANGELOG_v2.0.md                   # Version 2.0 changes (Church-specific)
├── FILE_ORGANIZATION.md                # This file - organization guide
├── GenomeVault_Academic_Paper.tex      # LaTeX source (108 KB)
├── GenomeVault_Academic_Paper.pdf      # Final PDF (429 KB, 33 pages)
├── GenomeVault_Academic_Paper.md       # Markdown version (for reference)
└── paper_figures/                      # All 4 figures (PDF + PNG)
    ├── figure1_differential_encoding_overview.pdf (54 KB)
    ├── figure1_differential_encoding_overview.png (535 KB)
    ├── figure2_chunking_strategies.pdf (34 KB)
    ├── figure2_chunking_strategies.png (399 KB)
    ├── figure3_hypervector_encoding.pdf (39 KB)
    ├── figure3_hypervector_encoding.png (368 KB)
    ├── figure4_end_to_end_performance.pdf (40 KB)
    ├── figure4_end_to_end_performance.png (434 KB)
    ├── table_s1_system_specs.csv
    └── table_s2_benchmark_config.csv
```

---

## Quick Reference in `/docs/` Root

### Active Files (Keep):

**Paper Files**:
- `GenomeVault_Academic_Paper.tex` - LaTeX source (primary)
- `GenomeVault_Academic_Paper.pdf` - Compiled PDF (primary)
- `GenomeVault_Academic_Paper.aux` - LaTeX auxiliary file
- `GenomeVault_Academic_Paper.log` - LaTeX compilation log
- `GenomeVault_Academic_Paper.out` - LaTeX outline file

**Navigation**:
- `PAPERS.md` - Quick navigation index for all paper versions
- `GenomeVault_Paper_Current/` - Complete current version folder

**Other Documentation** (non-paper):
- Various `.md` files for system documentation
- Architecture diagrams, API references, guides

### Archived Files:

**Location**: `/docs/archive_paper_versions/`

**What's Archived**:
- Old markdown versions (`GenomeVault_Academic_Paper_*.md`)
- Old compilation logs (`compile_*.log`)
- Backup figures folder (`paper_figures_backup_*`)
- Test PDFs (`test_output.pdf`)
- Old README files (`PAPER_README_*.md`)

---

## Compilation Workflow

### Standard LaTeX Compilation:

```bash
# Navigate to docs directory
cd /Users/rohanvinaik/genomevault/docs

# Compile (first pass)
pdflatex GenomeVault_Academic_Paper.tex

# Compile (second pass for cross-references)
pdflatex GenomeVault_Academic_Paper.tex

# Result: GenomeVault_Academic_Paper.pdf (33 pages, 429 KB)
```

### After Compilation:

The generated PDF is in `/docs/` root. To update the Paper_Current folder:

```bash
# Copy updated PDF to Paper_Current
cp GenomeVault_Academic_Paper.pdf GenomeVault_Paper_Current/

# If LaTeX source was modified, copy that too
cp GenomeVault_Academic_Paper.tex GenomeVault_Paper_Current/
```

---

## Version History

### Version 2.0 (Current - Oct 20, 2025)

**File**: `GenomeVault_Academic_Paper.tex` (108 KB) → `GenomeVault_Academic_Paper.pdf` (429 KB, 33 pages)

**Major Changes**:
- Blockchain genomics section (Nebula/HLTH.network)
- DNA↔HDC structural correspondence
- Research applications section
- Collaboration-focused limitations
- Strategic positioning for Church audience

**Complete Changelog**: See [`CHANGELOG_v2.0.md`](CHANGELOG_v2.0.md)

### Version 1.0 (Archived - Oct 19, 2025)

**Files**: Multiple markdown versions in `archive_paper_versions/`

**Characteristics**:
- 31 pages
- Markdown-based with Pandoc conversion
- Original fingerprinting-focused content
- General academic audience

---

## File Naming Conventions

### Current Version:
- **Base name**: `GenomeVault_Academic_Paper`
- **Extensions**: `.tex` (source), `.pdf` (output), `.aux/.log/.out` (LaTeX artifacts)
- **No version suffix** - current version is always unversioned

### Archived Versions:
- **Pattern**: `GenomeVault_Academic_Paper_[DESCRIPTOR]_[DATE].md`
- **Examples**:
  - `GenomeVault_Academic_Paper_20251019.md`
  - `GenomeVault_Academic_Paper_FIXED_20251019.md`
  - `GenomeVault_Academic_Paper_CORRECTED.md`

### Figures:
- **Pattern**: `figure[N]_[description].[pdf|png]`
- **Examples**:
  - `figure1_differential_encoding_overview.pdf`
  - `figure2_chunking_strategies.png`

---

## Access Patterns

### For Quick PDF Access:

**Desktop**: `/Users/rohanvinaik/Desktop/GenomeVault_Academic_Paper.pdf`
**Primary**: `/Users/rohanvinaik/genomevault/docs/GenomeVault_Academic_Paper.pdf`
**Reference**: `/Users/rohanvinaik/genomevault/docs/GenomeVault_Paper_Current/GenomeVault_Academic_Paper.pdf`

All three are identical copies of the current v2.0 PDF.

### For Editing LaTeX:

**Primary Source**: `/Users/rohanvinaik/genomevault/docs/GenomeVault_Academic_Paper.tex`
**Reference Copy**: `/Users/rohanvinaik/genomevault/docs/GenomeVault_Paper_Current/GenomeVault_Academic_Paper.tex`

Edit the **primary source** in `/docs/` root, then copy to Paper_Current after compilation.

### For Figures:

**Only Location**: `/Users/rohanvinaik/genomevault/docs/GenomeVault_Paper_Current/paper_figures/`

Figures are referenced in LaTeX relative to this path:
```latex
\includegraphics[width=\textwidth]{GenomeVault_Paper_Current/paper_figures/figure1_differential_encoding_overview.pdf}
```

---

## Maintenance

### When Creating New Version:

1. **Archive current version**:
   ```bash
   cp GenomeVault_Academic_Paper.tex archive_paper_versions/GenomeVault_Academic_Paper_v2.0_$(date +%Y%m%d).tex
   cp GenomeVault_Academic_Paper.pdf archive_paper_versions/GenomeVault_Academic_Paper_v2.0_$(date +%Y%m%d).pdf
   ```

2. **Update version documentation**:
   - Update `GenomeVault_Paper_Current/README.md` with new version number
   - Create new `CHANGELOG_vX.X.md` documenting changes
   - Update `PAPERS.md` in `/docs/` root

3. **Compile and verify**:
   ```bash
   pdflatex GenomeVault_Academic_Paper.tex
   pdflatex GenomeVault_Academic_Paper.tex
   ```

4. **Update all copies**:
   ```bash
   cp GenomeVault_Academic_Paper.pdf GenomeVault_Paper_Current/
   cp GenomeVault_Academic_Paper.pdf ~/Desktop/
   ```

### When Adding New Figures:

1. **Generate figures** using scripts in `/scripts/`:
   ```bash
   python scripts/generate_paper_figures_v2.py
   ```

2. **Place in paper_figures/**:
   ```bash
   mv figure*.pdf GenomeVault_Paper_Current/paper_figures/
   mv figure*.png GenomeVault_Paper_Current/paper_figures/
   ```

3. **Reference in LaTeX**:
   ```latex
   \includegraphics[width=\textwidth]{GenomeVault_Paper_Current/paper_figures/figure5_new_analysis.pdf}
   ```

4. **Recompile** to embed figures

---

## Cleanup History

### Cleanup v1.0 (October 20, 2025)

**Motivation**: `/docs/` folder was overloaded with old drafts, duplicate folders, and scattered files.

**Actions Taken**:
1. ✅ Moved 6 old markdown versions to `archive_paper_versions/`
2. ✅ Archived old compilation log (`compile.log`)
3. ✅ Removed duplicate `paper_figures/` folder from `/docs/` root
4. ✅ Archived old `PAPER_README.md`
5. ✅ Copied LaTeX source to `Paper_Current/` for complete reference
6. ✅ Created `PAPERS.md` navigation index
7. ✅ Created this organization guide

**Result**:
- Clean `/docs/` root with only active paper files
- All current materials consolidated in `GenomeVault_Paper_Current/`
- All old versions safely archived
- Clear navigation structure

**Files Moved to Archive** (October 20, 2025):
```
archive_paper_versions/
├── GenomeVault_Academic_Paper_20251019.md          (51 KB)
├── GenomeVault_Academic_Paper_FIXED_20251019.md    (51 KB)
├── compile_20251019.log                             (114 KB)
├── paper_figures_backup_20251019/                   (duplicates)
└── PAPER_README_20251019.md                         (7.9 KB)
```

---

## Summary

**Current Structure** (Post-Cleanup):
- ✅ Clean `/docs/` root with LaTeX source and PDF
- ✅ Complete current version in `GenomeVault_Paper_Current/`
- ✅ All old versions archived in `archive_paper_versions/`
- ✅ Easy navigation via `PAPERS.md`
- ✅ Clear compilation workflow
- ✅ No duplicate files

**Best Practices**:
- Always edit **primary** LaTeX source in `/docs/` root
- Compile in `/docs/` directory
- Copy outputs to `Paper_Current/` after successful compilation
- Archive old versions before major changes
- Keep figures only in `Paper_Current/paper_figures/`
- Update documentation when making structural changes

---

**For Questions**: See [`README.md`](README.md) in this folder for paper-specific details.
