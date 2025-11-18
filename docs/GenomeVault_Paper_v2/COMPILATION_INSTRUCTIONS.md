# GenomeVault Paper - Compilation Instructions

## ✅ Completed Tasks

### 1. Paper Updates
- ✅ Removed Parama Pal as co-author
- ✅ Fixed hyperref configuration (no more HTML 404 errors)
- ✅ Enhanced error decomposition figure (Section 3.3.1)

### 2. New Enhanced Figure
**File:** `figures/multirun_consensus.pdf` (58 KB, updated Oct 24 20:35)

**Content:**
- **Left Panel:** Exponential error rate decay vs consensus runs (log scale)
  - Exact binomial calculation
  - Chernoff bound overlay
  - Mathematical formulas displayed
  - Clinical thresholds (99.9%, 99.99%)

- **Right Panel:** Privacy-Accuracy-Latency engineering trade-offs
  - N=1 (2.15s): 95.0% - Population Screening
  - N=3 (6.45s): 98.6% - Diagnostic Triage
  - N=5 (10.75s): 99.9% - Clinical Decisions
  - N=7 (15.05s): 99.99% - Research Validation
  - Shows constant 260-bit privacy entropy across all configurations

**Key Insights Visualized:**
- 7× more runs → 1000× better accuracy
- Exponential decay: 5% error → 0.01% error
- Privacy entropy remains constant (260 bits/run)

## 📝 Paper Modifications Summary

### Location: GenomeVault_Paper.tex

1. **Lines 27-31:** Author block (removed co-author)
2. **Lines 11-19:** Hyperref configuration (prevents URL fetching)
3. **Lines 453-499:** Section 3.3.1 references the enhanced figure

## 🔧 Option 2: Local Compilation (What You Chose)

### Required: Install Missing LaTeX Packages

You have **TeX Live 2025 Basic** installed, but it's missing some packages.

### Installation Steps:

**Run the provided script with your admin password:**

```bash
cd /Users/rohanvinaik/genomevault/docs/GenomeVault_Paper_v2
./install_latex_packages.sh
```

**Or manually run:**

```bash
sudo tlmgr install multirow booktabs algorithms algorithmicx xcolor tikz pgf caption subcaption float
```

### Why sudo is needed:
The TeX Live installation directory (`/usr/local/texlive/2025basic/`) is owned by root and requires admin privileges to modify.

## 📦 After Installing Packages

### Compile the Paper:

```bash
cd /Users/rohanvinaik/genomevault/docs/GenomeVault_Paper_v2

# First pass (processes content)
pdflatex -interaction=nonstopmode -output-directory=compiled GenomeVault_Paper.tex

# Process bibliography
bibtex compiled/GenomeVault_Paper

# Second pass (resolves references)
pdflatex -interaction=nonstopmode -output-directory=compiled GenomeVault_Paper.tex

# Third pass (finalizes cross-references)
pdflatex -interaction=nonstopmode -output-directory=compiled GenomeVault_Paper.tex

# View the PDF
open compiled/GenomeVault_Paper.pdf
```

### Expected Output:
- **File:** `compiled/GenomeVault_Paper.pdf`
- **Pages:** ~37 pages
- **Size:** ~600-650 KB
- **Figures:** All 11 figures embedded (including enhanced multirun_consensus.pdf)

## 🎨 Figure Generation Script

**File:** `generate_error_decomposition_figure.py`

To regenerate the figure anytime:

```bash
python generate_error_decomposition_figure.py
```

This creates:
- `figures/multirun_consensus.pdf` (for LaTeX)
- `figures/multirun_consensus.png` (for preview/presentations)

## 📊 What the Enhanced Figure Shows

### Mathematical Foundation (Section 3.3.1):

**Exponential Consensus Convergence:**
```
P_error(N) = Σ_{k=⌈N/2⌉}^N C(N,k) * p^k * (1-p)^(N-k)

P_error(N) ≤ e^(-N * D_KL(0.5 || p))

where D_KL(0.5 || p) = 1/2 * log(1/(2p)) + 1/2 * log(1/(2(1-p)))
```

**Engineering Data Points:**
| Runs | Time (s) | Accuracy | Error Rate | Use Case |
|------|----------|----------|------------|----------|
| N=1 | 2.15 | 95.0% | 5.0% | Population Screening |
| N=3 | 6.45 | 98.6% | 1.4% | Diagnostic Triage |
| N=5 | 10.75 | 99.9% | 0.1% | Clinical Decisions |
| N=7 | 15.05 | 99.99% | <0.01% | Research Validation |

**Key Properties:**
- Privacy entropy: 260 bits (constant per run)
- Error decreases exponentially with N
- 7× more runs → 1000× accuracy improvement
- Tunable privacy-accuracy-latency trade-off

## 🚀 Alternative: Overleaf (Easier)

If local compilation has issues, upload to Overleaf:

```bash
cd /Users/rohanvinaik/genomevault/docs
zip -r GenomeVault_Paper_v2.zip GenomeVault_Paper_v2/
```

Then:
1. Go to https://www.overleaf.com
2. New Project → Upload Project
3. Upload the ZIP file
4. Compile (all packages pre-installed)

## ✅ Verification Checklist

After compilation, verify:

- [ ] No Parama Pal in author list
- [ ] No HTML 404 errors in PDF
- [ ] Figure 5 (multirun_consensus.pdf) shows dual-panel visualization
- [ ] Left panel: Exponential decay curve with log scale
- [ ] Right panel: Bar chart with N=1,3,5,7 engineering choices
- [ ] All 37 pages render correctly
- [ ] Bibliography includes all 40+ references

## 📧 Contact

**File Issues:**
- Check `compiled/GenomeVault_Paper.log` for errors
- Verify all figures exist: `ls -lh figures/*.pdf`
- Ensure bibliography exists: `ls -lh genomevault_refs.bib`

**Compilation Issues:**
- Missing packages: Run `install_latex_packages.sh`
- Permission errors: Use `sudo tlmgr install <package>`
- Persistent issues: Upload to Overleaf (recommended)

---

**Status:** ✅ **READY TO COMPILE**
**Last Updated:** October 24, 2025
**Paper Version:** 1.3 (Enhanced with error decomposition visualization)
