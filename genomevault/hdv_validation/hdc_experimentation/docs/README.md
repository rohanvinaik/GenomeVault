# HDC Experimentation Documentation

**Research Area:** Hyperdimensional Computing for Genomic Compression
**Focus:** Split Binary Quantization Architecture
**Last Updated:** 2025-11-21

---

## Organization

This directory contains all documentation for the HDC experimentation research, organized into three categories:

```
docs/
├── README.md                    # This file - navigation guide
├── theory/                      # Theoretical analysis & experiment logs
│   ├── EXPERIMENTAL_DATA_COLLECTION.md  # Primary research log ⭐
│   ├── SPLIT_BINARY_ARCHITECTURE_ISSUE_ANALYSIS.md
│   └── SPLIT_BANK_ARCHITECTURE.md
├── reports/                     # Summaries & quick-start guides
│   ├── SPLIT_BINARY_QUICKSTART.md
│   ├── split_binary_validation_report.md
│   └── split_binary_validation_report_BEFORE_FIX.md
└── results/                     # Raw experimental results (JSON)
    ├── split_binary_validation_results.json
    └── split_binary_validation_results_BEFORE_FIX.json
```

---

## Quick Navigation

### 🔬 For Researchers

**START HERE:**
📄 [`theory/EXPERIMENTAL_DATA_COLLECTION.md`](theory/EXPERIMENTAL_DATA_COLLECTION.md)

This is the primary scientific log documenting:
- All experimental runs with parameters and results
- Hypotheses and their validation
- Key findings and insights
- Parameter space exploration
- Theoretical framework
- Future research directions

### 🏃 For Quick Implementation

**START HERE:**
📄 [`reports/SPLIT_BINARY_QUICKSTART.md`](reports/SPLIT_BINARY_QUICKSTART.md)

Quick-start guide for using the split binary architecture.

### 📊 For Latest Results

**CHECK:**
📄 [`results/split_binary_validation_results.json`](results/split_binary_validation_results.json)

Latest validation accuracy numbers in machine-readable format.

### 🧠 For Architecture Deep-Dive

**READ:**
1. [`theory/SPLIT_BANK_ARCHITECTURE.md`](theory/SPLIT_BANK_ARCHITECTURE.md) - Biophysical bank design
2. [`theory/SPLIT_BINARY_ARCHITECTURE_ISSUE_ANALYSIS.md`](theory/SPLIT_BINARY_ARCHITECTURE_ISSUE_ANALYSIS.md) - Root cause analysis and solutions

---

## Key Documents

### Theory (Primary Research)

#### [`EXPERIMENTAL_DATA_COLLECTION.md`](theory/EXPERIMENTAL_DATA_COLLECTION.md) ⭐
**Purpose:** Comprehensive scientific log of all experimentation
**Contents:**
- Research questions
- All experimental runs (parameters, hypotheses, results)
- Parameter space exploration
- Theoretical framework
- Major discoveries
- Future work

**Use this for:**
- Understanding the research progression
- Citing experimental results
- Planning future experiments
- Writing papers/reports

#### [`SPLIT_BINARY_ARCHITECTURE_ISSUE_ANALYSIS.md`](theory/SPLIT_BINARY_ARCHITECTURE_ISSUE_ANALYSIS.md)
**Purpose:** Root cause analysis of the flat 36% accuracy issue
**Contents:**
- Signal-to-noise analysis
- SNR = D/N relationship
- Sparsification effects
- Parameter optimization rationale

**Use this for:**
- Understanding why initial experiment failed
- Learning about D/N ratio criticality
- Parameter selection guidelines

#### [`SPLIT_BANK_ARCHITECTURE.md`](theory/SPLIT_BANK_ARCHITECTURE.md)
**Purpose:** Biophysical motivation for 6-bank design
**Contents:**
- Hydrophobic (AT) bank
- Major groove (GC) bank
- Hinge flexibility (purine-pyrimidine) bank
- Rationale for ternary → binary split

**Use this for:**
- Understanding bank design choices
- Biophysical interpretability
- Alternative bank proposals

### Reports (Summaries)

#### [`split_binary_validation_report.md`](reports/split_binary_validation_report.md)
**Purpose:** Latest validation results summary
**Contents:**
- Accuracy metrics
- Per-nucleotide performance
- Confusion matrices

**Use this for:**
- Quick accuracy checks
- Presenting results
- Comparing runs

#### [`SPLIT_BINARY_QUICKSTART.md`](reports/SPLIT_BINARY_QUICKSTART.md)
**Purpose:** Quick implementation guide
**Contents:**
- How to encode a genome
- How to query positions
- Expected performance

**Use this for:**
- Getting started quickly
- Code examples
- Troubleshooting

### Results (Raw Data)

#### [`split_binary_validation_results.json`](results/split_binary_validation_results.json)
**Purpose:** Latest experimental results in JSON format
**Contents:**
- Accuracy metrics
- Per-nucleotide breakdown
- Sample metadata
- Timestamps

**Use this for:**
- Automated analysis
- Plotting/visualization
- Reproducibility

---

## Research Status

### Current Experiment (2025-11-21)

**Experiment 3: Genome Structure Exploitation**
- **Hypothesis:** Human genome's non-random structure (45% repeats, conserved motifs) enables D=5,120 to achieve >97% accuracy
- **Parameters:** D=5,120, N=1,024, SNR=5.0, binary quantization, no sparsification
- **Status:** 🔄 Encoding in progress
- **Expected completion:** ~2-3 hours
- **Documentation:** See `EXPERIMENTAL_DATA_COLLECTION.md` > Experiment 3

### Key Findings So Far

1. **Storage scales with D/N ratio only** (not absolute values)
2. **SNR = D/N must be ≥5.0** for position-level resolution
3. **Sparsification is harmful** (should keep 100% of data)
4. **Overlap must scale with chunk size** (higher % for smaller chunks)

---

## How to Contribute

### Adding a New Experiment

1. **Update** [`theory/EXPERIMENTAL_DATA_COLLECTION.md`](theory/EXPERIMENTAL_DATA_COLLECTION.md):
   - Add new experiment section with parameters, hypothesis, results
   - Update "Current Experiment" status
   - Add to parameter space exploration if novel parameters tested

2. **Save results** to `results/` directory:
   - Use descriptive filename: `experiment_<N>_<description>_<date>.json`
   - Include all metadata (parameters, timestamp, accuracy, etc.)

3. **Create summary report** in `reports/`:
   - Brief markdown summary for quick reference
   - Link to full experiment details in `EXPERIMENTAL_DATA_COLLECTION.md`

### Updating Documentation

**Theory documents:** Should be comprehensive, mathematical, and permanent record
**Report documents:** Should be concise, practical, and user-friendly
**Result files:** Machine-readable JSON for automation and plotting

---

## Citation

If using this research, please cite:

```
[To be added after publication]
```

For now, reference:
> GenomeVault HDC Experimentation Suite
> Split Binary Quantization for Genomic Compression
> November 2025

---

## Contact

For questions about this research:
- **Experimental questions:** See `EXPERIMENTAL_DATA_COLLECTION.md`
- **Implementation questions:** See `SPLIT_BINARY_QUICKSTART.md`
- **Architecture questions:** See `SPLIT_BANK_ARCHITECTURE.md`

---

**Last Updated:** 2025-11-21
**Maintained by:** GenomeVault research team
