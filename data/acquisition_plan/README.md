# GenomeVault Data Acquisition System

**Navigation hub for scaling GenomeVault benchmarks from k=3 to k=10+ with diverse, production-grade genomic datasets.**

---

## Quick Start

**New to data acquisition?** Start here:

1. **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - High-level overview (5 min read)
2. **[Quick Start Guide](QUICK_START_GUIDE.md)** - Step-by-step execution (15 min)
3. **[Complete Plan](DATA_ACQUISITION_PLAN.md)** - Full reference documentation (60+ pages)

**Want to run it now?**

```bash
# Phase 1: Scale European pool from k=3 to k=10
./scripts/create_data_structure.sh
./scripts/download_european_samples.sh  # Downloads 7 additional samples
python scripts/generate_sample_metadata.py --pool european
python scripts/generate_pool_manifest.py --pool european

# Verify download
python scripts/verify_downloaded_data.py --pool european
```

---

## Current Status

| Pool | Current Size | Target Size | Status |
|------|--------------|-------------|--------|
| **European** | k=3 | k=10 | ⚠️ Need 7 more samples |
| **East Asian** | k=0 | k=10 | ⏳ Not started |
| **African** | k=0 | k=10 | ⏳ Not started |
| **South Asian** | k=0 | k=10 | ⏳ Not started |

---

## Documentation Structure

```
data/acquisition_plan/
├── README.md                      # ← You are here (navigation hub)
├── IMPLEMENTATION_SUMMARY.md      # High-level overview and motivation
├── QUICK_START_GUIDE.md          # Step-by-step execution guide
└── DATA_ACQUISITION_PLAN.md      # Complete 60+ page reference

scripts/
├── create_data_structure.sh       # Setup organized directories
├── download_european_samples.sh   # Phase 1: European k=3→k=10
├── download_diversity_samples.sh  # Phase 2: Global diversity
├── generate_sample_metadata.py    # Per-sample JSON metadata
├── generate_pool_manifest.py      # Pool-level aggregation
└── verify_downloaded_data.py      # Quality checks
```

---

## Key Features

✅ **Production-Grade Data Sources**
- 1000 Genomes Project (publicly validated)
- GIAB/Genome in a Bottle (NIST reference standards)
- Ashkenazi Trio (clinical validation datasets)

✅ **Automated Download & Verification**
- Parallel downloads with progress tracking
- MD5 checksums for data integrity
- Automatic retry on network failures

✅ **Population Diversity**
- European (CEU, GBR, FIN)
- East Asian (CHB, JPT, CHS)
- African (YRI, LWK, GWD)
- South Asian (PJL, GIH, ITU)

✅ **Organized Metadata**
- Per-sample JSON with ancestry, coverage, quality metrics
- Pool-level manifests for API integration
- Standardized schema for programmatic access

---

## Quick Reference

### Download Commands

```bash
# European pool (k=3 → k=10)
./scripts/download_european_samples.sh

# All diversity pools (k=0 → k=10 each)
./scripts/download_diversity_samples.sh --population all

# Single ancestry
./scripts/download_diversity_samples.sh --population east_asian
```

### Metadata Generation

```bash
# Generate metadata for all European samples
python scripts/generate_sample_metadata.py --pool european

# Generate pool manifest
python scripts/generate_pool_manifest.py --pool european

# Verify all metadata is consistent
python scripts/verify_downloaded_data.py --check-metadata
```

### Integration with GenomeVault

```bash
# Update reference pool configuration
python scripts/update_pool_config.py --pool european --k 10

# Run benchmark with new k=10 pool
python benchmarks/run_enhanced_privacy_pipeline.py --k-anonymity 10
```

---

## Support & Troubleshooting

See [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) for detailed instructions and troubleshooting.

---

**Last Updated**: October 24, 2025  
**Status**: Ready for Phase 1 execution  
**Documentation**: See data/acquisition_plan/ for complete guides
