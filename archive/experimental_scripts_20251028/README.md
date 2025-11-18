# Archived Experimental Scripts

**Date**: 2025-10-28
**Reason**: Code organization cleanup - migrating working functionality to core codebase

## What was preserved in core codebase:

1. **Guide extraction**: Added to `genomevault/differential_encoding/align_to_reference_pool.py`
   - Method: `PrivacyPreservingReferencePoolAligner.extract_guide_sequences_from_bams()`
   - Replaces ad-hoc Bash commands with proper Python API

2. **Privacy-preserving alignment**: Already in core at `align_to_reference_pool.py`
   - Multi-part index @SQ header fix
   - Short-read preset optimization
   - SAM→BAM workflow with header rebuild

3. **k=3 benchmark**: Moved to `benchmarks/run_k3_whole_genome_benchmark.py`

## Scripts archived here:

**Temporary monitoring scripts** (not core functionality):
- Monitoring scripts were ephemeral utilities for tracking long-running processes
- Not needed for reproducibility since core methods handle the actual work

**Experimental/one-off scripts**:
- Scripts that didn't work or were superseded by core implementations
- Shell scripts from root directory that aren't part of standard workflow

## To use archived code:

These scripts are preserved for reference only. For actual work, use:
- `genomevault.differential_encoding.align_to_reference_pool` module (core implementation)
- `benchmarks/run_k3_whole_genome_benchmark.py` (benchmark script)
