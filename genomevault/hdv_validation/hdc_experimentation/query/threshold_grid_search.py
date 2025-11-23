#!/usr/bin/env python3
"""
Threshold Grid Search for Biophysical Layers

Systematically tests threshold combinations to optimize individual layer frequencies.

Phase 1: Individual layers (AT_DOMINANT, GC_DOMINANT, EXTREME_AT, EXTREME_GC)
Phase 2: Multi-bank contexts (TATA, CpG, heterochromatin)
Phase 3: Exotic structural motifs

Usage:
    cd genomevault/hdv_validation/hdc_experimentation
    python3 query/threshold_grid_search.py

Output:
    - Tests 12,500 threshold configurations
    - Outputs top 10 configurations ranked by total error
    - Provides exact threshold values to apply to lens_aware_simd_query_engine.py

Author: Claude Code
Date: November 23, 2025
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import time
import logging
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from lens_aware_simd_query_engine import (
    LensAwareSIMDQueryEngine,
    BIOPHYSICAL_CONTEXTS,
    LAYER_TO_BIT,
    AdaptiveThresholdCalibrator,
    BiophysicalSignatureEncoder
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for a single threshold test."""
    at_dominant_ratio: float
    at_dominant_percentile: int
    gc_dominant_ratio: float
    gc_dominant_percentile: int
    extreme_at_percentile: int
    extreme_gc_percentile: int


@dataclass
class LayerResult:
    """Result for a single layer test."""
    config: ThresholdConfig
    at_dominant_freq: float
    gc_dominant_freq: float
    extreme_at_freq: float
    extreme_gc_freq: float
    at_dominant_error: float
    gc_dominant_error: float
    extreme_at_error: float
    extreme_gc_error: float
    total_error: float


class GridSearchOptimizer:
    """Optimizes biophysical layer thresholds via grid search."""

    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)

        # Load bank magnitudes once (using proven loop approach from lens_aware_simd_query_engine.py)
        logger.info("Loading bank magnitudes from HDF5...")
        with h5py.File(self.h5_path, 'r') as f:
            dataset_name = 'split_ternary_vectors' if 'split_ternary_vectors' in f else 'all_bank_vectors'
            all_banks = f[dataset_name][:]

            num_chunks = len(all_banks)
            self.bank_mags = np.zeros((num_chunks, 6), dtype=np.float32)

            logger.info(f"Computing bank magnitudes for {num_chunks:,} chunks...")

            if dataset_name == 'split_ternary_vectors':
                # Split ternary format: extract banks chunk by chunk
                for i in range(num_chunks):
                    gc_bank = all_banks[i, 1, :]      # Vector1_GC
                    at_bank = all_banks[i, 3, :]      # Vector2_AT
                    hinge_bank = all_banks[i, 2, :]   # Hinge

                    # Bank 1: AT pathway
                    self.bank_mags[i, 0] = np.sum(at_bank[at_bank > 0])    # T-rich
                    self.bank_mags[i, 1] = np.sum(-at_bank[at_bank < 0])   # A-rich

                    # Bank 2: GC pathway
                    self.bank_mags[i, 2] = np.sum(gc_bank[gc_bank > 0])    # G-rich
                    self.bank_mags[i, 3] = np.sum(-gc_bank[gc_bank < 0])   # C-rich

                    # Bank 3: Hinge
                    self.bank_mags[i, 4] = np.sum(hinge_bank[hinge_bank > 0])   # Y→R
                    self.bank_mags[i, 5] = np.sum(-hinge_bank[hinge_bank < 0])  # R→Y

                    if i % 500000 == 0 and i > 0:
                        logger.info(f"  Processed {i:,} / {num_chunks:,} chunks...")
            else:
                # Standard 3-bank format
                for i in range(num_chunks):
                    at_bank = all_banks[i, 0, :]
                    gc_bank = all_banks[i, 1, :]
                    hinge_bank = all_banks[i, 2, :]

                    self.bank_mags[i, 0] = np.sum(at_bank[at_bank > 0])
                    self.bank_mags[i, 1] = np.sum(-at_bank[at_bank < 0])
                    self.bank_mags[i, 2] = np.sum(gc_bank[gc_bank > 0])
                    self.bank_mags[i, 3] = np.sum(-gc_bank[gc_bank < 0])
                    self.bank_mags[i, 4] = np.sum(hinge_bank[hinge_bank > 0])
                    self.bank_mags[i, 5] = np.sum(-hinge_bank[hinge_bank < 0])

                    if i % 500000 == 0 and i > 0:
                        logger.info(f"  Processed {i:,} / {num_chunks:,} chunks...")

        self.n_chunks = len(self.bank_mags)
        logger.info(f"Loaded {self.n_chunks:,} chunks")

        # Pre-compute bank totals and ratios
        self.bank1_total = self.bank_mags[:, 0] + self.bank_mags[:, 1]
        self.bank2_total = self.bank_mags[:, 2] + self.bank_mags[:, 3]
        self.at_gc_ratio = self.bank1_total / (self.bank2_total + 1e-6)
        self.gc_at_ratio = self.bank2_total / (self.bank1_total + 1e-6)

        # Target frequencies
        self.targets = {
            'at_dominant': 0.22,
            'gc_dominant': 0.18,
            'extreme_at': 0.03,
            'extreme_gc': 0.02,
        }

    def test_config(self, config: ThresholdConfig) -> LayerResult:
        """Test a single threshold configuration."""
        # Compute thresholds from percentiles
        at_mag_thresh = np.percentile(self.bank1_total, config.at_dominant_percentile)
        gc_mag_thresh = np.percentile(self.bank2_total, config.gc_dominant_percentile)
        extreme_at_thresh = np.percentile(self.at_gc_ratio, config.extreme_at_percentile)
        extreme_gc_thresh = np.percentile(self.gc_at_ratio, config.extreme_gc_percentile)

        # Test AT_DOMINANT
        at_dominant = (self.at_gc_ratio > config.at_dominant_ratio) & (self.bank1_total > at_mag_thresh)
        at_dominant_freq = np.sum(at_dominant) / self.n_chunks

        # Test GC_DOMINANT
        gc_dominant = (self.gc_at_ratio > config.gc_dominant_ratio) & (self.bank2_total > gc_mag_thresh)
        gc_dominant_freq = np.sum(gc_dominant) / self.n_chunks

        # Test EXTREME_AT
        extreme_at = self.at_gc_ratio > extreme_at_thresh
        extreme_at_freq = np.sum(extreme_at) / self.n_chunks

        # Test EXTREME_GC
        extreme_gc = self.gc_at_ratio > extreme_gc_thresh
        extreme_gc_freq = np.sum(extreme_gc) / self.n_chunks

        # Compute errors
        at_dom_error = abs(at_dominant_freq - self.targets['at_dominant']) / self.targets['at_dominant']
        gc_dom_error = abs(gc_dominant_freq - self.targets['gc_dominant']) / self.targets['gc_dominant']
        extreme_at_error = abs(extreme_at_freq - self.targets['extreme_at']) / self.targets['extreme_at']
        extreme_gc_error = abs(extreme_gc_freq - self.targets['extreme_gc']) / self.targets['extreme_gc']

        total_error = at_dom_error + gc_dom_error + extreme_at_error + extreme_gc_error

        return LayerResult(
            config=config,
            at_dominant_freq=at_dominant_freq,
            gc_dominant_freq=gc_dominant_freq,
            extreme_at_freq=extreme_at_freq,
            extreme_gc_freq=extreme_gc_freq,
            at_dominant_error=at_dom_error,
            gc_dominant_error=gc_dom_error,
            extreme_at_error=extreme_at_error,
            extreme_gc_error=extreme_gc_error,
            total_error=total_error,
        )

    def grid_search(
        self,
        at_ratios: List[float],
        at_percentiles: List[int],
        gc_ratios: List[float],
        gc_percentiles: List[int],
        extreme_at_percentiles: List[int],
        extreme_gc_percentiles: List[int],
    ) -> List[LayerResult]:
        """Run grid search over threshold space."""
        configs = [
            ThresholdConfig(
                at_dominant_ratio=at_r,
                at_dominant_percentile=at_p,
                gc_dominant_ratio=gc_r,
                gc_dominant_percentile=gc_p,
                extreme_at_percentile=ea_p,
                extreme_gc_percentile=eg_p,
            )
            for at_r, at_p, gc_r, gc_p, ea_p, eg_p in product(
                at_ratios, at_percentiles, gc_ratios, gc_percentiles,
                extreme_at_percentiles, extreme_gc_percentiles
            )
        ]

        logger.info(f"Testing {len(configs):,} threshold configurations...")
        logger.info("")

        results = []
        start_time = time.perf_counter()

        for i, config in enumerate(configs):
            if i % 1000 == 0 and i > 0:
                elapsed = time.perf_counter() - start_time
                rate = i / elapsed
                remaining = len(configs) - i
                eta_seconds = remaining / rate
                eta_minutes = eta_seconds / 60

                logger.info(f"  Progress: {i:,}/{len(configs):,} ({i/len(configs)*100:.1f}%) | "
                           f"Rate: {rate:.0f} configs/sec | ETA: {eta_minutes:.1f} min")

            result = self.test_config(config)
            results.append(result)

        return results


def main():
    """Run Phase 1: Individual layer optimization."""

    h5_path = Path("output/encoded_genome_6banks_split_ternary.h5")

    if not h5_path.exists():
        logger.error(f"File not found: {h5_path}")
        logger.error("Make sure you run this script from the hdc_experimentation directory:")
        logger.error("  cd genomevault/hdv_validation/hdc_experimentation")
        logger.error("  python3 query/threshold_grid_search.py")
        return 1

    logger.info("="*80)
    logger.info("PHASE 1: Individual Layer Threshold Optimization")
    logger.info("="*80)
    logger.info("")

    # Initialize optimizer
    optimizer = GridSearchOptimizer(h5_path)

    # Define search space
    logger.info("Search space:")
    at_ratios = [1.3, 1.4, 1.5, 1.6, 1.7]
    at_percentiles = [60, 65, 70, 75, 80]
    gc_ratios = [1.1, 1.15, 1.2, 1.25, 1.3]
    gc_percentiles = [50, 55, 60, 65, 70]
    extreme_at_percentiles = [95, 96, 97, 98, 99]
    extreme_gc_percentiles = [96, 97, 98, 99]

    logger.info(f"  AT_DOMINANT ratios: {at_ratios}")
    logger.info(f"  AT_DOMINANT percentiles: {at_percentiles}")
    logger.info(f"  GC_DOMINANT ratios: {gc_ratios}")
    logger.info(f"  GC_DOMINANT percentiles: {gc_percentiles}")
    logger.info(f"  EXTREME_AT percentiles: {extreme_at_percentiles}")
    logger.info(f"  EXTREME_GC percentiles: {extreme_gc_percentiles}")

    total_configs = (len(at_ratios) * len(at_percentiles) * len(gc_ratios) *
                     len(gc_percentiles) * len(extreme_at_percentiles) * len(extreme_gc_percentiles))
    logger.info(f"  Total configurations: {total_configs:,}")
    logger.info("")

    # Run grid search
    t0 = time.perf_counter()
    results = optimizer.grid_search(
        at_ratios, at_percentiles, gc_ratios, gc_percentiles,
        extreme_at_percentiles, extreme_gc_percentiles
    )
    t1 = time.perf_counter()

    logger.info(f"✓ Grid search complete in {(t1-t0):.1f}s")
    logger.info("")

    # Sort by total error
    results.sort(key=lambda r: r.total_error)

    # Report top 10 configurations
    logger.info("="*80)
    logger.info("TOP 10 CONFIGURATIONS (by total error)")
    logger.info("="*80)
    logger.info("")

    for i, result in enumerate(results[:10], 1):
        logger.info(f"Rank {i}:")
        logger.info(f"  Configuration:")
        logger.info(f"    AT_DOMINANT: ratio={result.config.at_dominant_ratio}, percentile={result.config.at_dominant_percentile}")
        logger.info(f"    GC_DOMINANT: ratio={result.config.gc_dominant_ratio}, percentile={result.config.gc_dominant_percentile}")
        logger.info(f"    EXTREME_AT: percentile={result.config.extreme_at_percentile}")
        logger.info(f"    EXTREME_GC: percentile={result.config.extreme_gc_percentile}")
        logger.info(f"  Frequencies:")
        logger.info(f"    AT_DOMINANT: {result.at_dominant_freq*100:.2f}% (target 22.0%, error {result.at_dominant_error*100:.1f}%)")
        logger.info(f"    GC_DOMINANT: {result.gc_dominant_freq*100:.2f}% (target 18.0%, error {result.gc_dominant_error*100:.1f}%)")
        logger.info(f"    EXTREME_AT:  {result.extreme_at_freq*100:.2f}% (target  3.0%, error {result.extreme_at_error*100:.1f}%)")
        logger.info(f"    EXTREME_GC:  {result.extreme_gc_freq*100:.2f}% (target  2.0%, error {result.extreme_gc_error*100:.1f}%)")
        logger.info(f"  Total error: {result.total_error:.4f}")
        logger.info("")

    # Report best config
    best = results[0]
    logger.info("="*80)
    logger.info("BEST CONFIGURATION")
    logger.info("="*80)
    logger.info("")
    logger.info("Optimal thresholds:")
    logger.info(f"  at_dominant_ratio = {best.config.at_dominant_ratio}")
    logger.info(f"  at_dominant_percentile = {best.config.at_dominant_percentile}")
    logger.info(f"  gc_dominant_ratio = {best.config.gc_dominant_ratio}")
    logger.info(f"  gc_dominant_percentile = {best.config.gc_dominant_percentile}")
    logger.info(f"  extreme_at_percentile = {best.config.extreme_at_percentile}")
    logger.info(f"  extreme_gc_percentile = {best.config.extreme_gc_percentile}")
    logger.info("")
    logger.info("Apply these to lens_aware_simd_query_engine.py:")
    logger.info(f"  Line ~408: at_dominant_ratio = {best.config.at_dominant_ratio}")
    logger.info(f"  Line ~409: at_dominant_magnitude = np.percentile(bank1_total, {best.config.at_dominant_percentile})")
    logger.info(f"  Line ~410: gc_dominant_ratio = {best.config.gc_dominant_ratio}")
    logger.info(f"  Line ~411: gc_dominant_magnitude = np.percentile(bank2_total, {best.config.gc_dominant_percentile})")
    logger.info(f"  Line ~417: extreme_at_threshold = np.percentile(at_gc_ratio, {best.config.extreme_at_percentile})")
    logger.info(f"  Line ~418: extreme_gc_threshold = np.percentile(gc_at_ratio, {best.config.extreme_gc_percentile})")

    return 0


if __name__ == '__main__':
    exit(main())
