#!/usr/bin/env python3
"""
Validate Split Binary Quantization Architecture

Verifies that the 6-bank split binary encoding:
1. Has the correct structure (6 banks, proper dimensions)
2. Maintains half sparsity per bank (~3.7% active vs original 7.44%)
3. Preserves the within-lens split architecture
4. Correct bank naming and metadata
"""

import h5py
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def analyze_sparsity(data: np.ndarray, name: str, sample_size: int = 1000):
    """Analyze and log sparsity statistics for a bank"""
    # Sample random chunks if dataset is large
    total_chunks = data.shape[0]
    if total_chunks > sample_size:
        sample_indices = np.random.choice(total_chunks, size=sample_size, replace=False)
        sample_indices = np.sort(sample_indices)
        sample_data = data[sample_indices, :]
    else:
        sample_data = data

    zeros = (sample_data == 0).sum()
    ones = (sample_data == 1).sum()
    total = sample_data.size

    zero_pct = 100.0 * zeros / total
    one_pct = 100.0 * ones / total

    logger.info(f"  {name}:")
    logger.info(f"    Active (1):   {one_pct:6.2f}%  ({ones:,} bits)")
    logger.info(f"    Inactive (0): {zero_pct:6.2f}%  ({zeros:,} bits)")

    return one_pct


def validate_split_binary(
    binary_path: Path,
    ternary_path: Path,
    sample_size: int = 1000
):
    """
    Validate split binary quantization

    Args:
        binary_path: Path to 6-bank split binary HDF5
        ternary_path: Path to 3-bank ternary HDF5 (for comparison)
        sample_size: Number of chunks to sample for analysis
    """
    logger.info("="*80)
    logger.info("SPLIT BINARY ARCHITECTURE VALIDATION")
    logger.info("="*80)
    logger.info("")

    # Load binary file
    logger.info(f"Loading split binary: {binary_path}")
    with h5py.File(binary_path, 'r') as f:
        if 'binary_bank_vectors' not in f:
            logger.error("❌ FAILED: 'binary_bank_vectors' dataset not found!")
            return False

        binary_data = f['binary_bank_vectors']
        shape = binary_data.shape
        dtype = binary_data.dtype

        logger.info(f"  Shape: {shape}")
        logger.info(f"  Dtype: {dtype}")

        # Check dimensions
        if len(shape) != 3:
            logger.error(f"❌ FAILED: Expected 3D array, got {len(shape)}D")
            return False

        total_chunks, num_banks, dimension = shape

        if num_banks != 6:
            logger.error(f"❌ FAILED: Expected 6 banks, got {num_banks}")
            return False

        logger.info(f"  ✓ Correct structure: {total_chunks:,} chunks × {num_banks} banks × {dimension:,}D")
        logger.info("")

        # Check metadata
        if 'architecture' in binary_data.attrs:
            arch = binary_data.attrs['architecture']
            logger.info(f"Architecture: {arch}")

        if 'bank_names' in binary_data.attrs:
            bank_names = binary_data.attrs['bank_names']
            expected_names = [
                'Hydrophobic_A', 'Hydrophobic_T',
                'MajorGroove_G', 'MajorGroove_C',
                'Hinge_pos', 'Hinge_neg'
            ]

            if list(bank_names) == expected_names:
                logger.info(f"  ✓ Correct bank names:")
                for i, name in enumerate(bank_names):
                    logger.info(f"    Bank {i}: {name}")
            else:
                logger.warning(f"⚠️  Bank names don't match expected")
                logger.warning(f"  Expected: {expected_names}")
                logger.warning(f"  Got: {list(bank_names)}")

        logger.info("")

        # Analyze sparsity for each bank
        logger.info(f"Analyzing sparsity (sample size: {min(sample_size, total_chunks):,} chunks)...")
        logger.info("")

        bank_labels = [
            "Hydrophobic_A (A positions)",
            "Hydrophobic_T (T positions)",
            "MajorGroove_G (G positions)",
            "MajorGroove_C (C positions)",
            "Hinge_pos (dinucleotide +)",
            "Hinge_neg (dinucleotide -)"
        ]

        sparsities = []
        for bank_idx in range(num_banks):
            bank_data = binary_data[:, bank_idx, :]
            sparsity = analyze_sparsity(bank_data, f"Bank {bank_idx} ({bank_labels[bank_idx]})", sample_size)
            sparsities.append(sparsity)
            logger.info("")

    # Load ternary file for comparison
    logger.info(f"Loading ternary file for comparison: {ternary_path}")
    with h5py.File(ternary_path, 'r') as f:
        ternary_data = f['all_bank_vectors']
        ternary_shape = ternary_data.shape

        logger.info(f"  Ternary shape: {ternary_shape}")

        # Sample and analyze ternary sparsity
        sample_indices = np.random.choice(
            ternary_shape[0],
            size=min(sample_size, ternary_shape[0]),
            replace=False
        )
        sample_indices = np.sort(sample_indices)
        sample_data = ternary_data[sample_indices, :, :]

        logger.info("")
        logger.info("Ternary bank sparsity (for comparison):")

        ternary_labels = ["Hydrophobic", "MajorGroove", "Hinge"]
        ternary_sparsities = []

        for bank_idx in range(3):
            bank_data = sample_data[:, bank_idx, :]

            # Count +1, -1, 0
            pos_ones = (bank_data == 1).sum()
            neg_ones = (bank_data == -1).sum()
            zeros = (bank_data == 0).sum()
            total = bank_data.size

            active_pct = 100.0 * (pos_ones + neg_ones) / total
            ternary_sparsities.append(active_pct)

            logger.info(f"  {ternary_labels[bank_idx]}:")
            logger.info(f"    Active (+1/-1): {active_pct:6.2f}%")
            logger.info(f"    Inactive (0):   {100.0 * zeros / total:6.2f}%")

    logger.info("")
    logger.info("="*80)
    logger.info("VALIDATION RESULTS")
    logger.info("="*80)
    logger.info("")

    # Check if binary banks have roughly half the sparsity of ternary banks
    validation_passed = True

    # Hydrophobic split (banks 0 and 1)
    hydrophobic_binary_avg = (sparsities[0] + sparsities[1]) / 2
    hydrophobic_ternary = ternary_sparsities[0]
    hydrophobic_ratio = hydrophobic_binary_avg / (hydrophobic_ternary / 2)

    logger.info(f"Hydrophobic lens:")
    logger.info(f"  Ternary active: {hydrophobic_ternary:.2f}%")
    logger.info(f"  Binary avg active: {hydrophobic_binary_avg:.2f}%")
    logger.info(f"  Expected: {hydrophobic_ternary / 2:.2f}%")
    logger.info(f"  Ratio: {hydrophobic_ratio:.2f}x")

    if abs(hydrophobic_ratio - 1.0) > 0.2:  # Allow 20% deviation
        logger.warning(f"  ⚠️  Sparsity ratio outside expected range")
        validation_passed = False
    else:
        logger.info(f"  ✓ Sparsity ratio correct")

    logger.info("")

    # MajorGroove split (banks 2 and 3)
    majorgroove_binary_avg = (sparsities[2] + sparsities[3]) / 2
    majorgroove_ternary = ternary_sparsities[1]
    majorgroove_ratio = majorgroove_binary_avg / (majorgroove_ternary / 2)

    logger.info(f"MajorGroove lens:")
    logger.info(f"  Ternary active: {majorgroove_ternary:.2f}%")
    logger.info(f"  Binary avg active: {majorgroove_binary_avg:.2f}%")
    logger.info(f"  Expected: {majorgroove_ternary / 2:.2f}%")
    logger.info(f"  Ratio: {majorgroove_ratio:.2f}x")

    if abs(majorgroove_ratio - 1.0) > 0.2:
        logger.warning(f"  ⚠️  Sparsity ratio outside expected range")
        validation_passed = False
    else:
        logger.info(f"  ✓ Sparsity ratio correct")

    logger.info("")

    # Hinge split (banks 4 and 5)
    hinge_binary_avg = (sparsities[4] + sparsities[5]) / 2
    hinge_ternary = ternary_sparsities[2]
    hinge_ratio = hinge_binary_avg / (hinge_ternary / 2)

    logger.info(f"Hinge lens:")
    logger.info(f"  Ternary active: {hinge_ternary:.2f}%")
    logger.info(f"  Binary avg active: {hinge_binary_avg:.2f}%")
    logger.info(f"  Expected: {hinge_ternary / 2:.2f}%")
    logger.info(f"  Ratio: {hinge_ratio:.2f}x")

    if abs(hinge_ratio - 1.0) > 0.2:
        logger.warning(f"  ⚠️  Sparsity ratio outside expected range")
        validation_passed = False
    else:
        logger.info(f"  ✓ Sparsity ratio correct")

    logger.info("")
    logger.info("="*80)

    if validation_passed:
        logger.info("✅ VALIDATION PASSED")
        logger.info("")
        logger.info("Split binary architecture is correct:")
        logger.info("  ✓ 6 banks with proper structure")
        logger.info("  ✓ Within-lens splitting preserved")
        logger.info("  ✓ Half sparsity per bank maintained")
        logger.info("  ✓ Ready for decoder validation testing")
    else:
        logger.error("❌ VALIDATION FAILED")
        logger.error("Split binary architecture has issues - review sparsity ratios")

    logger.info("="*80)

    return validation_passed


def main():
    """Validate split binary quantization"""
    binary_path = Path("genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_binary.h5")
    ternary_path = Path("genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5")

    if not binary_path.exists():
        logger.error(f"Binary file not found: {binary_path}")
        return 1

    if not ternary_path.exists():
        logger.error(f"Ternary file not found: {ternary_path}")
        return 1

    success = validate_split_binary(binary_path, ternary_path, sample_size=1000)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
