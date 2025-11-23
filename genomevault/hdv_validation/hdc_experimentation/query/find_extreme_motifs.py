"""
Find Extreme Structural Motifs

Search for chunks with LUDICROUS biophysical signatures - the kind that ONLY
occur in real structural elements (CpG islands, ALU repeats, poly-A tails).

Document BOTH magnitude AND sign patterns across all 3 banks.

Saves results to JSON for later analysis with identify_extreme_motifs.py.
"""

import h5py
import numpy as np
import json
from pathlib import Path

def analyze_extreme_chunk(banks: dict, chunk_idx: int, chunk_bp: int):
    """Analyze a chunk with extreme biophysical properties."""

    print(f"\nChunk {chunk_idx:,} ({chunk_bp:,} bp):")
    print("="*80)

    for bank_name in ['bank1', 'bank2', 'bank3']:
        bank = banks[bank_name]

        # Magnitude analysis
        magnitude = np.linalg.norm(bank)
        zeros = np.sum(bank == 0)
        density = 1 - (zeros / bank.size)

        # Sign analysis
        positives = np.sum(bank > 0)
        negatives = np.sum(bank < 0)
        pos_ratio = positives / bank.size
        neg_ratio = negatives / bank.size

        # Value statistics
        pos_values = bank[bank > 0]
        neg_values = bank[bank < 0]

        pos_mean = np.mean(pos_values) if len(pos_values) > 0 else 0
        neg_mean = np.mean(neg_values) if len(neg_values) > 0 else 0

        print(f"  {bank_name}:")
        print(f"    Magnitude: {magnitude:.2f}")
        print(f"    Density:   {density:.1%} ({zeros:,} zeros)")
        print(f"    Sign dist: {pos_ratio:.1%} pos, {neg_ratio:.1%} neg")
        print(f"    Pos mean:  {pos_mean:.2f} ({len(pos_values):,} values)")
        print(f"    Neg mean:  {neg_mean:.2f} ({len(neg_values):,} values)")


def compute_chunk_metrics(banks: dict) -> dict:
    """
    Compute full metrics for a chunk for JSON export.

    Returns dict with:
        - bank magnitudes (bank1_mag, bank2_mag, bank3_mag)
        - density (overall sparsity)
        - ratio (bank2/bank1)
    """
    bank1_mag = float(np.linalg.norm(banks['bank1']))
    bank2_mag = float(np.linalg.norm(banks['bank2']))
    bank3_mag = float(np.linalg.norm(banks['bank3']))

    # Compute overall density
    total_zeros = sum(np.sum(bank == 0) for bank in banks.values())
    total_elements = sum(bank.size for bank in banks.values())
    density = 1 - (total_zeros / total_elements)

    # Compute ratio
    ratio = bank2_mag / bank1_mag if bank1_mag > 0 else 0.0

    return {
        'bank1_mag': bank1_mag,
        'bank2_mag': bank2_mag,
        'bank3_mag': bank3_mag,
        'density': float(density),
        'ratio': ratio,
    }


h5_path = "genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5"

print("="*80)
print("FINDING EXTREME STRUCTURAL MOTIFS")
print("="*80)
print()
print("Strategy: Look for chunks where bank1/bank2 magnitude ratio is EXTREME")
print("(either >>1 for AT-rich or <<1 for GC-rich)")
print()

with h5py.File(h5_path, 'r') as f:
    all_banks = f['all_bank_vectors']
    total_chunks = all_banks.shape[0]

    # Find chunks with extreme bank1/bank2 ratios
    extreme_gc_rich = []  # bank2 >> bank1 (CpG islands, ALU)
    extreme_at_rich = []  # bank1 >> bank2 (poly-A tails)

    print(f"Scanning {total_chunks:,} chunks for extreme patterns...")
    print()

    sample_size = 10000  # Sample 10k chunks for speed
    np.random.seed(42)
    sample_indices = np.random.choice(total_chunks, sample_size, replace=False)

    for chunk_idx in sample_indices:
        all_banks_data = all_banks[chunk_idx, :, :]

        bank1_mag = np.linalg.norm(all_banks_data[0, :])
        bank2_mag = np.linalg.norm(all_banks_data[1, :])

        if bank1_mag > 0 and bank2_mag > 0:
            ratio = bank2_mag / bank1_mag

            # Extreme GC-rich: bank2/bank1 ratio > 1.05
            if ratio > 1.05:
                extreme_gc_rich.append((chunk_idx, ratio, bank1_mag, bank2_mag))

            # Extreme AT-rich: bank2/bank1 ratio < 0.95
            elif ratio < 0.95:
                extreme_at_rich.append((chunk_idx, ratio, bank1_mag, bank2_mag))

    # Sort by extremeness
    extreme_gc_rich.sort(key=lambda x: x[1], reverse=True)
    extreme_at_rich.sort(key=lambda x: x[1])

    print(f"Found {len(extreme_gc_rich):,} GC-rich chunks (bank2/bank1 > 1.05)")
    print(f"Found {len(extreme_at_rich):,} AT-rich chunks (bank2/bank1 < 0.95)")
    print()

    # Analyze top 3 most extreme examples of each type
    print("="*80)
    print("TOP 3 EXTREME GC-RICH CHUNKS (CpG islands / ALU repeats)")
    print("="*80)

    for i, (chunk_idx, ratio, bank1_mag, bank2_mag) in enumerate(extreme_gc_rich[:3]):
        all_banks_data = all_banks[chunk_idx, :, :]
        banks = {
            'bank1': all_banks_data[0, :],
            'bank2': all_banks_data[1, :],
            'bank3': all_banks_data[2, :],
        }

        chunk_bp = chunk_idx * 896
        print(f"\n{'='*80}")
        print(f"GC-RICH #{i+1}: bank2/bank1 ratio = {ratio:.4f}")
        print(f"  bank1 (AT): {bank1_mag:.2f}, bank2 (GC): {bank2_mag:.2f}")
        analyze_extreme_chunk(banks, chunk_idx, chunk_bp)

    print()
    print("="*80)
    print("TOP 3 EXTREME AT-RICH CHUNKS (poly-A tails / AT-repeat regions)")
    print("="*80)

    for i, (chunk_idx, ratio, bank1_mag, bank2_mag) in enumerate(extreme_at_rich[:3]):
        all_banks_data = all_banks[chunk_idx, :, :]
        banks = {
            'bank1': all_banks_data[0, :],
            'bank2': all_banks_data[1, :],
            'bank3': all_banks_data[2, :],
        }

        chunk_bp = chunk_idx * 896
        print(f"\n{'='*80}")
        print(f"AT-RICH #{i+1}: bank2/bank1 ratio = {ratio:.4f}")
        print(f"  bank1 (AT): {bank1_mag:.2f}, bank2 (GC): {bank2_mag:.2f}")
        analyze_extreme_chunk(banks, chunk_idx, chunk_bp)

    print()
    print("="*80)
    print("RECOMMENDED THRESHOLDS (INTERSECTION OF ALL CRITERIA)")
    print("="*80)
    print()

    if extreme_gc_rich:
        top_gc = extreme_gc_rich[0]
        print(f"GC-RICH MOTIFS (CpG islands, ALU):")
        print(f"  - bank2/bank1 ratio > {top_gc[1]:.4f}  (ludicrously GC-heavy)")
        print(f"  - bank2 magnitude > {top_gc[3]:.2f}")
        print(f"  - bank1 magnitude < {top_gc[2]:.2f}")
        print()

    if extreme_at_rich:
        top_at = extreme_at_rich[0]
        print(f"AT-RICH MOTIFS (poly-A tails, AT repeats):")
        print(f"  - bank2/bank1 ratio < {top_at[1]:.4f}  (ludicrously AT-heavy)")
        print(f"  - bank1 magnitude > {top_at[2]:.2f}")
        print(f"  - bank2 magnitude < {top_at[3]:.2f}")

    print()
    print("="*80)

    # Export top n=50 chunks to JSON for later analysis
    print("\n" + "="*80)
    print("EXPORTING TOP n=50 CHUNKS TO JSON")
    print("="*80)

    n_export = 50
    export_data = {
        'GC_RICH': {
            'description': 'Top 50 most GC-rich chunks (bank2/bank1 ratio highest)',
            'chunk_indices': [],
            'metrics': [],
        },
        'AT_RICH': {
            'description': 'Top 50 most AT-rich chunks (bank2/bank1 ratio lowest)',
            'chunk_indices': [],
            'metrics': [],
        },
        'BALANCED': {
            'description': 'Middle 50 chunks (bank2/bank1 ratio near 1.0)',
            'chunk_indices': [],
            'metrics': [],
        },
    }

    # Export top n=50 GC-rich chunks
    for chunk_idx, ratio, bank1_mag, bank2_mag in extreme_gc_rich[:n_export]:
        all_banks_data = all_banks[chunk_idx, :, :]
        banks = {
            'bank1': all_banks_data[0, :],
            'bank2': all_banks_data[1, :],
            'bank3': all_banks_data[2, :],
        }
        metrics = compute_chunk_metrics(banks)
        export_data['GC_RICH']['chunk_indices'].append(int(chunk_idx))
        export_data['GC_RICH']['metrics'].append(metrics)

    # Export top n=50 AT-rich chunks
    for chunk_idx, ratio, bank1_mag, bank2_mag in extreme_at_rich[:n_export]:
        all_banks_data = all_banks[chunk_idx, :, :]
        banks = {
            'bank1': all_banks_data[0, :],
            'bank2': all_banks_data[1, :],
            'bank3': all_banks_data[2, :],
        }
        metrics = compute_chunk_metrics(banks)
        export_data['AT_RICH']['chunk_indices'].append(int(chunk_idx))
        export_data['AT_RICH']['metrics'].append(metrics)

    # Find balanced chunks (ratio near 1.0)
    all_chunks_with_ratios = []
    for chunk_idx in sample_indices:
        all_banks_data = all_banks[chunk_idx, :, :]
        bank1_mag = np.linalg.norm(all_banks_data[0, :])
        bank2_mag = np.linalg.norm(all_banks_data[1, :])
        if bank1_mag > 0 and bank2_mag > 0:
            ratio = bank2_mag / bank1_mag
            # Only keep chunks near ratio=1.0 (within 0.98-1.02)
            if 0.98 <= ratio <= 1.02:
                all_chunks_with_ratios.append((chunk_idx, ratio, abs(ratio - 1.0)))

    # Sort by closeness to 1.0
    all_chunks_with_ratios.sort(key=lambda x: x[2])

    # Export top n=50 balanced chunks
    for chunk_idx, ratio, deviation in all_chunks_with_ratios[:n_export]:
        all_banks_data = all_banks[chunk_idx, :, :]
        banks = {
            'bank1': all_banks_data[0, :],
            'bank2': all_banks_data[1, :],
            'bank3': all_banks_data[2, :],
        }
        metrics = compute_chunk_metrics(banks)
        export_data['BALANCED']['chunk_indices'].append(int(chunk_idx))
        export_data['BALANCED']['metrics'].append(metrics)

    # Save to JSON
    output_file = "/tmp/extreme_motifs_n50.json"
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\nExported {n_export} chunks per category to {output_file}")
    print(f"  - GC_RICH: {len(export_data['GC_RICH']['chunk_indices'])} chunks")
    print(f"  - AT_RICH: {len(export_data['AT_RICH']['chunk_indices'])} chunks")
    print(f"  - BALANCED: {len(export_data['BALANCED']['chunk_indices'])} chunks")
    print()
    print("="*80)
