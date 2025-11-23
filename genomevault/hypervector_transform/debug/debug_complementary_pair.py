#!/usr/bin/env python3
"""
Debug Complementary Pair Encoding

Test with a simple known sequence to verify the architecture works.
"""

import numpy as np
from pathlib import Path

# Simple test: encode a known 2000bp sequence, then query it back
def test_simple_sequence():
    print("=" * 80)
    print("SIMPLE SEQUENCE TEST")
    print("=" * 80)
    print()

    # Parameters
    D = 10000
    N = 2000

    # Generate position codebook
    np.random.seed(42)
    codebook = np.random.choice([-1, 1], size=(N, D)).astype(np.int8)

    # Create a simple known sequence
    test_sequence = "ATGC" * 500  # 2000 nucleotides
    print(f"Test sequence: {test_sequence[:50]}... (length: {len(test_sequence)})")
    print()

    # Encode using Complementary Pair architecture
    AT_vec = np.zeros(D, dtype=np.float32)
    GC_vec = np.zeros(D, dtype=np.float32)

    for i, nucleotide in enumerate(test_sequence):
        pos_vec = codebook[i].astype(np.float32)

        if nucleotide == 'A':
            AT_vec += pos_vec
        elif nucleotide == 'T':
            AT_vec -= pos_vec
        elif nucleotide == 'G':
            GC_vec += pos_vec
        elif nucleotide == 'C':
            GC_vec -= pos_vec

    print(f"AT_vec stats: mean={AT_vec.mean():.4f}, std={AT_vec.std():.4f}, min={AT_vec.min():.4f}, max={AT_vec.max():.4f}")
    print(f"GC_vec stats: mean={GC_vec.mean():.4f}, std={GC_vec.std():.4f}, min={GC_vec.min():.4f}, max={GC_vec.max():.4f}")
    print()

    # Test retrieval
    print("Testing retrieval for first 20 positions:")
    print()
    print(f"{'Pos':<6} {'True':<6} {'Pred':<6} {'OK':<4} {'AT Sim':<10} {'GC Sim':<10} {'Pair':<6}")
    print("-" * 70)

    correct = 0
    total = 0

    for pos in range(20):
        true_nucleotide = test_sequence[pos]

        # Query
        pos_vec = codebook[pos].astype(np.float32)

        sim_AT = np.dot(pos_vec, AT_vec) / (np.linalg.norm(AT_vec) + 1e-10)
        sim_GC = np.dot(pos_vec, GC_vec) / (np.linalg.norm(GC_vec) + 1e-10)

        # Two-stage retrieval
        if abs(sim_AT) > abs(sim_GC):
            pair = 'AT'
            predicted_nucleotide = 'A' if sim_AT > 0 else 'T'
        else:
            pair = 'GC'
            predicted_nucleotide = 'G' if sim_GC > 0 else 'C'

        is_correct = (predicted_nucleotide == true_nucleotide)
        if is_correct:
            correct += 1
        total += 1

        sym = '✓' if is_correct else '✗'
        print(f"{pos:<6} {true_nucleotide:<6} {predicted_nucleotide:<6} {sym:<4} {sim_AT:<10.4f} {sim_GC:<10.4f} {pair:<6}")

    print()
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print()

    # Expected theoretical accuracy
    SNR = 2 * D / N
    print(f"SNR: {SNR:.2f}")
    print(f"Expected P(sign error): ~0.079%")
    print(f"Expected accuracy: ~99.92%")
    print()

    if accuracy >= 95:
        print("✅ TEST PASSED!")
    else:
        print("⚠️ TEST FAILED - accuracy too low")
        print()
        print("Debugging info:")
        print(f"  AT_vec norm: {np.linalg.norm(AT_vec):.4f}")
        print(f"  GC_vec norm: {np.linalg.norm(GC_vec):.4f}")
        print(f"  Expected signal strength: {np.sqrt(2 * D / N):.4f}")


if __name__ == "__main__":
    test_simple_sequence()
