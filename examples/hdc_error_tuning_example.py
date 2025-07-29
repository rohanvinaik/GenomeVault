"""
Example usage of HDC error handling with uncertainty tuning
"""

import asyncio

import numpy as np

from genomevault.hypervector.error_handling import (
    AdaptiveHDCEncoder,
    ErrorBudget,
    ErrorBudgetAllocator,
)


async def main():
    """Demonstrate HDC error handling capabilities"""
    # 1. Basic error budget allocation
    print("=== Error Budget Allocation ===")
    allocator = ErrorBudgetAllocator()

    # Different accuracy/confidence settings
    settings = [
        ("Fast", 0.02, 10),  # 2% error, 1 in 1K confidence
        ("Balanced", 0.01, 15),  # 1% error, 1 in 32K confidence
        ("Clinical", 0.001, 25),  # 0.1% error, 1 in 33M confidence
    ]

    for name, epsilon, delta_exp in settings:
        budget = allocator.plan_budget(epsilon, delta_exp)
        latency = allocator.estimate_latency(budget)
        bandwidth = allocator.estimate_bandwidth(budget)

        print(f"\n{name} Setting:")
        print(f"  Dimension: {budget.dimension:,}")
        print(f"  Repeats: {budget.repeats}")
        print(f"  Estimated latency: {latency}ms")
        print(f"  Estimated bandwidth: {bandwidth:.1f}MB")

    # 2. Encoding with error handling
    print("\n=== Adaptive Encoding ===")

    # Sample genomic variants
    variants = [
        {
            "chromosome": "chr1",
            "position": 762273,
            "ref": "G",
            "alt": "A",
            "type": "SNP",
        },
        {
            "chromosome": "chr2",
            "position": 1234567,
            "ref": "T",
            "alt": "C",
            "type": "SNP",
        },
        {
            "chromosome": "chr7",
            "position": 5566778,
            "ref": "A",
            "alt": "G",
            "type": "SNP",
        },
    ]

    # Create encoder and budget
    encoder = AdaptiveHDCEncoder(dimension=10000)
    budget = ErrorBudget(
        dimension=10000,
        parity_g=3,
        repeats=10,
        epsilon=0.01,
        delta_exp=15,
        ecc_enabled=True,
    )

    # Encode with budget
    encoded_vector, metadata = encoder.encode_with_budget(variants, budget)

    print(f"Encoded vector shape: {encoded_vector.shape}")
    print(f"Median error: {metadata['median_error']:.6f}")
    print(f"Error within bound: {metadata['error_within_bound']}")
    print(f"Number of proofs: {len(metadata['proofs'])}")

    # 3. Compare with and without ECC
    print("\n=== ECC Impact Comparison ===")

    budget_no_ecc = ErrorBudget(
        dimension=10000,
        parity_g=0,
        repeats=10,
        epsilon=0.01,
        delta_exp=15,
        ecc_enabled=False,
    )

    budget_with_ecc = ErrorBudget(
        dimension=10000,
        parity_g=3,
        repeats=10,
        epsilon=0.01,
        delta_exp=15,
        ecc_enabled=True,
    )

    _, meta_no_ecc = encoder.encode_with_budget(variants, budget_no_ecc)
    _, meta_with_ecc = encoder.encode_with_budget(variants, budget_with_ecc)

    print(f"Without ECC - Median error: {meta_no_ecc['median_error']:.6f}")
    print(f"With ECC    - Median error: {meta_with_ecc['median_error']:.6f}")
    print(
        f"Error reduction: {(1 - meta_with_ecc['median_error']/meta_no_ecc['median_error'])*100:.1f}%"
    )

    # 4. API usage example
    print("\n=== API Usage Example ===")
    print("POST /api/hdc/estimate_budget")
    print("Body: {")
    print('  "epsilon": 0.005,')
    print('  "delta_exp": 20,')
    print('  "ecc_enabled": true')
    print("}")
    print("\nExpected Response: {")
    print('  "dimension": 120000,')
    print('  "repeats": 27,')
    print('  "estimated_latency_ms": 1400,')
    print('  "estimated_bandwidth_mb": 5.8')
    print("}")

    # 5. Preset configurations
    print("\n=== Preset Configurations ===")
    from genomevault.core.constants import HDC_ERROR_CONFIG

    for preset_name, preset_config in HDC_ERROR_CONFIG["presets"].items():
        budget = allocator.plan_budget(
            epsilon=preset_config["epsilon"],
            delta_exp=preset_config["delta_exp"],
            ecc_enabled=preset_config["ecc"],
        )
        print(f"\n{preset_name}: {preset_config.get('description', '')}")
        print(
            f"  Settings: {preset_config['epsilon']*100}% error, 1 in {2**preset_config['delta_exp']:,} confidence"
        )
        print(f"  Dimension: {budget.dimension:,}, Repeats: {budget.repeats}")


if __name__ == "__main__":
    asyncio.run(main())
