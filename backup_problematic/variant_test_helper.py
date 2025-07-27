from typing import Any, Dict

#!/usr/bin/env python3
"""
GenomeVault Variant Testing Helper
Shows how to run experiments with specific genomic variants
"""

import argparse
import sys
from pathlib import Path


def print_examples() -> None:
       """TODO: Add docstring for print_examples"""
     """Print example commands for testing specific variants"""

    print("\n🧬 GenomeVault Variant Testing Examples")
    print("=" * 60)

    print("\n1. ENCODE A SPECIFIC VARIANT:")
    print("-" * 30)
    print("# Encode a single SNP variant")
    print("python experiments/test_encoding.py --variant chr1:123456:A:G")
    print("\n# Encode with higher dimension for better precision")
    print("python experiments/test_encoding.py --variant chr1:123456:A:G --dimension 50000")
    print("\n# Test privacy preservation")
    print("python experiments/test_encoding.py --variant chr1:123456:A:G --test-privacy")

    print("\n\n2. SEARCH FOR SIMILAR VARIANTS:")
    print("-" * 30)
    print("# Search for variants similar to your query")
    print("python experiments/variant_search.py --query chr1:123456:A:G --database variants.txt")
    print("\n# Search with higher precision (more dimensions)")
    print("python experiments/variant_search.py --query chr1:123456:A:G --dimension 50000")
    print("\n# Get more results")
    print("python experiments/variant_search.py --query chr1:123456:A:G --top-k 20")

    print("\n\n3. COMMON DISEASE VARIANTS:")
    print("-" * 30)
    print("# BRCA1 variant (breast cancer)")
    print("python experiments/test_encoding.py --variant chr17:41276045:C:T")
    print("\n# APOE4 variant (Alzheimer's)")
    print("python experiments/test_encoding.py --variant chr19:45411941:T:C")
    print("\n# Factor V Leiden (blood clotting)")
    print("python experiments/test_encoding.py --variant chr1:169519049:G:A")

    print("\n\n4. BATCH PROCESSING:")
    print("-" * 30)
    print("# Create a variants file (variants.txt):")
    print("cat > variants.txt << EOF")
    print("chr1\t123456\tA\tG")
    print("chr2\t234567\tC\tT")
    print("chr3\t345678\tG\tA")
    print("EOF")
    print("\n# Then search against it")
    print("python experiments/variant_search.py --query chr1:100000:A:G --database variants.txt")

    print("\n\n5. PERFORMANCE BENCHMARKS:")
    print("-" * 30)
    print("# Benchmark HD encoding performance")
    print("python benchmarks/benchmark_packed_hypervector.py")
    print("\n# Run full HDC benchmarks")
    print("python scripts/bench_hdc.py")

    print("\n\n6. ADVANCED KAN-HD EXPERIMENTS:")
    print("-" * 30)
    print("# Run comprehensive KAN-HD demo")
    print("python examples/kan_hd_enhanced_demo.py")
    print("\n# Test KAN hybrid encoding")
    print("python examples/kan_hybrid_example.py")

    print("\n\n7. PRIVACY-PRESERVING WORKFLOWS:")
    print("-" * 30)
    print("# Test HD + PIR integration")
    print("python examples/hdc_pir_integration_demo.py")
    print("\n# Full privacy stack (HD + PIR + ZK)")
    print("python examples/hdc_pir_zk_integration_demo.py")

    print("\n\n💡 TIPS:")
    print("-" * 30)
    print("• Variant format: chr:position:ref:alt (e.g., chr1:123456:A:G)")
    print("• Higher dimensions (50000-100000) = better accuracy but slower")
    print("• Use --test-privacy to verify encoding security")
    print("• Check logs in genomevault/.benchmarks/ for detailed metrics")
    print("\n")


def create_test_variant_file(filename="test_variants.txt", n_variants=10) -> Dict[str, Any]:
       """TODO: Add docstring for create_test_variant_file"""
     """Create a test file with random variants"""
    import random

    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    bases = ["A", "T", "G", "C"]

    with open(filename, "w") as f:
        for _ in range(n_variants):
            chrom = random.choice(chromosomes)
            pos = random.randint(100000, 200000000)
            ref = random.choice(bases)
            alt = random.choice([b for b in bases if b != ref])
            f.write(f"{chrom}\t{pos}\t{ref}\t{alt}\n")

    print(f"Created {filename} with {n_variants} test variants")


def main() -> None:
        """TODO: Add docstring for main"""
    parser = argparse.ArgumentParser(
        description="GenomeVault variant testing helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--create-test-file", action="store_true", help="Create a test variants file"
    )

    parser.add_argument(
        "--n-variants", type=int, default=10, help="Number of test variants to create (default: 10)"
    )

    args = parser.parse_args()

    if args.create_test_file:
        create_test_variant_file(n_variants=args.n_variants)
    else:
        print_examples()


if __name__ == "__main__":
    main()
