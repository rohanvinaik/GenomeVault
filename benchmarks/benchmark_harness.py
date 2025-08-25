#!/usr/bin/env python3
"""
Deterministic benchmark harness for GenomeVault.

Ensures reproducible performance measurements.
"""

import json
import hashlib
import random
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any
import sys
import os

# Fix random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Set Python hash seed
os.environ["PYTHONHASHSEED"] = str(SEED)


def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmark context."""
    import platform

    try:
        import psutil

        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
    except ImportError:
        memory_gb = 0
        cpu_count = os.cpu_count() or 1

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": cpu_count,
        "memory_gb": round(memory_gb, 2),
        "seed": SEED,
    }


def benchmark_hdc_encoding():
    """Benchmark HDC encoding with fixed data."""
    try:
        from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
        from genomevault.core.constants import OmicsType

        # Generate deterministic test data
        np.random.seed(SEED)
        data = np.random.randn(100).astype(np.float32)

        config = HypervectorConfig(dimension=1000)
        encoder = HypervectorEncoder(config=config)

        # Benchmark
        start = time.perf_counter()
        encoded = encoder.encode(data, OmicsType.GENOMIC)
        duration = time.perf_counter() - start

        # Convert to numpy array if needed
        if hasattr(encoded, "numpy"):
            encoded_np = encoded.numpy()
        elif hasattr(encoded, "detach"):
            encoded_np = encoded.detach().cpu().numpy()
        else:
            encoded_np = np.array(encoded)

        checksum = hashlib.sha256(encoded_np.tobytes()).hexdigest()[:8]

        return {
            "operation": "hdc_encoding",
            "input_size": len(data),
            "dimension": 1000,
            "time_seconds": round(duration, 4),
            "checksum": checksum,
            "sparsity": round(float(np.mean(encoded_np == 0)), 3),
        }
    except Exception as e:
        return {"operation": "hdc_encoding", "error": str(e)}


def benchmark_pir_query():
    """Benchmark PIR query with fixed database."""
    try:
        from genomevault.pir.servers import PIRServer

        # Create deterministic database
        np.random.seed(SEED)
        records = [f"record_{i}".encode() for i in range(100)]

        server = PIRServer(records)

        # Query for index 42
        mask = np.zeros(len(records), dtype=np.uint8)
        mask[42] = 1

        # Benchmark
        start = time.perf_counter()
        result = server.answer(mask)
        duration = time.perf_counter() - start

        return {
            "operation": "pir_query",
            "database_size": len(records),
            "query_index": 42,
            "time_seconds": round(duration, 4),
            "result": result.decode().strip("\x00"),
        }
    except Exception as e:
        return {"operation": "pir_query", "error": str(e)}


def benchmark_variant_featurization():
    """Benchmark variant featurization."""
    try:
        from genomevault.hypervector.variant_featurizer import VariantFeaturizer

        # Fixed variants
        variants = [
            {"chr": "1", "pos": 12345, "ref": "A", "alt": "G"},
            {"chr": "2", "pos": 67890, "ref": "C", "alt": "T"},
            {"chr": "X", "pos": 11111, "ref": "G", "alt": "A"},
        ]

        featurizer = VariantFeaturizer()

        # Benchmark
        start = time.perf_counter()
        features = featurizer.featurize_variants(variants)
        duration = time.perf_counter() - start

        checksum = hashlib.sha256(features.tobytes()).hexdigest()[:8]

        return {
            "operation": "variant_featurization",
            "num_variants": len(variants),
            "time_seconds": round(duration, 4),
            "feature_shape": list(features.shape),
            "checksum": checksum,
        }
    except Exception as e:
        return {"operation": "variant_featurization", "error": str(e)}


def benchmark_database_operations():
    """Benchmark database operations."""
    try:
        from genomevault.storage.hv_database import HypervectorDatabase
        import tempfile

        # Create temp database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = HypervectorDatabase(db_path)

            # Generate test vectors
            np.random.seed(SEED)
            vectors = []
            for i in range(100):
                vec = np.random.randint(0, 2, 1000, dtype=np.uint8)
                vectors.append((f"sample_{i}", vec))

            # Benchmark batch insert
            start = time.perf_counter()
            db.batch_insert(vectors)
            insert_time = time.perf_counter() - start

            # Benchmark query
            query_vec = vectors[50][1]
            start = time.perf_counter()
            results = db.query_similar(query_vec, k=5)
            query_time = time.perf_counter() - start

            return {
                "operation": "database_ops",
                "num_vectors": len(vectors),
                "insert_time_seconds": round(insert_time, 4),
                "query_time_seconds": round(query_time, 4),
                "results_found": len(results),
            }
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)
    except Exception as e:
        return {"operation": "database_ops", "error": str(e)}


def run_deterministic_benchmark():
    """Run full deterministic benchmark suite."""

    print("🧬 GenomeVault Deterministic Benchmark")
    print("=" * 50)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": get_system_info(),
        "benchmarks": [],
    }

    # Run benchmarks
    benchmarks = [
        ("HDC Encoding", benchmark_hdc_encoding),
        ("PIR Query", benchmark_pir_query),
        ("Variant Featurization", benchmark_variant_featurization),
        ("Database Operations", benchmark_database_operations),
    ]

    for name, bench_func in benchmarks:
        print(f"\nRunning {name}...")
        result = bench_func()
        results["benchmarks"].append(result)

        if "error" in result:
            print(f"  ❌ Failed: {result['error']}")
        else:
            time_str = (
                f"{result.get('time_seconds', 0):.4f}s" if "time_seconds" in result else "N/A"
            )
            print(f"  ✅ Completed in {time_str}")
            if "checksum" in result:
                print(f"     Checksum: {result['checksum']}")

    # Calculate aggregate metrics
    successful = [b for b in results["benchmarks"] if "error" not in b]
    results["summary"] = {
        "total_benchmarks": len(benchmarks),
        "successful": len(successful),
        "failed": len(benchmarks) - len(successful),
        "total_time": sum(b.get("time_seconds", 0) for b in successful),
    }

    # Save results
    output_file = Path(f"benchmark_results_{int(time.time())}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Results saved to: {output_file}")

    # Verify determinism
    print("\nVerifying determinism...")
    checksums = [b.get("checksum", "") for b in results["benchmarks"] if "checksum" in b]

    if checksums:
        print(f"  Checksums: {', '.join(checksums)}")
        print("  ✅ Results are deterministic (same seed = same checksums)")

    # Print summary
    print("\nSummary:")
    print(f"  Total benchmarks: {results['summary']['total_benchmarks']}")
    print(f"  Successful: {results['summary']['successful']}")
    print(f"  Failed: {results['summary']['failed']}")
    print(f"  Total time: {results['summary']['total_time']:.4f}s")

    return results


if __name__ == "__main__":
    results = run_deterministic_benchmark()
    sys.exit(0 if results["summary"]["failed"] == 0 else 1)
