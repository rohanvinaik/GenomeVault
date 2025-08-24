#!/usr/bin/env python3
"""
Comprehensive ZK proof pipeline test with performance analysis.

Tests all optimizations and identifies bottlenecks:
- Batch constraint generation
- Adaptive circuit selection
- Witness caching
- Parallel proof generation
- Memory pooling
- GPU acceleration
- Performance monitoring
"""

import time
import sys
import hashlib
import numpy as np
import psutil
import traceback

# Add genomevault to path
sys.path.insert(0, "/Users/rohanvinaik/genomevault")

from genomevault.zk_proofs.prover import Prover
from genomevault.zk_proofs.circuits.optimized.diabetes_risk_alert import (
    OptimizedDiabetesRiskCircuit,
)
from genomevault.zk_proofs.circuits.adaptive_variant import (
    AdaptiveVariantPresenceCircuit as AdaptiveVariantCircuit,
)
from genomevault.zk_proofs.witness_cache import get_witness_cache
from genomevault.zk_proofs.parallel_prover import ParallelProver, ProofTask
from genomevault.zk_proofs.memory_pool import get_memory_manager, MemoryEfficientProver
from genomevault.zk_proofs.gpu_prover import GPUProver, get_gpu_prover
from genomevault.zk_proofs.performance_monitor import get_monitor
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class PipelineProfiler:
    """Profile each component of the ZK pipeline."""

    def __init__(self):
        self.results = {}
        self.bottlenecks = []

    def profile_section(self, name: str, func, *args, **kwargs):
        """Profile a section of code."""
        print(f"\n🔍 Profiling: {name}")
        print("-" * 40)

        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # CPU before
        cpu_before = process.cpu_percent()

        # Run function
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            traceback.print_exc()

        elapsed = time.perf_counter() - start

        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_delta = mem_after - mem_before

        # CPU after
        cpu_after = process.cpu_percent()

        # Store results
        self.results[name] = {
            "time_ms": elapsed * 1000,
            "memory_delta_mb": mem_delta,
            "cpu_percent": cpu_after,
            "success": success,
            "error": error,
        }

        # Print results
        if success:
            print(f"  ✅ Time: {elapsed*1000:.2f}ms")
        else:
            print(f"  ❌ Failed: {error}")
        print(f"  💾 Memory: {mem_delta:+.1f}MB (total: {mem_after:.1f}MB)")
        print(f"  🔥 CPU: {cpu_after:.1f}%")

        # Identify bottlenecks
        if elapsed > 0.1:  # >100ms is slow
            self.bottlenecks.append((name, elapsed * 1000))

        return result

    def print_summary(self):
        """Print profiling summary."""
        print("\n" + "=" * 60)
        print("PROFILING SUMMARY")
        print("=" * 60)

        # Sort by time
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]["time_ms"], reverse=True)

        print("\n⏱️  Performance Breakdown:")
        print("Component                          | Time (ms) | Memory | Status")
        print("-----------------------------------|-----------|--------|--------")

        total_time = sum(r["time_ms"] for r in self.results.values())

        for name, stats in sorted_results:
            pct = (stats["time_ms"] / total_time * 100) if total_time > 0 else 0
            status = "✅" if stats["success"] else "❌"
            print(
                f"{name:34s} | {stats['time_ms']:9.2f} | {stats['memory_delta_mb']:+6.1f} | {status} {pct:5.1f}%"
            )

        print(f"{'TOTAL':34s} | {total_time:9.2f} |        |")

        if self.bottlenecks:
            print("\n🚨 Bottlenecks (>100ms):")
            for name, time_ms in sorted(self.bottlenecks, key=lambda x: x[1], reverse=True):
                print(f"  - {name}: {time_ms:.2f}ms")

        return total_time


def test_1_basic_proof_generation():
    """Test basic proof generation without optimizations."""
    prover = Prover(use_circom=False)

    # Variant presence proof
    public_inputs = {
        "variant_hash": hashlib.sha256(b"chr1:12345:A:G").hexdigest(),
        "reference_hash": "ref_" + hashlib.sha256(b"hg38").hexdigest()[:8],
        "commitment_root": "root_" + hashlib.sha256(b"merkle").hexdigest()[:8],
    }

    private_inputs = {
        "variant_data": {"chr": "chr1", "pos": 12345, "ref": "A", "alt": "G"},
        "merkle_proof": ["proof1", "proof2"],
        "witness_randomness": hashlib.sha256(b"random").hexdigest(),
    }

    proof = prover.generate_proof("variant_presence", public_inputs, private_inputs)
    return proof


def test_2_batch_constraint_generation():
    """Test optimized batch constraint generation."""
    circuit = OptimizedDiabetesRiskCircuit()

    # Generate constraints in batch
    risk_factors = np.random.randn(10).tolist()
    constraints = circuit.generate_constraint_batch(1000, risk_factors)

    print(f"  Generated {len(constraints.constraints)} constraints")
    print(f"  Batch ID: {constraints.batch_id}")

    # Test cache hit
    constraints2 = circuit.generate_constraint_batch(1000, risk_factors)
    print(f"  Cache hit: {constraints.batch_id == constraints2.batch_id}")

    return constraints


def test_3_adaptive_circuit_selection():
    """Test adaptive circuit selection based on input size."""
    circuit = AdaptiveVariantCircuit()

    # Small input (should use small circuit)
    small_variants = [{"chr": f"chr{i}", "pos": i * 1000, "alt": "G"} for i in range(10)]

    selected = circuit.select_circuit(len(small_variants))
    print(f"  Small input ({len(small_variants)} variants): {selected.__class__.__name__}")

    # Large input (should use large circuit)
    large_variants = [{"chr": f"chr{i%22+1}", "pos": i * 1000, "alt": "G"} for i in range(100)]

    selected = circuit.select_circuit(len(large_variants))
    print(f"  Large input ({len(large_variants)} variants): {selected.__class__.__name__}")

    # Process both
    query = {"chr": "chr1", "pos": 5000, "alt": "G"}

    result1 = circuit.process(small_variants, query)
    result2 = circuit.process(large_variants, query)

    print(f"  Small processing time: {result1.get('processing_time_ms', 0):.2f}ms")
    print(f"  Large processing time: {result2.get('processing_time_ms', 0):.2f}ms")

    return result1, result2


def test_4_witness_caching():
    """Test witness generation caching."""
    cache = get_witness_cache()
    cache.clear()  # Start fresh

    prover = Prover(use_circom=False)

    # First generation (cache miss)
    public_inputs = {"threshold": 0.5}
    private_inputs = {"value": 0.75, "witness_randomness": "test"}

    start = time.perf_counter()
    proof1 = prover.generate_proof("diabetes_risk_alert", public_inputs, private_inputs)
    time1 = (time.perf_counter() - start) * 1000

    # Second generation (cache hit)
    start = time.perf_counter()
    proof2 = prover.generate_proof("diabetes_risk_alert", public_inputs, private_inputs)
    time2 = (time.perf_counter() - start) * 1000

    print(f"  First generation: {time1:.2f}ms")
    print(f"  Second generation (cached): {time2:.2f}ms")
    print(f"  Speedup: {time1/time2:.1f}x")

    # Check cache stats
    stats = cache.get_stats()
    print(
        f"  Cache stats: {stats['total_entries']} entries, "
        f"{stats['cache_hits']} hits, {stats['cache_misses']} misses"
    )

    return stats


def test_5_parallel_proof_generation():
    """Test parallel proof generation."""
    # Create tasks
    tasks = []
    for i in range(20):
        variant_hash = hashlib.sha256(f"variant_{i}".encode()).hexdigest()

        task = ProofTask(
            task_id=f"task_{i}",
            circuit_name="variant_presence",
            public_inputs={
                "variant_hash": variant_hash,
                "reference_hash": "ref_hash",
                "commitment_root": "root_hash",
            },
            private_inputs={
                "variant_data": {"chr": f"chr{i%22+1}", "pos": i * 1000, "ref": "A", "alt": "G"},
                "merkle_proof": ["p1", "p2"],
                "witness_randomness": f"random_{i}",
            },
            priority=i % 3,
        )
        tasks.append(task)

    # Sequential baseline
    prover = Prover(use_circom=False)
    start = time.perf_counter()
    for task in tasks[:5]:
        prover.generate_proof(task.circuit_name, task.public_inputs, task.private_inputs)
    seq_time = time.perf_counter() - start

    # Parallel execution
    parallel_prover = ParallelProver(max_workers=4)
    start = time.perf_counter()
    results = parallel_prover.generate_witness_batch(tasks[:5])
    par_time = time.perf_counter() - start

    parallel_prover.shutdown()

    print(f"  Sequential (5 proofs): {seq_time*1000:.2f}ms")
    print(f"  Parallel (5 proofs): {par_time*1000:.2f}ms")
    print(f"  Speedup: {seq_time/par_time:.1f}x")

    successful = sum(1 for _, _, error in results if error is None)
    print(f"  Success rate: {successful}/{len(results)}")

    return results


def test_6_memory_pool():
    """Test memory pool pre-allocation."""
    manager = get_memory_manager()

    # Allocate workspace
    workspace = manager.allocate_workspace(
        "variant_presence",
        ["constraint_generation", "witness_computation", "polynomial_evaluation"],
    )

    print(f"  Allocated {len(workspace)} buffers")

    # Simulate usage
    for op, buffer in workspace.items():
        # Simulate computation
        buffer[:100] = np.random.randn(100)
        print(f"  {op}: {len(buffer)} bytes")

    # Release workspace
    manager.release_workspace("variant_presence", workspace)

    # Get stats
    stats = manager.get_global_stats()
    print(f"  Total memory allocated: {stats['total_allocated_mb']:.2f}MB")
    print(f"  Total buffers: {stats['total_buffers']}")

    # Test memory-efficient prover
    mem_prover = MemoryEfficientProver()

    witness = mem_prover.generate_witness_with_pool(
        "variant_presence", {"variant_hash": "test"}, {"variant_data": {"chr": "1", "pos": 12345}}
    )

    if hasattr(witness, "metadata") and witness.metadata.get("memory_pool_used"):
        print("  ✅ Memory pool used for witness generation")

    return stats


def test_7_gpu_acceleration():
    """Test GPU acceleration."""
    gpu_prover = get_gpu_prover()

    if gpu_prover:
        print(f"  GPU detected: {gpu_prover.device}")

        # Get device info
        info = gpu_prover.get_device_info()
        print(f"  Backend: {info.get('backend', 'unknown')}")
        print(f"  Device: {info.get('device', 'unknown')}")

        # Test witness generation
        variants = [{"chr": f"chr{i}", "pos": i * 1000, "alt": "G"} for i in range(100)]
        query = {"chr": "chr1", "pos": 1000, "alt": "G"}

        witness = gpu_prover.accelerate_witness_generation(
            "variant_presence", {"variants": variants, "query": query}, constraint_count=15000
        )

        print(f"  Computation device: {witness.get('computation_device', 'unknown')}")
        if "gpu_time_ms" in witness:
            print(f"  GPU time: {witness['gpu_time_ms']:.2f}ms")

        # Test circuit optimization
        settings = gpu_prover.optimize_for_circuit("variant_presence", 20000)
        print(
            f"  Optimized settings: batch_size={settings['batch_size']}, "
            f"precision={settings['precision']}"
        )

        return witness
    else:
        print("  No GPU available - skipping GPU tests")
        return None


def test_8_performance_monitoring():
    """Test performance monitoring integration."""
    monitor = get_monitor()

    # Clear previous data
    monitor.metrics.clear()
    monitor.circuit_stats.clear()

    # Generate some proofs with monitoring
    prover = Prover(use_circom=False)

    circuits = ["variant_presence", "diabetes_risk_alert", "polygenic_risk_score"]

    for circuit in circuits:
        for i in range(5):
            try:
                if circuit == "variant_presence":
                    public = {
                        "variant_hash": hashlib.sha256(f"v{i}".encode()).hexdigest(),
                        "reference_hash": "ref",
                        "commitment_root": "root",
                    }
                    private = {
                        "variant_data": {"chr": "1", "pos": i},
                        "merkle_proof": ["p1"],
                        "witness_randomness": f"r{i}",
                    }
                else:
                    public = {"threshold": 0.5}
                    private = {"value": 0.7, "witness_randomness": f"r{i}"}

                prover.generate_proof(circuit, public, private)

            except Exception as e:
                print(f"  Warning: {circuit} failed - {e}")

    # Get dashboard data
    data = monitor.get_dashboard_data()

    print(f"  Total operations: {data['summary']['total_operations']}")
    print(f"  Success rate: {data['summary']['success_rate']:.1%}")
    print(f"  Cache hit rate: {data['summary']['overall_cache_hit_rate']:.1%}")

    # Check for alerts
    if data["alerts"]:
        print(f"  ⚠️  {len(data['alerts'])} alerts triggered")

    return data


def test_9_end_to_end_pipeline():
    """Test complete end-to-end pipeline with all optimizations."""
    print("\n🔄 Testing full pipeline with all optimizations...")

    # Setup
    gpu_prover = GPUProver() if get_gpu_prover() else None
    parallel_prover = ParallelProver(max_workers=4)
    monitor = get_monitor()

    # Create batch of tasks
    tasks = []
    for i in range(10):
        task = ProofTask(
            task_id=f"e2e_{i}",
            circuit_name="variant_presence" if i % 2 == 0 else "diabetes_risk_alert",
            public_inputs=(
                {
                    "variant_hash": hashlib.sha256(f"e2e_{i}".encode()).hexdigest(),
                    "reference_hash": "ref",
                    "commitment_root": "root",
                }
                if i % 2 == 0
                else {
                    "glucose_threshold": 126,
                    "risk_threshold": 0.75,
                    "result_commitment": "commit",
                }
            ),
            private_inputs=(
                {
                    "variant_data": {"chr": f"{i%22+1}", "pos": i * 1000},
                    "merkle_proof": ["p1"],
                    "witness_randomness": f"r_{i}",
                }
                if i % 2 == 0
                else {
                    "glucose_reading": 130 + i,
                    "risk_score": 0.8 + i * 0.01,
                    "witness_randomness": f"r_{i}",
                }
            ),
            priority=i % 3,
        )
        tasks.append(task)

    # Run pipeline
    start = time.perf_counter()

    # 1. Parallel generation with all optimizations
    results = parallel_prover.generate_witness_batch(tasks)

    # 2. Process results
    successful = sum(1 for _, _, error in results if error is None)

    total_time = (time.perf_counter() - start) * 1000

    print(f"  Processed {len(tasks)} proofs in {total_time:.2f}ms")
    print(f"  Success rate: {successful}/{len(tasks)}")
    print(f"  Throughput: {len(tasks)/total_time*1000:.1f} proofs/sec")

    # Get performance stats
    perf_stats = parallel_prover.get_performance_stats()
    print(f"  Avg queue time: {perf_stats['avg_queue_time_ms']:.2f}ms")
    print(f"  Avg processing time: {perf_stats['avg_processing_time_ms']:.2f}ms")

    parallel_prover.shutdown()

    return results


def identify_optimization_opportunities(profiler: PipelineProfiler):
    """Analyze results and identify optimization opportunities."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 60)

    opportunities = []

    # Analyze each component
    for name, stats in profiler.results.items():
        if not stats["success"]:
            opportunities.append(f"Fix failures in {name}")
            continue

        time_ms = stats["time_ms"]

        if "basic" in name and time_ms > 10:
            opportunities.append(f"Enable caching for {name} (currently {time_ms:.1f}ms)")

        if "parallel" in name and time_ms > 50:
            opportunities.append(f"Increase worker pool for {name}")

        if "gpu" in name and stats.get("computation_device") == "cpu":
            opportunities.append("Enable GPU acceleration (currently using CPU)")

        if stats["memory_delta_mb"] > 100:
            opportunities.append(
                f"Optimize memory usage in {name} (using {stats['memory_delta_mb']:.1f}MB)"
            )

    # Check cache effectiveness
    cache_stats = profiler.results.get("4. Witness Caching", {})
    if cache_stats.get("success"):
        # Could check cache hit rate from monitoring
        pass

    # Print opportunities
    if opportunities:
        print("\n🎯 Optimization Recommendations:")
        for i, opp in enumerate(opportunities, 1):
            print(f"  {i}. {opp}")
    else:
        print("\n✅ Pipeline is well-optimized!")

    # Performance targets
    print("\n📊 Performance Targets:")
    print("  • Witness generation: <5ms (with cache)")
    print("  • Proof generation: <50ms (with GPU)")
    print("  • Batch processing: >100 proofs/sec")
    print("  • Cache hit rate: >80%")
    print("  • Memory usage: <500MB")


def main():
    """Run comprehensive pipeline test."""
    print("🧬 GenomeVault ZK Pipeline Performance Test")
    print("=" * 60)

    profiler = PipelineProfiler()

    # Test each component
    tests = [
        ("1. Basic Proof Generation", test_1_basic_proof_generation),
        ("2. Batch Constraints", test_2_batch_constraint_generation),
        ("3. Adaptive Circuits", test_3_adaptive_circuit_selection),
        ("4. Witness Caching", test_4_witness_caching),
        ("5. Parallel Generation", test_5_parallel_proof_generation),
        ("6. Memory Pooling", test_6_memory_pool),
        ("7. GPU Acceleration", test_7_gpu_acceleration),
        ("8. Performance Monitoring", test_8_performance_monitoring),
        ("9. End-to-End Pipeline", test_9_end_to_end_pipeline),
    ]

    for name, test_func in tests:
        profiler.profile_section(name, test_func)

    # Print summary
    total_time = profiler.print_summary()

    # Identify optimizations
    identify_optimization_opportunities(profiler)

    # Final verdict
    print("\n" + "=" * 60)
    if total_time < 1000:  # <1 second for all tests
        print("⚡ EXCELLENT PERFORMANCE - Pipeline is highly optimized!")
    elif total_time < 5000:  # <5 seconds
        print("✅ GOOD PERFORMANCE - Minor optimizations possible")
    else:
        print("⚠️  NEEDS OPTIMIZATION - See recommendations above")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
