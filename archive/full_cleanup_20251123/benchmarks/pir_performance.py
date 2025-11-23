#!/usr/bin/env python3
"""
PIR Performance Benchmark Suite for GenomeVault.

Tests PIR query performance with varying database sizes,
measures overhead of fixed-size padding, and validates
constant-time execution guarantees.
"""

import sys
import time
import json
import argparse
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.pir.xor_scheme import XORPIRScheme, XORSchemeParams
from genomevault.pir.byzantine_handler import ByzantineHandler, ByzantineConfig
from genomevault.pir.query_processor import ConstantTimeQueryProcessor, ProcessorConfig
from genomevault.observability.metrics import metrics_manager


@dataclass
class BenchmarkConfig:
    """Configuration for PIR benchmarks."""

    database_sizes: List[int] = None  # Number of blocks
    block_size: int = 1024  # 1KB blocks as per SECURITY.md
    num_servers: int = 2
    num_queries: int = 100
    warmup_queries: int = 10
    enable_byzantine: bool = True
    enable_constant_time: bool = True
    max_genome_size: int = 3 * 1024 * 1024 * 1024  # 3GB
    output_dir: str = "benchmark_results"

    def __post_init__(self):
        if self.database_sizes is None:
            # Test with varying database sizes
            self.database_sizes = [
                1000,  # 1MB
                10000,  # 10MB
                100000,  # 100MB
                1000000,  # 1GB
                3000000,  # 3GB (max genome size)
            ]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    database_size: int
    block_size: int
    num_queries: int
    query_generation_time_ms: List[float]
    server_processing_time_ms: List[float]
    response_combination_time_ms: List[float]
    total_time_ms: List[float]
    response_sizes: List[int]
    padding_overhead_percent: float
    timing_variance: float
    success_rate: float
    byzantine_faults_detected: int

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for the benchmark."""
        return {
            "database_size": self.database_size,
            "block_size": self.block_size,
            "num_queries": self.num_queries,
            "query_generation": {
                "mean_ms": statistics.mean(self.query_generation_time_ms),
                "median_ms": statistics.median(self.query_generation_time_ms),
                "p95_ms": np.percentile(self.query_generation_time_ms, 95),
                "p99_ms": np.percentile(self.query_generation_time_ms, 99),
            },
            "server_processing": {
                "mean_ms": statistics.mean(self.server_processing_time_ms),
                "median_ms": statistics.median(self.server_processing_time_ms),
                "p95_ms": np.percentile(self.server_processing_time_ms, 95),
                "p99_ms": np.percentile(self.server_processing_time_ms, 99),
            },
            "total_latency": {
                "mean_ms": statistics.mean(self.total_time_ms),
                "median_ms": statistics.median(self.total_time_ms),
                "p95_ms": np.percentile(self.total_time_ms, 95),
                "p99_ms": np.percentile(self.total_time_ms, 99),
            },
            "padding_overhead_percent": self.padding_overhead_percent,
            "timing_variance": self.timing_variance,
            "success_rate": self.success_rate,
            "byzantine_faults": self.byzantine_faults_detected,
            "throughput_qps": self.num_queries / (sum(self.total_time_ms) / 1000),
        }


class PIRPerformanceBenchmark:
    """
    PIR performance benchmark suite.

    Tests:
    1. Query latency with varying database sizes
    2. Fixed-size padding overhead
    3. Constant-time execution validation
    4. Byzantine fault tolerance performance
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark suite.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.results: List[BenchmarkResult] = []

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize PIR components
        self._initialize_pir()

    def _initialize_pir(self) -> None:
        """Initialize PIR scheme and processors."""
        # XOR scheme parameters
        self.xor_params = XORSchemeParams(
            database_size=self.config.database_sizes[0],
            block_size=self.config.block_size,
            num_servers=self.config.num_servers,
            response_padding_size=self.config.block_size,
            enable_byzantine_protection=self.config.enable_byzantine,
        )

        # Byzantine handler
        if self.config.enable_byzantine:
            byzantine_config = ByzantineConfig(
                num_servers=self.config.num_servers, min_servers=2, redundancy_factor=2
            )
            self.byzantine_handler = ByzantineHandler(byzantine_config)
        else:
            self.byzantine_handler = None

        # Query processor
        processor_config = ProcessorConfig(
            constant_time_ns=10_000_000 if self.config.enable_constant_time else 0,
            enable_caching=False,  # Disable for benchmarking
        )
        self.processor = ConstantTimeQueryProcessor(
            processor_config,
            self.xor_params,
            ByzantineConfig() if self.config.enable_byzantine else None,
        )

    def run_benchmarks(self) -> List[BenchmarkResult]:
        """
        Run complete benchmark suite.

        Returns:
            List of benchmark results
        """
        print("Starting PIR Performance Benchmarks")
        print(f"Configuration: {json.dumps(asdict(self.config), indent=2)}")
        print("-" * 80)

        for db_size in self.config.database_sizes:
            print(
                f"\nBenchmarking with database size: {db_size} blocks ({db_size * self.config.block_size / 1e9:.2f} GB)"
            )

            # Update parameters for this database size
            self.xor_params.database_size = db_size

            # Generate test database
            database = self._generate_database(db_size)

            # Run benchmark
            result = self._benchmark_database(database, db_size)
            self.results.append(result)

            # Print summary
            stats = result.get_statistics()
            print(f"  Mean latency: {stats['total_latency']['mean_ms']:.2f} ms")
            print(f"  P95 latency: {stats['total_latency']['p95_ms']:.2f} ms")
            print(f"  P99 latency: {stats['total_latency']['p99_ms']:.2f} ms")
            print(f"  Throughput: {stats['throughput_qps']:.2f} QPS")
            print(f"  Padding overhead: {stats['padding_overhead_percent']:.1f}%")
            print(f"  Success rate: {stats['success_rate']*100:.1f}%")

        # Save results
        self._save_results()

        # Generate plots
        self._generate_plots()

        return self.results

    def _generate_database(self, num_blocks: int) -> np.ndarray:
        """
        Generate test database.

        Args:
            num_blocks: Number of blocks in database

        Returns:
            Binary database array
        """
        # Each block is block_size bytes = block_size * 8 bits
        bits_per_block = self.config.block_size * 8

        # Generate random binary data
        database = np.random.randint(0, 2, (num_blocks, bits_per_block), dtype=np.uint8)

        return database

    def _benchmark_database(self, database: np.ndarray, db_size: int) -> BenchmarkResult:
        """
        Benchmark PIR operations on a database.

        Args:
            database: Test database
            db_size: Database size in blocks

        Returns:
            Benchmark results
        """
        query_gen_times = []
        server_proc_times = []
        response_comb_times = []
        total_times = []
        response_sizes = []
        successes = 0
        byzantine_faults = 0

        # Initialize XOR scheme
        scheme = XORPIRScheme(self.xor_params)

        # Progress bar
        pbar = tqdm(
            total=self.config.num_queries + self.config.warmup_queries, desc="Queries", leave=False
        )

        # Run warmup queries
        for _ in range(self.config.warmup_queries):
            index = np.random.randint(0, db_size)

            # Generate queries
            q1, q2 = scheme.generate_queries(index, db_size)

            # Process queries (warmup, don't record times)
            r1 = scheme.process_query_constant_time(q1, database, 1)
            r2 = scheme.process_query_constant_time(q2, database, 2)

            pbar.update(1)

        # Run benchmark queries
        for i in range(self.config.num_queries):
            index = np.random.randint(0, db_size)

            # Time query generation
            start = time.perf_counter()
            q1, q2 = scheme.generate_queries(index, db_size)
            query_gen_time = (time.perf_counter() - start) * 1000
            query_gen_times.append(query_gen_time)

            # Time server processing
            start = time.perf_counter()
            r1 = scheme.process_query_constant_time(q1, database, 1)
            r2 = scheme.process_query_constant_time(q2, database, 2)
            server_proc_time = (time.perf_counter() - start) * 1000
            server_proc_times.append(server_proc_time)

            # Record response sizes
            response_sizes.append(len(r1.response_data))
            response_sizes.append(len(r2.response_data))

            # Time response combination
            start = time.perf_counter()
            try:
                result = scheme.combine_responses(r1, r2)
                response_comb_time = (time.perf_counter() - start) * 1000
                response_comb_times.append(response_comb_time)
                successes += 1
            except Exception as e:
                response_comb_time = (time.perf_counter() - start) * 1000
                response_comb_times.append(response_comb_time)
                if "Byzantine" in str(e):
                    byzantine_faults += 1

            # Total time
            total_time = query_gen_time + server_proc_time + response_comb_time
            total_times.append(total_time)

            # Record metrics
            if metrics_manager.enabled:
                metrics_manager.record_pir_query(
                    server_id=1,
                    query_type="benchmark",
                    duration=total_time / 1000,
                    response_size=len(r1.response_data),
                    status="success" if successes > i else "failure",
                )

            pbar.update(1)

        pbar.close()

        # Calculate padding overhead
        actual_data_size = self.config.block_size
        padded_size = statistics.mean(response_sizes)
        padding_overhead = ((padded_size - actual_data_size) / actual_data_size) * 100

        # Calculate timing variance (for constant-time validation)
        timing_variance = statistics.stdev(server_proc_times) if len(server_proc_times) > 1 else 0

        return BenchmarkResult(
            database_size=db_size,
            block_size=self.config.block_size,
            num_queries=self.config.num_queries,
            query_generation_time_ms=query_gen_times,
            server_processing_time_ms=server_proc_times,
            response_combination_time_ms=response_comb_times,
            total_time_ms=total_times,
            response_sizes=response_sizes,
            padding_overhead_percent=padding_overhead,
            timing_variance=timing_variance,
            success_rate=successes / self.config.num_queries,
            byzantine_faults_detected=byzantine_faults,
        )

    def _save_results(self) -> None:
        """Save benchmark results to files."""
        # Save raw results as JSON
        results_data = []
        for result in self.results:
            stats = result.get_statistics()
            results_data.append(stats)

        output_file = Path(self.config.output_dir) / "pir_benchmark_results.json"
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResults saved to {output_file}")

        # Save as CSV for analysis
        df = pd.DataFrame(results_data)
        csv_file = Path(self.config.output_dir) / "pir_benchmark_results.csv"
        df.to_csv(csv_file, index=False)

        # Save detailed metrics
        detailed_file = Path(self.config.output_dir) / "pir_benchmark_detailed.json"
        detailed_data = []
        for result in self.results:
            detailed_data.append(
                {
                    "database_size": result.database_size,
                    "query_gen_times": result.query_generation_time_ms,
                    "server_proc_times": result.server_processing_time_ms,
                    "response_comb_times": result.response_combination_time_ms,
                    "total_times": result.total_time_ms,
                    "response_sizes": result.response_sizes,
                }
            )

        with open(detailed_file, "w") as f:
            json.dump(detailed_data, f)

    def _generate_plots(self) -> None:
        """Generate performance plots."""
        if not self.results:
            return

        # Prepare data
        db_sizes = []
        mean_latencies = []
        p95_latencies = []
        p99_latencies = []
        throughputs = []
        padding_overheads = []
        timing_variances = []

        for result in self.results:
            stats = result.get_statistics()
            db_sizes.append(result.database_size * self.config.block_size / 1e9)  # GB
            mean_latencies.append(stats["total_latency"]["mean_ms"])
            p95_latencies.append(stats["total_latency"]["p95_ms"])
            p99_latencies.append(stats["total_latency"]["p99_ms"])
            throughputs.append(stats["throughput_qps"])
            padding_overheads.append(stats["padding_overhead_percent"])
            timing_variances.append(stats["timing_variance"])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("PIR Performance Benchmark Results", fontsize=16)

        # Plot 1: Latency vs Database Size
        ax1 = axes[0, 0]
        ax1.plot(db_sizes, mean_latencies, "b-", label="Mean", marker="o")
        ax1.plot(db_sizes, p95_latencies, "g--", label="P95", marker="s")
        ax1.plot(db_sizes, p99_latencies, "r-.", label="P99", marker="^")
        ax1.set_xlabel("Database Size (GB)")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Query Latency vs Database Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        # Add SLO lines
        ax1.axhline(y=500, color="orange", linestyle=":", label="P95 SLO (500ms)")
        ax1.axhline(y=2000, color="red", linestyle=":", label="P99 SLO (2s)")

        # Plot 2: Throughput vs Database Size
        ax2 = axes[0, 1]
        ax2.plot(db_sizes, throughputs, "g-", marker="o")
        ax2.set_xlabel("Database Size (GB)")
        ax2.set_ylabel("Throughput (QPS)")
        ax2.set_title("Query Throughput vs Database Size")
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")

        # Plot 3: Padding Overhead
        ax3 = axes[0, 2]
        ax3.bar(range(len(db_sizes)), padding_overheads, color="orange")
        ax3.set_xlabel("Database Size Index")
        ax3.set_ylabel("Padding Overhead (%)")
        ax3.set_title("Fixed-Size Padding Overhead")
        ax3.set_xticks(range(len(db_sizes)))
        ax3.set_xticklabels([f"{s:.1f}GB" for s in db_sizes], rotation=45)
        ax3.grid(True, alpha=0.3, axis="y")

        # Plot 4: Timing Variance (Constant-Time Validation)
        ax4 = axes[1, 0]
        ax4.plot(db_sizes, timing_variances, "r-", marker="o")
        ax4.set_xlabel("Database Size (GB)")
        ax4.set_ylabel("Timing Variance (ms)")
        ax4.set_title("Constant-Time Execution Variance")
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale("log")
        ax4.axhline(y=1.0, color="green", linestyle="--", label="Target (<1ms)")
        ax4.legend()

        # Plot 5: Latency Distribution
        ax5 = axes[1, 1]
        for i, result in enumerate(self.results):
            if i % 2 == 0:  # Plot every other result for clarity
                ax5.hist(result.total_time_ms, bins=30, alpha=0.5, label=f"{db_sizes[i]:.1f}GB")
        ax5.set_xlabel("Latency (ms)")
        ax5.set_ylabel("Frequency")
        ax5.set_title("Latency Distribution")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Plot 6: Success Rate
        ax6 = axes[1, 2]
        success_rates = [r.success_rate * 100 for r in self.results]
        ax6.plot(db_sizes, success_rates, "g-", marker="o")
        ax6.set_xlabel("Database Size (GB)")
        ax6.set_ylabel("Success Rate (%)")
        ax6.set_title("Query Success Rate")
        ax6.grid(True, alpha=0.3)
        ax6.set_xscale("log")
        ax6.set_ylim([95, 101])
        ax6.axhline(y=99.9, color="green", linestyle="--", label="SLO (99.9%)")
        ax6.legend()

        plt.tight_layout()

        # Save plot
        plot_file = Path(self.config.output_dir) / "pir_benchmark_plots.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        print(f"Plots saved to {plot_file}")

        plt.show()

    def validate_slos(self) -> Dict[str, bool]:
        """
        Validate against defined SLOs.

        Returns:
            Dictionary of SLO name to pass/fail status
        """
        slo_results = {}

        for result in self.results:
            stats = result.get_statistics()
            db_size_gb = result.database_size * self.config.block_size / 1e9

            # P95 latency ≤ 500ms for standard queries (< 1GB)
            if db_size_gb <= 1.0:
                slo_key = f"p95_latency_standard_{db_size_gb:.1f}GB"
                slo_results[slo_key] = stats["total_latency"]["p95_ms"] <= 500

            # P99 ≤ 2s for complex genomic queries (all sizes)
            slo_key = f"p99_latency_complex_{db_size_gb:.1f}GB"
            slo_results[slo_key] = stats["total_latency"]["p99_ms"] <= 2000

            # 99.9% availability (success rate)
            slo_key = f"availability_{db_size_gb:.1f}GB"
            slo_results[slo_key] = stats["success_rate"] >= 0.999

            # Constant-time execution (variance < 1ms)
            slo_key = f"constant_time_{db_size_gb:.1f}GB"
            slo_results[slo_key] = stats["timing_variance"] < 1.0

        # Print SLO validation results
        print("\n" + "=" * 80)
        print("SLO Validation Results:")
        print("-" * 80)

        for slo_name, passed in slo_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            color = "\033[92m" if passed else "\033[91m"
            reset = "\033[0m"
            print(f"{color}{status}{reset} {slo_name}")

        overall_pass = all(slo_results.values())
        print("-" * 80)
        if overall_pass:
            print("\033[92m✓ All SLOs PASSED\033[0m")
        else:
            failed_count = sum(1 for v in slo_results.values() if not v)
            print(f"\033[91m✗ {failed_count} SLOs FAILED\033[0m")

        return slo_results


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="PIR Performance Benchmark Suite")
    parser.add_argument("--database-sizes", nargs="+", type=int, help="Database sizes in blocks")
    parser.add_argument(
        "--num-queries", type=int, default=100, help="Number of queries per benchmark"
    )
    parser.add_argument(
        "--block-size", type=int, default=1024, help="Block size in bytes (default: 1024 for 1KB)"
    )
    parser.add_argument("--num-servers", type=int, default=2, help="Number of PIR servers")
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument(
        "--disable-byzantine", action="store_true", help="Disable Byzantine fault tolerance"
    )
    parser.add_argument(
        "--disable-constant-time", action="store_true", help="Disable constant-time execution"
    )

    args = parser.parse_args()

    # Create configuration
    config = BenchmarkConfig(
        database_sizes=args.database_sizes,
        block_size=args.block_size,
        num_servers=args.num_servers,
        num_queries=args.num_queries,
        enable_byzantine=not args.disable_byzantine,
        enable_constant_time=not args.disable_constant_time,
        output_dir=args.output_dir,
    )

    # Run benchmarks
    benchmark = PIRPerformanceBenchmark(config)
    results = benchmark.run_benchmarks()

    # Validate SLOs
    slo_results = benchmark.validate_slos()

    # Save SLO results
    slo_file = Path(config.output_dir) / "slo_validation.json"
    with open(slo_file, "w") as f:
        json.dump(slo_results, f, indent=2)

    print(f"\nBenchmark complete. Results saved to {config.output_dir}/")

    # Return exit code based on SLO validation
    return 0 if all(slo_results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
