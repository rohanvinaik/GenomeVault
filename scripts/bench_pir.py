from __future__ import annotations

"""
PIR Performance Benchmarking Script
Measures latency, throughput, and resource usage for PIR operations.
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from tabulate import tabulate

from genomevault.pir.it_pir_protocol import BatchPIRProtocol, PIRParameters, PIRProtocol
from genomevault.pir.server.enhanced_pir_server import EnhancedPIRServer, ServerConfig


class PIRBenchmark:
    """PIR performance benchmarking suite."""

    def __init__(self, output_dir: Path):
        """Initialize the instance.
        Args:        output_dir: Parameter value."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "benchmarks": {},
        }

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": os.uname().system,
            "python_version": os.sys.version,
        }

    async def benchmark_query_generation(
        self, database_sizes: list[int], num_iterations: int = 100
    ):
        """Benchmark query vector generation."""
        logger.info("\n=== Query Generation Benchmark ===")
        results = []

        for db_size in database_sizes:
            params = PIRParameters(database_size=db_size, num_servers=2)
            protocol = PIRProtocol(params)

            # Warm up
            for _ in range(10):
                protocol.generate_query_vectors(0)

            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                index = np.random.randint(0, db_size)
                protocol.generate_query_vectors(index)
            elapsed = time.time() - start_time

            queries_per_sec = num_iterations / elapsed
            avg_time_ms = (elapsed / num_iterations) * 1000

            results.append(
                {
                    "database_size": db_size,
                    "queries_per_second": queries_per_sec,
                    "avg_time_ms": avg_time_ms,
                    "total_time_s": elapsed,
                }
            )

            logger.info(
                f"DB Size: {db_size:>8} | "
                f"Queries/sec: {queries_per_sec:>8.1f} | "
                f"Avg time: {avg_time_ms:>6.2f}ms"
            )

        self.results["benchmarks"]["query_generation"] = results
        return results

    async def benchmark_server_response(self, database_sizes: list[int], num_iterations: int = 10):
        """Benchmark server response computation."""
        logger.info("\n=== Server Response Benchmark ===")
        results = []

        for db_size in database_sizes:
            params = PIRParameters(database_size=db_size)
            protocol = PIRProtocol(params)

            # Create database
            database = np.random.randint(0, 256, (db_size, 1024), dtype=np.uint8)

            # Generate queries
            queries = []
            for _ in range(num_iterations):
                index = np.random.randint(0, db_size)
                query_vectors = protocol.generate_query_vectors(index)
                queries.append((index, query_vectors[0]))

            # Measure CPU and memory before
            process = psutil.Process()
            cpu_before = process.cpu_percent()
            mem_before = process.memory_info().rss / (1024**2)  # MB

            # Benchmark
            start_time = time.time()
            for _, query in queries:
                response = protocol.process_server_response(query, database)
            elapsed = time.time() - start_time

            # Measure CPU and memory after
            cpu_after = process.cpu_percent()
            mem_after = process.memory_info().rss / (1024**2)  # MB

            responses_per_sec = num_iterations / elapsed
            avg_time_ms = (elapsed / num_iterations) * 1000

            results.append(
                {
                    "database_size": db_size,
                    "responses_per_second": responses_per_sec,
                    "avg_time_ms": avg_time_ms,
                    "cpu_usage_percent": cpu_after - cpu_before,
                    "memory_delta_mb": mem_after - mem_before,
                    "throughput_mbps": (db_size * 1024 * num_iterations / elapsed) / (1024**2),
                }
            )

            logger.info(
                f"DB Size: {db_size:>8} | "
                f"Responses/sec: {responses_per_sec:>6.1f} | "
                f"Avg time: {avg_time_ms:>7.2f}ms | "
                f"CPU: {cpu_after - cpu_before:>5.1f}%"
            )

        self.results["benchmarks"]["server_response"] = results
        return results

    async def benchmark_end_to_end(self, database_sizes: list[int], num_servers: list[int]):
        """Benchmark end-to-end PIR retrieval."""
        logger.info("\n=== End-to-End PIR Benchmark ===")
        results = []

        for db_size in database_sizes:
            for n_servers in num_servers:
                params = PIRParameters(database_size=db_size, num_servers=n_servers)
                protocol = PIRProtocol(params)

                # Create database
                database = np.random.randint(0, 256, (db_size, 1024), dtype=np.uint8)

                # Benchmark multiple retrievals
                latencies = []
                for _ in range(20):
                    index = np.random.randint(0, db_size)

                    start_time = time.time()

                    # Generate queries
                    queries = protocol.generate_query_vectors(index)

                    # Process on each server
                    responses = []
                    for query in queries:
                        response = protocol.process_server_response(query, database)
                        responses.append(response)

                    # Reconstruct
                    reconstructed = protocol.reconstruct_element(responses)

                    elapsed = time.time() - start_time
                    latencies.append(elapsed * 1000)  # ms

                    # Verify correctness
                    assert np.array_equal(reconstructed, database[index])

                avg_latency = np.mean(latencies)
                p50_latency = np.percentile(latencies, 50)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)

                results.append(
                    {
                        "database_size": db_size,
                        "num_servers": n_servers,
                        "avg_latency_ms": avg_latency,
                        "p50_latency_ms": p50_latency,
                        "p95_latency_ms": p95_latency,
                        "p99_latency_ms": p99_latency,
                        "bandwidth_per_query_kb": (n_servers * db_size) / 1024,
                    }
                )

                logger.info(
                    f"DB: {db_size:>8} | Servers: {n_servers} | "
                    f"Avg: {avg_latency:>6.1f}ms | "
                    f"P95: {p95_latency:>6.1f}ms | "
                    f"P99: {p99_latency:>6.1f}ms"
                )

        self.results["benchmarks"]["end_to_end"] = results
        return results

    async def benchmark_batch_pir(self, database_size: int, batch_sizes: list[int]):
        """Benchmark batch PIR operations."""
        logger.info("\n=== Batch PIR Benchmark ===")
        results = []

        params = PIRParameters(database_size=database_size)
        batch_protocol = BatchPIRProtocol(params)

        # Create database
        database = np.random.randint(0, 256, (database_size, 1024), dtype=np.uint8)

        for batch_size in batch_sizes:
            # Generate batch of indices
            indices = np.random.choice(database_size, batch_size, replace=False).tolist()

            # Measure batch query generation
            start_time = time.time()
            batch_queries = batch_protocol.generate_batch_queries(indices)
            query_gen_time = (time.time() - start_time) * 1000

            # Measure batch processing
            start_time = time.time()
            total_responses = 0

            for bucket_queries in batch_queries:
                for queries in bucket_queries:
                    for server_queries in queries:
                        for query in server_queries:
                            response = batch_protocol.process_server_response(query, database)
                            total_responses += 1

            process_time = (time.time() - start_time) * 1000

            results.append(
                {
                    "batch_size": batch_size,
                    "query_generation_ms": query_gen_time,
                    "processing_ms": process_time,
                    "total_time_ms": query_gen_time + process_time,
                    "items_per_second": batch_size / ((query_gen_time + process_time) / 1000),
                    "buckets_used": len(batch_queries),
                }
            )

            logger.info(
                f"Batch size: {batch_size:>4} | "
                f"Gen: {query_gen_time:>6.1f}ms | "
                f"Process: {process_time:>7.1f}ms | "
                f"Items/sec: {batch_size / ((query_gen_time + process_time) / 1000):>6.1f}"
            )

        self.results["benchmarks"]["batch_pir"] = results
        return results

    async def benchmark_enhanced_server(self, database_size: int, cache_sizes: list[int]):
        """Benchmark enhanced PIR server with caching."""
        logger.info("\n=== Enhanced Server Benchmark ===")
        results = []

        for cache_size_mb in cache_sizes:
            # Create server config
            config = ServerConfig(
                server_id="bench-server",
                is_trusted_signatory=True,
                database_path="/tmp/bench_db",
                cache_size_mb=cache_size_mb,
                enable_preprocessing=True,
            )

            # Create server
            server = EnhancedPIRServer(config)

            # Generate test queries
            num_unique_queries = 50
            num_total_queries = 200

            params = PIRParameters(database_size=database_size)
            protocol = PIRProtocol(params)

            # Create query pool
            query_pool = []
            for _ in range(num_unique_queries):
                index = np.random.randint(0, database_size)
                query_vector = protocol.generate_query_vectors(index)[0]
                query_data = {
                    "query_id": f"bench-{len(query_pool)}",
                    "query_vector": query_vector.tolist(),
                    "protocol_version": "1.0",
                    "timestamp": time.time(),
                }
                query_pool.append(query_data)

            # Benchmark with cache
            cache_hits = 0
            cache_misses = 0
            latencies = []

            for i in range(num_total_queries):
                # Pick random query from pool (some will repeat)
                query = query_pool[np.random.randint(0, num_unique_queries)].copy()
                query["query_id"] = f"bench-run-{i}"

                start_time = time.time()
                response = await server.process_query(query)
                elapsed = (time.time() - start_time) * 1000

                latencies.append(elapsed)

                if response.get("cached"):
                    cache_hits += 1
                else:
                    cache_misses += 1

            cache_hit_rate = cache_hits / num_total_queries
            avg_latency = np.mean(latencies)

            # Get server stats
            stats = server.get_server_statistics()

            results.append(
                {
                    "cache_size_mb": cache_size_mb,
                    "cache_hit_rate": cache_hit_rate,
                    "avg_latency_ms": avg_latency,
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "memory_used_mb": stats["cache_size_mb"],
                }
            )

            logger.info(
                f"Cache: {cache_size_mb:>4}MB | "
                f"Hit rate: {cache_hit_rate:>5.1%} | "
                f"Avg latency: {avg_latency:>6.1f}ms"
            )

        self.results["benchmarks"]["enhanced_server"] = results
        return results

    async def benchmark_network_latency(self, num_shards: list[int], rtt_ms: float = 70):
        """Benchmark network latency impact."""
        logger.info("\n=== Network Latency Benchmark ===")
        results = []

        for n_shards in num_shards:
            # Simulate TLS handshakes
            tls_handshake_ms = rtt_ms * 1.5  # 1.5 RTT for TLS

            # Total latency
            total_latency = tls_handshake_ms * n_shards

            # Simulated configuration
            if n_shards == 5:
                config = "3 LN + 2 TS"
            elif n_shards == 3:
                config = "1 LN + 2 TS"
            else:
                config = f"{n_shards} servers"

            results.append(
                {
                    "num_shards": n_shards,
                    "configuration": config,
                    "rtt_ms": rtt_ms,
                    "total_latency_ms": total_latency,
                    "tls_overhead_ms": tls_handshake_ms,
                }
            )

            logger.info(
                f"Shards: {n_shards} ({config:>12}) | "
                f"RTT: {rtt_ms}ms | "
                f"Total: {total_latency:>5.0f}ms"
            )

        self.results["benchmarks"]["network_latency"] = results
        return results

    def save_results(self):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"pir_benchmark_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\nResults saved to: {filename}")

        # Also save summary
        self._save_summary()

    def _save_summary(self):
        """Save human-readable summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"pir_benchmark_summary_{timestamp}.txt"

        with open(filename, "w") as f:
            f.write("PIR Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"System: {self.results['system_info']['platform']}\n")
            f.write(f"CPUs: {self.results['system_info']['cpu_count']}\n")
            f.write(f"Memory: {self.results['system_info']['memory_gb']:.1f} GB\n\n")

            # Summary tables
            for bench_name, bench_results in self.results["benchmarks"].items():
                f.write(f"\n{bench_name.replace('_', ' ').title()}\n")
                f.write("-" * 40 + "\n")

                if bench_results:
                    # Convert to table
                    headers = list(bench_results[0].keys())
                    rows = [[r[h] for h in headers] for r in bench_results]

                    # Format numbers
                    formatted_rows = []
                    for row in rows:
                        formatted_row = []
                        for val in row:
                            if isinstance(val, float):
                                formatted_row.append(f"{val:.2f}")
                            else:
                                formatted_row.append(str(val))
                        formatted_rows.append(formatted_row)

                    table = tabulate(formatted_rows, headers=headers, tablefmt="grid")
                    f.write(table + "\n")


async def main():
    """Run PIR benchmarks."""
    parser = argparse.ArgumentParser(description="PIR Performance Benchmarking")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/pir"),
        help="Output directory for results",
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")

    args = parser.parse_args()

    # Create benchmark suite
    benchmark = PIRBenchmark(args.output)

    logger.info("Starting PIR Performance Benchmarks")
    logger.info("=" * 50)

    if args.quick:
        # Quick benchmarks
        await benchmark.benchmark_query_generation([1000, 10000])
        await benchmark.benchmark_server_response([1000, 10000])
        await benchmark.benchmark_end_to_end([1000], [2, 3])
    else:
        # Full benchmarks
        await benchmark.benchmark_query_generation([1000, 10000, 100000, 1000000])
        await benchmark.benchmark_server_response([1000, 10000, 100000])
        await benchmark.benchmark_end_to_end([1000, 10000, 100000], [2, 3, 5])
        await benchmark.benchmark_batch_pir(100000, [10, 50, 100, 500])
        await benchmark.benchmark_enhanced_server(100000, [512, 1024, 2048])
        await benchmark.benchmark_network_latency([2, 3, 5, 10])

    # Save results
    benchmark.save_results()

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
