#!/usr/bin/env python3
"""10M row PIR benchmark for GenomeVault - tests at 10x scale"""

import time
import psutil
import numpy as np
from dataclasses import dataclass
from typing import List
import json
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomevault.pir.accelerated_pir import AcceleratedPIREngine

@dataclass
class BenchmarkResult:
    """PIR benchmark result"""
    num_rows: int
    row_size_bytes: int
    total_size_mb: float
    query_time_ms: float
    throughput_mbps: float
    cpu_percent: float
    memory_mb: float
    backend: str

def run_10m_benchmark():
    """Run 10M row PIR benchmark with Metal acceleration"""
    
    print("=" * 80)
    print("GenomeVault 10M Row PIR Benchmark")
    print("=" * 80)
    
    # Configuration
    num_rows = 10_000_000  # 10M rows
    row_size = 1000  # 1KB per row (genomic variant data)
    num_servers = 3
    
    # Create synthetic database
    print(f"\n📊 Creating database with {num_rows:,} rows ({row_size} bytes each)...")
    start = time.time()
    
    # Create database in chunks to manage memory
    chunk_size = 100_000
    db_parts = []
    
    for i in range(0, num_rows, chunk_size):
        chunk_rows = min(chunk_size, num_rows - i)
        # Create deterministic data for chunk
        np.random.seed(42 + i // chunk_size)
        chunk = np.random.bytes(chunk_rows * row_size)
        db_parts.append(chunk)
        
        if (i + chunk_size) % 1_000_000 == 0:
            print(f"  Generated {(i + chunk_size):,} rows...")
    
    # Combine all parts
    database = b''.join(db_parts)
    db_size_mb = len(database) / (1024 * 1024)
    
    creation_time = time.time() - start
    print(f"✅ Database created in {creation_time:.1f}s")
    print(f"   Size: {db_size_mb:.1f} MB")
    
    # Initialize PIR engine with Metal acceleration
    print(f"\n🚀 Initializing AcceleratedPIREngine with {num_servers} servers...")
    start = time.time()
    
    engine = AcceleratedPIREngine(database, n_servers=num_servers)
    
    init_time = time.time() - start
    print(f"✅ Engine initialized in {init_time:.1f}s")
    
    # Run queries
    print(f"\n🔍 Running PIR queries...")
    
    # Warmup
    print("  Warmup query...")
    _ = engine.query(0)
    
    # Actual benchmark
    query_indices = [0, num_rows // 4, num_rows // 2, 3 * num_rows // 4, num_rows - 1]
    results = []
    
    for idx in query_indices:
        # Monitor resources
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        # Run query
        start = time.time()
        result = engine.query(idx)
        query_time = (time.time() - start) * 1000  # ms
        
        # Get resource usage
        cpu_after = process.cpu_percent()
        mem_after = process.memory_info().rss / (1024 * 1024)
        
        # Calculate throughput
        throughput = (row_size / 1024 / 1024) / (query_time / 1000)  # MB/s
        
        # Store result
        benchmark = BenchmarkResult(
            num_rows=num_rows,
            row_size_bytes=row_size,
            total_size_mb=db_size_mb,
            query_time_ms=query_time,
            throughput_mbps=throughput,
            cpu_percent=cpu_after,
            memory_mb=mem_after,
            backend="Metal GPU"  # AcceleratedPIREngine uses Metal
        )
        results.append(benchmark)
        
        print(f"  Query {idx:,}: {query_time:.2f}ms (CPU: {cpu_after:.1f}%, Mem: {mem_after:.0f}MB)")
    
    # Calculate statistics
    query_times = [r.query_time_ms for r in results]
    avg_time = np.mean(query_times)
    p50 = np.percentile(query_times, 50)
    p95 = np.percentile(query_times, 95)
    p99 = np.percentile(query_times, 99)
    
    print("\n" + "=" * 80)
    print("📈 BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\n🗄️  Database:")
    print(f"   • Rows: {num_rows:,}")
    print(f"   • Row size: {row_size:,} bytes")
    print(f"   • Total size: {db_size_mb:.1f} MB")
    print(f"   • Servers: {num_servers}")
    
    print(f"\n⚡ Performance:")
    print(f"   • Average query: {avg_time:.2f}ms")
    print(f"   • P50 latency: {p50:.2f}ms")
    print(f"   • P95 latency: {p95:.2f}ms")
    print(f"   • P99 latency: {p99:.2f}ms")
    
    print(f"\n💻 Resources:")
    print(f"   • Backend: Metal GPU (AcceleratedPIREngine)")
    print(f"   • Avg CPU: {np.mean([r.cpu_percent for r in results]):.1f}%")
    print(f"   • Peak Memory: {max(r.memory_mb for r in results):.0f} MB")
    
    print(f"\n🎯 Efficiency:")
    print(f"   • Queries/second: {1000/avg_time:.1f}")
    print(f"   • Throughput: {np.mean([r.throughput_mbps for r in results]):.2f} MB/s")
    print(f"   • Rows/second: {num_rows / (avg_time/1000):,.0f}")
    
    # Compare to smaller scales
    print(f"\n📊 Scale Comparison:")
    print(f"   • 100K rows: ~200-500ms")
    print(f"   • 1M rows: ~918ms")
    print(f"   • 10M rows: {avg_time:.0f}ms")
    print(f"   • Scaling factor: {avg_time/918:.1f}x from 1M")
    
    # Save results
    output_file = "benchmark_results/pir_10m_results.json"
    os.makedirs("benchmark_results", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'num_rows': num_rows,
                'row_size_bytes': row_size,
                'total_size_mb': db_size_mb,
                'num_servers': num_servers
            },
            'results': {
                'avg_query_ms': avg_time,
                'p50_ms': p50,
                'p95_ms': p95,
                'p99_ms': p99,
                'backend': results[0].backend,
                'queries_per_second': 1000/avg_time,
                'rows_per_second': num_rows / (avg_time/1000)
            },
            'individual_queries': [
                {
                    'index': idx,
                    'time_ms': r.query_time_ms,
                    'cpu_percent': r.cpu_percent,
                    'memory_mb': r.memory_mb
                }
                for idx, r in zip(query_indices, results)
            ]
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Clinical use case analysis
    print("\n" + "=" * 80)
    print("🏥 CLINICAL USE CASE ANALYSIS")
    print("=" * 80)
    
    if avg_time < 1000:
        print("✅ Sub-second queries suitable for real-time clinical decisions")
    elif avg_time < 5000:
        print("✅ Acceptable for interactive clinical workflows")
    else:
        print("⚠️  May require optimization for real-time clinical use")
    
    print(f"\n📋 Recommended Use Cases at 10M Scale:")
    print(f"   • Population-wide genomic studies")
    print(f"   • Large cohort association analyses")
    print(f"   • National biobank queries")
    print(f"   • Pharmacogenomics screening programs")
    
    print(f"\n🔬 Research Applications:")
    print(f"   • GWAS with {num_rows:,} variants")
    print(f"   • Rare variant discovery across populations")
    print(f"   • Multi-ethnic genomic databases")
    print(f"   • Longitudinal genomic monitoring")

if __name__ == "__main__":
    run_10m_benchmark()