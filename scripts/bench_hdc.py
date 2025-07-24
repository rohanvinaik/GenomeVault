#!/usr/bin/env python3
"""
HDC Performance Benchmarking Script

Comprehensive benchmarking for the HDC encoding implementation
following Stage 4 requirements.
"""

import json
import time
import psutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from genomevault.hypervector_transform.hdc_encoder import (
    HypervectorEncoder,
    HypervectorConfig,
    CompressionTier,
    ProjectionType,
    OmicsType,
    create_encoder
)
from genomevault.hypervector_transform.binding_operations import (
    HypervectorBinder,
    BindingType
)


class HDCBenchmark:
    """Comprehensive HDC benchmarking suite"""
    
    def __init__(self, output_dir: str = "benchmarks/hdc"):
        """Initialize benchmark suite"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "benchmarks": {}
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": os.sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": os.sys.platform
        }
    
    def benchmark_encoding_throughput(self) -> Dict[str, Any]:
        """Benchmark encoding throughput across dimensions and tiers"""
        print("Benchmarking encoding throughput...")
        
        results = {
            "description": "Encoding throughput (operations/second)",
            "data": {}
        }
        
        # Test parameters
        feature_sizes = [100, 500, 1000, 5000]
        dimensions = [5000, 10000, 15000, 20000]
        
        for dim in dimensions:
            results["data"][f"dim_{dim}"] = {}
            
            for feat_size in feature_sizes:
                encoder = create_encoder(dimension=dim)
                features = np.random.randn(feat_size)
                
                # Warm-up
                for _ in range(10):
                    _ = encoder.encode(features, OmicsType.GENOMIC)
                
                # Benchmark
                num_trials = 100
                start = time.perf_counter()
                
                for _ in range(num_trials):
                    _ = encoder.encode(features, OmicsType.GENOMIC)
                
                elapsed = time.perf_counter() - start
                throughput = num_trials / elapsed
                
                results["data"][f"dim_{dim}"][f"features_{feat_size}"] = {
                    "throughput_ops_per_sec": throughput,
                    "avg_time_ms": (elapsed / num_trials) * 1000,
                    "total_time_sec": elapsed
                }
                
                print(f"  Dim={dim}, Features={feat_size}: {throughput:.1f} ops/s")
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage across tiers"""
        print("\nBenchmarking memory usage...")
        
        results = {
            "description": "Memory usage by compression tier",
            "data": {}
        }
        
        features = np.random.randn(1000)
        
        for tier in CompressionTier:
            encoder = create_encoder(compression_tier=tier.value)
            
            # Encode
            hv = encoder.encode(features, OmicsType.GENOMIC, tier)
            
            # Calculate memory
            memory_bytes = hv.element_size() * hv.nelement()
            memory_kb = memory_bytes / 1024
            
            # Calculate compression ratio
            original_size = features.nbytes
            compression_ratio = original_size / memory_bytes
            
            # Get sparsity
            sparsity = (hv == 0).float().mean().item()
            
            results["data"][tier.value] = {
                "dimension": hv.shape[0],
                "memory_kb": memory_kb,
                "memory_mb": memory_kb / 1024,
                "compression_ratio": compression_ratio,
                "bytes_per_element": hv.element_size(),
                "sparsity": sparsity
            }
            
            print(f"  {tier.value}: {memory_kb:.1f} KB (ratio: {compression_ratio:.2f}x, sparsity: {sparsity:.2%})")
        
        return results
    
    def benchmark_binding_operations(self) -> Dict[str, Any]:
        """Benchmark different binding operations"""
        print("\nBenchmarking binding operations...")
        
        results = {
            "description": "Binding operation performance",
            "data": {}
        }
        
        dimension = 10000
        binder = HypervectorBinder(dimension)
        
        # Create test vectors
        num_vectors = 5
        vectors = [torch.randn(dimension) for _ in range(num_vectors)]
        
        for binding_type in BindingType:
            # Test with 2 vectors (some operations don't support more)
            test_vectors = vectors[:2]
            
            try:
                # Warm-up
                for _ in range(10):
                    _ = binder.bind(test_vectors, binding_type)
                
                # Benchmark
                num_trials = 100
                start = time.perf_counter()
                
                for _ in range(num_trials):
                    _ = binder.bind(test_vectors, binding_type)
                
                elapsed = time.perf_counter() - start
                throughput = num_trials / elapsed
                
                results["data"][binding_type.value] = {
                    "throughput_ops_per_sec": throughput,
                    "avg_time_ms": (elapsed / num_trials) * 1000,
                    "supported": True
                }
                
                print(f"  {binding_type.value}: {throughput:.1f} ops/s")
                
            except Exception as e:
                results["data"][binding_type.value] = {
                    "supported": False,
                    "error": str(e)
                }
                print(f"  {binding_type.value}: Not supported for this configuration")
        
        return results
    
    def benchmark_similarity_computation(self) -> Dict[str, Any]:
        """Benchmark similarity computation"""
        print("\nBenchmarking similarity computation...")
        
        results = {
            "description": "Similarity computation performance",
            "data": {}
        }
        
        dimensions = [5000, 10000, 20000]
        metrics = ["cosine", "euclidean", "hamming"]
        
        for dim in dimensions:
            results["data"][f"dim_{dim}"] = {}
            encoder = create_encoder(dimension=dim)
            
            # Create test vectors
            v1 = torch.randn(dim)
            v2 = torch.randn(dim)
            
            for metric in metrics:
                # Warm-up
                for _ in range(100):
                    _ = encoder.similarity(v1, v2, metric)
                
                # Benchmark
                num_trials = 1000
                start = time.perf_counter()
                
                for _ in range(num_trials):
                    _ = encoder.similarity(v1, v2, metric)
                
                elapsed = time.perf_counter() - start
                throughput = num_trials / elapsed
                
                results["data"][f"dim_{dim}"][metric] = {
                    "throughput_ops_per_sec": throughput,
                    "avg_time_us": (elapsed / num_trials) * 1e6,
                    "total_time_ms": elapsed * 1000
                }
                
                print(f"  Dim={dim}, Metric={metric}: {throughput:.1f} ops/s ({(elapsed/num_trials)*1e6:.1f} µs/op)")
        
        return results
    
    def benchmark_projection_types(self) -> Dict[str, Any]:
        """Benchmark different projection types"""
        print("\nBenchmarking projection types...")
        
        results = {
            "description": "Performance by projection type",
            "data": {}
        }
        
        dimension = 10000
        features = np.random.randn(1000)
        
        for proj_type in ProjectionType:
            try:
                config = HypervectorConfig(
                    dimension=dimension,
                    projection_type=proj_type
                )
                encoder = HypervectorEncoder(config)
                
                # Warm-up
                for _ in range(10):
                    _ = encoder.encode(features, OmicsType.GENOMIC)
                
                # Benchmark
                num_trials = 50
                start = time.perf_counter()
                
                for _ in range(num_trials):
                    _ = encoder.encode(features, OmicsType.GENOMIC)
                
                elapsed = time.perf_counter() - start
                throughput = num_trials / elapsed
                
                # Get cache stats
                cache_stats = encoder.get_projection_stats()
                
                results["data"][proj_type.value] = {
                    "throughput_ops_per_sec": throughput,
                    "avg_time_ms": (elapsed / num_trials) * 1000,
                    "cache_entries": cache_stats["num_cached_matrices"],
                    "cache_memory_mb": cache_stats["memory_mb"],
                    "supported": True
                }
                
                print(f"  {proj_type.value}: {throughput:.1f} ops/s (cache: {cache_stats['memory_mb']:.1f} MB)")
                
            except Exception as e:
                results["data"][proj_type.value] = {
                    "supported": False,
                    "error": str(e)
                }
                print(f"  {proj_type.value}: Not implemented")
        
        return results
    
    def benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark scalability with batch processing"""
        print("\nBenchmarking batch scalability...")
        
        results = {
            "description": "Batch processing scalability",
            "data": {}
        }
        
        encoder = create_encoder(dimension=10000)
        batch_sizes = [1, 10, 50, 100, 500]
        
        baseline_throughput = None
        
        for batch_size in batch_sizes:
            # Create batch
            features = torch.randn(batch_size, 1000)
            
            # Warm-up
            for _ in range(5):
                _ = encoder.encode(features, OmicsType.GENOMIC)
            
            # Benchmark
            num_trials = 20
            start = time.perf_counter()
            
            for _ in range(num_trials):
                _ = encoder.encode(features, OmicsType.GENOMIC)
            
            elapsed = time.perf_counter() - start
            
            # Calculate throughput per sample
            total_samples = batch_size * num_trials
            throughput_per_sample = total_samples / elapsed
            
            if baseline_throughput is None:
                baseline_throughput = throughput_per_sample
            
            results["data"][f"batch_{batch_size}"] = {
                "throughput_samples_per_sec": throughput_per_sample,
                "avg_batch_time_ms": (elapsed / num_trials) * 1000,
                "speedup": throughput_per_sample / baseline_throughput,
                "efficiency": (throughput_per_sample / baseline_throughput) / batch_size
            }
            
            print(f"  Batch size={batch_size}: {throughput_per_sample:.1f} samples/s (speedup: {throughput_per_sample/baseline_throughput:.2f}x)")
        
        return results
    
    def benchmark_multimodal_encoding(self) -> Dict[str, Any]:
        """Benchmark multi-modal encoding performance"""
        print("\nBenchmarking multi-modal encoding...")
        
        results = {
            "description": "Multi-modal encoding and binding performance",
            "data": {}
        }
        
        encoder = create_encoder(dimension=10000)
        binder = HypervectorBinder(10000)
        
        # Create multi-modal test data
        modalities = {
            "genomic": np.random.randn(1000),
            "transcriptomic": np.random.randn(500),
            "epigenomic": np.random.randn(300),
            "clinical": {"age": 45, "bmi": 25.5, "gender": "F"}
        }
        
        # Benchmark individual encoding
        for modality, data in modalities.items():
            omics_type = OmicsType[modality.upper()] if modality != "clinical" else OmicsType.CLINICAL
            
            # Warm-up
            for _ in range(10):
                _ = encoder.encode(data, omics_type)
            
            # Benchmark
            num_trials = 50
            start = time.perf_counter()
            
            for _ in range(num_trials):
                _ = encoder.encode(data, omics_type)
            
            elapsed = time.perf_counter() - start
            
            results["data"][modality] = {
                "encoding_time_ms": (elapsed / num_trials) * 1000,
                "throughput_ops_per_sec": num_trials / elapsed
            }
        
        # Benchmark complete multi-modal pipeline
        start = time.perf_counter()
        
        encoded_vectors = []
        for modality, data in modalities.items():
            omics_type = OmicsType[modality.upper()] if modality != "clinical" else OmicsType.CLINICAL
            hv = encoder.encode(data, omics_type)
            encoded_vectors.append(hv)
        
        # Bind all modalities
        combined = binder.bind(encoded_vectors, BindingType.FOURIER)
        
        total_time = time.perf_counter() - start
        
        results["data"]["complete_pipeline"] = {
            "total_time_ms": total_time * 1000,
            "num_modalities": len(modalities),
            "final_dimension": combined.shape[0]
        }
        
        print(f"  Complete pipeline: {total_time*1000:.1f} ms for {len(modalities)} modalities")
        
        return results
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("Starting HDC benchmarking suite...")
        print(f"Output directory: {self.output_dir}")
        print(f"Timestamp: {self.timestamp}")
        print("-" * 60)
        
        # Run benchmarks
        self.results["benchmarks"]["encoding_throughput"] = self.benchmark_encoding_throughput()
        self.results["benchmarks"]["memory_usage"] = self.benchmark_memory_usage()
        self.results["benchmarks"]["binding_operations"] = self.benchmark_binding_operations()
        self.results["benchmarks"]["similarity_computation"] = self.benchmark_similarity_computation()
        self.results["benchmarks"]["projection_types"] = self.benchmark_projection_types()
        self.results["benchmarks"]["scalability"] = self.benchmark_scalability()
        self.results["benchmarks"]["multimodal"] = self.benchmark_multimodal_encoding()
        
        # Calculate summary metrics
        self.results["summary"] = self._calculate_summary()
        
        # Save results
        output_file = self.output_dir / f"{self.timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print("-" * 60)
        print(f"Benchmarks complete! Results saved to: {output_file}")
        
        # Generate visualizations
        self.generate_plots()
        
        # Generate performance badge
        self.generate_performance_badge()
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary metrics"""
        # Get reference throughput (10k dimension, 1000 features)
        throughput_data = self.results["benchmarks"]["encoding_throughput"]["data"]
        ref_throughput = throughput_data.get("dim_10000", {}).get("features_1000", {}).get("throughput_ops_per_sec", 0)
        
        # Memory efficiency
        mem_data = self.results["benchmarks"]["memory_usage"]["data"]
        clinical_memory = mem_data.get("clinical", {}).get("memory_kb", 0)
        
        return {
            "reference_throughput_ops_per_sec": ref_throughput,
            "clinical_tier_memory_kb": clinical_memory,
            "timestamp": self.timestamp
        }
    
    def generate_plots(self):
        """Generate visualization plots"""
        print("\nGenerating visualization plots...")
        
        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 150
        
        # Create plots directory
        plots_dir = self.output_dir / "plots" / self.timestamp
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Throughput by dimension
        self._plot_throughput_by_dimension(plots_dir)
        
        # 2. Memory usage by tier
        self._plot_memory_by_tier(plots_dir)
        
        # 3. Binding operation comparison
        self._plot_binding_comparison(plots_dir)
        
        # 4. Scalability plot
        self._plot_scalability(plots_dir)
        
        # 5. Projection type comparison
        self._plot_projection_comparison(plots_dir)
        
        print(f"Plots saved to {plots_dir}")
    
    def _plot_throughput_by_dimension(self, plots_dir: Path):
        """Plot encoding throughput by dimension"""
        data = self.results["benchmarks"]["encoding_throughput"]["data"]
        
        dimensions = []
        throughputs = []
        
        for dim_key, dim_data in data.items():
            dim = int(dim_key.split("_")[1])
            # Use 1000 feature size as reference
            throughput = dim_data.get("features_1000", {}).get("throughput_ops_per_sec", 0)
            dimensions.append(dim)
            throughputs.append(throughput)
        
        plt.figure()
        plt.plot(dimensions, throughputs, 'o-', linewidth=2, markersize=8, color='#1f77b4')
        plt.xlabel("Hypervector Dimension")
        plt.ylabel("Throughput (operations/second)")
        plt.title("HDC Encoding Throughput vs Dimension\n(1000 features)")
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(dimensions, throughputs):
            plt.annotate(f'{y:.0f}', xy=(x, y), xytext=(0, 5), 
                        textcoords='offset points', ha='center')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "throughput_by_dimension.png")
        plt.close()
    
    def _plot_memory_by_tier(self, plots_dir: Path):
        """Plot memory usage by compression tier"""
        data = self.results["benchmarks"]["memory_usage"]["data"]
        
        tiers = list(data.keys())
        memory_kb = [data[tier]["memory_kb"] for tier in tiers]
        dimensions = [data[tier]["dimension"] for tier in tiers]
        
        fig, ax = plt.subplots()
        bars = ax.bar(tiers, memory_kb, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        # Add value labels on bars
        for bar, kb, dim in zip(bars, memory_kb, dimensions):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{kb:.0f} KB\n({dim}D)', ha='center', va='bottom')
        
        ax.set_xlabel("Compression Tier")
        ax.set_ylabel("Memory Usage (KB)")
        ax.set_title("HDC Memory Usage by Compression Tier")
        
        plt.tight_layout()
        plt.savefig(plots_dir / "memory_by_tier.png")
        plt.close()
    
    def _plot_binding_comparison(self, plots_dir: Path):
        """Plot binding operation performance comparison"""
        data = self.results["benchmarks"]["binding_operations"]["data"]
        
        operations = []
        throughputs = []
        
        for op, op_data in data.items():
            if op_data.get("supported", False):
                operations.append(op)
                throughputs.append(op_data["throughput_ops_per_sec"])
        
        plt.figure()
        bars = plt.bar(operations, throughputs, color='skyblue')
        
        # Add value labels
        for bar, throughput in zip(bars, throughputs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{throughput:.0f}', ha='center', va='bottom')
        
        plt.xlabel("Binding Operation")
        plt.ylabel("Throughput (operations/second)")
        plt.title("HDC Binding Operation Performance")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "binding_comparison.png")
        plt.close()
    
    def _plot_scalability(self, plots_dir: Path):
        """Plot batch processing scalability"""
        data = self.results["benchmarks"]["scalability"]["data"]
        
        batch_sizes = []
        speedups = []
        efficiencies = []
        
        for batch_key, batch_data in data.items():
            batch_size = int(batch_key.split("_")[1])
            speedup = batch_data.get("speedup", 1)
            efficiency = batch_data.get("efficiency", 1)
            batch_sizes.append(batch_size)
            speedups.append(speedup)
            efficiencies.append(efficiency * 100)  # Convert to percentage
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Speedup plot
        ax1.plot(batch_sizes, speedups, 'o-', linewidth=2, markersize=8, label='Actual')
        ax1.plot(batch_sizes, batch_sizes, '--', alpha=0.5, label='Ideal linear')
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Speedup")
        ax1.set_title("HDC Batch Processing Speedup")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency plot
        ax2.plot(batch_sizes, efficiencies, 'o-', linewidth=2, markersize=8, color='green')
        ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Perfect efficiency')
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Efficiency (%)")
        ax2.set_title("HDC Batch Processing Efficiency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 120)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "scalability.png")
        plt.close()
    
    def _plot_projection_comparison(self, plots_dir: Path):
        """Plot projection type performance comparison"""
        data = self.results["benchmarks"]["projection_types"]["data"]
        
        proj_types = []
        throughputs = []
        memory_usage = []
        
        for proj_type, proj_data in data.items():
            if proj_data.get("supported", False):
                proj_types.append(proj_type)
                throughputs.append(proj_data["throughput_ops_per_sec"])
                memory_usage.append(proj_data.get("cache_memory_mb", 0))
        
        if not proj_types:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Throughput
        bars1 = ax1.bar(proj_types, throughputs, color='lightcoral')
        ax1.set_xlabel("Projection Type")
        ax1.set_ylabel("Throughput (ops/sec)")
        ax1.set_title("Performance by Projection Type")
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory usage
        bars2 = ax2.bar(proj_types, memory_usage, color='lightgreen')
        ax2.set_xlabel("Projection Type")
        ax2.set_ylabel("Cache Memory (MB)")
        ax2.set_title("Memory Usage by Projection Type")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "projection_comparison.png")
        plt.close()
    
    def generate_performance_badge(self):
        """Generate performance badge for README"""
        summary = self.results.get("summary", {})
        ref_throughput = summary.get("reference_throughput_ops_per_sec", 0)
        
        # Convert to GOPS (Giga operations per second) if > 1000
        if ref_throughput > 1000:
            badge_text = f"HDC Performance: {ref_throughput/1000:.1f} KOPS"
        else:
            badge_text = f"HDC Performance: {ref_throughput:.0f} OPS"
        
        # Save badge info
        badge_file = self.output_dir / "performance_badge.txt"
        with open(badge_file, "w") as f:
            f.write(badge_text)
        
        # Also create a JSON file for CI integration
        badge_json = self.output_dir / "performance_badge.json"
        with open(badge_json, "w") as f:
            json.dump({
                "label": "HDC Performance",
                "message": badge_text.split(": ")[1],
                "color": "brightgreen" if ref_throughput > 100 else "yellow"
            }, f)
        
        print(f"\nPerformance badge: {badge_text}")


def main():
    """Main benchmark entry point"""
    parser = argparse.ArgumentParser(description="HDC Performance Benchmarking")
    parser.add_argument("--output-dir", default="benchmarks/hdc", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = HDCBenchmark(args.output_dir)
    
    # Run benchmarks
    if args.quick:
        # Quick mode - only essential benchmarks
        print("Running quick benchmarks...")
        benchmark.results["benchmarks"]["encoding_throughput"] = benchmark.benchmark_encoding_throughput()
        benchmark.results["benchmarks"]["memory_usage"] = benchmark.benchmark_memory_usage()
        
        # Save quick results
        output_file = benchmark.output_dir / f"quick_{benchmark.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(benchmark.results, f, indent=2)
        
        print(f"\nQuick benchmark results saved to: {output_file}")
    else:
        # Full benchmarks
        benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()
