#!/usr/bin/env python3
"""
Comprehensive Differential Encoding Benchmark Suite
GenomeVault v2.0.0

Runs all differential encoding benchmarks and generates standardized results.
This reflects the major architectural update where differential encoding is now
a core feature, not an optional add-on.

Usage:
    python scripts/run_differential_encoding_benchmarks.py
    python scripts/run_differential_encoding_benchmarks.py --quick
    python scripts/run_differential_encoding_benchmarks.py --output custom_results/
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).parent.parent
BENCHMARKS_DIR = ROOT / "benchmarks" / "differential_encoding"
RESULTS_DIR = ROOT / "benchmark_results" / "differential_encoding"


class DifferentialEncodingBenchmarkRunner:
    """Orchestrate differential encoding benchmark suite"""

    def __init__(self, output_dir: Path, quick: bool = False):
        self.output_dir = output_dir
        self.quick = quick
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "genomevault_version": "2.0.0",
                "architecture": "differential_encoding_core",
                "mode": "quick" if quick else "full"
            },
            "benchmarks": {}
        }

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(self, name: str, script_path: Path, params: dict = None) -> dict:
        """Run a single benchmark script and capture results"""
        logger.info(f"Running {name}...")

        start_time = time.time()

        try:
            # Build command
            cmd = [sys.executable, str(script_path)]

            if self.quick:
                cmd.append("--quick")

            if params:
                for key, value in params.items():
                    cmd.extend([f"--{key}", str(value)])

            # Run benchmark
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=600 if self.quick else 3600  # 10 min quick, 1 hour full
            )

            elapsed = time.time() - start_time

            # Try to parse JSON output (handle multi-line JSON)
            json_output = None
            stdout_text = result.stdout.strip()

            # Try to parse entire stdout as JSON first
            try:
                json_output = json.loads(stdout_text)
            except json.JSONDecodeError:
                # If that fails, look for JSON in individual lines
                output_lines = stdout_text.split('\n')
                for line in reversed(output_lines):
                    if line.strip().startswith('{'):
                        try:
                            json_output = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue

            return {
                "status": "success",
                "elapsed_seconds": elapsed,
                "results": json_output,
                "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
            }

        except subprocess.TimeoutExpired:
            logger.error(f"{name} timed out")
            return {
                "status": "timeout",
                "elapsed_seconds": time.time() - start_time
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"{name} failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "stderr": e.stderr[-500:] if e.stderr and len(e.stderr) > 500 else e.stderr
            }
        except Exception as e:
            logger.error(f"{name} error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def run_chunking_benchmark(self):
        """Benchmark adaptive chunking strategies"""
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Adaptive Chunking Benchmark")
        logger.info("="*60)

        result = self.run_benchmark(
            "chunking",
            BENCHMARKS_DIR / "benchmark_chunking.py"
        )

        self.results["benchmarks"]["chunking"] = result

        if result["status"] == "success" and result["results"]:
            metrics = result["results"]
            logger.info(f"  ✓ Best strategy: {metrics.get('best_strategy', 'N/A')}")
            logger.info(f"  ✓ Average time: {metrics.get('avg_time_ms', 0):.2f}ms")

        return result["status"] == "success"

    def run_difference_computation_benchmark(self):
        """Benchmark differential encoding computation"""
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Difference Computation Benchmark")
        logger.info("="*60)

        result = self.run_benchmark(
            "difference_computation",
            BENCHMARKS_DIR / "benchmark_difference_computation.py"
        )

        self.results["benchmarks"]["difference_computation"] = result

        if result["status"] == "success" and result["results"]:
            metrics = result["results"]
            logger.info(f"  ✓ Encoding time: {metrics.get('encoding_time_ms', 0):.2f}ms")
            logger.info(f"  ✓ Throughput: {metrics.get('throughput_variants_per_sec', 0):.0f} variants/sec")

        return result["status"] == "success"

    def run_hypervector_encoding_benchmark(self):
        """Benchmark hypervector projection and binding"""
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: Hypervector Encoding Benchmark")
        logger.info("="*60)

        result = self.run_benchmark(
            "hypervector_encoding",
            BENCHMARKS_DIR / "benchmark_hypervector_encoding.py"
        )

        self.results["benchmarks"]["hypervector_encoding"] = result

        if result["status"] == "success" and result["results"]:
            metrics = result["results"]
            logger.info(f"  ✓ MLX acceleration: {metrics.get('mlx_time_ms', 0):.2f}ms")
            logger.info(f"  ✓ Compression ratio: {metrics.get('compression_ratio', 0):.0f}:1")

        return result["status"] == "success"

    def run_end_to_end_benchmark(self):
        """Benchmark complete differential encoding pipeline"""
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: End-to-End Pipeline Benchmark")
        logger.info("="*60)

        result = self.run_benchmark(
            "end_to_end",
            BENCHMARKS_DIR / "benchmark_end_to_end.py"
        )

        self.results["benchmarks"]["end_to_end"] = result

        if result["status"] == "success" and result["results"]:
            metrics = result["results"]
            logger.info(f"  ✓ Total pipeline time: {metrics.get('total_time_ms', 0):.2f}ms")
            logger.info(f"  ✓ Final size: {metrics.get('final_size_kb', 0):.2f}KB")

        return result["status"] == "success"

    def generate_summary(self):
        """Generate executive summary of benchmark results"""
        logger.info("\n" + "="*60)
        logger.info("GENERATING SUMMARY")
        logger.info("="*60)

        summary = {
            "system": "GenomeVault v2.0.0 (Differential Encoding)",
            "timestamp": self.results["metadata"]["timestamp"],
            "overall_status": "success",
            "key_metrics": {}
        }

        # Extract key metrics from benchmarks
        benchmarks = self.results["benchmarks"]

        # Helper function to extract results from stdout if results is null
        def get_results(benchmark_data):
            results = benchmark_data.get("results")
            if results:
                return results
            # Try to parse from stdout
            stdout = benchmark_data.get("stdout", "")
            try:
                # Try to extract JSON from stdout (starts with { and ends with })
                import re
                json_match = re.search(r'\{[\s\S]*\}', stdout)
                if json_match:
                    return json.loads(json_match.group())
            except (json.JSONDecodeError, AttributeError):
                # If full JSON parse fails, try to extract individual fields
                try:
                    import re
                    extracted = {}
                    # Extract common numeric fields
                    for field in ['best_strategy', 'avg_time_ms', 'encoding_time_ms', 'throughput_variants_per_sec',
                                   'compression_ratio', 'mlx_time_ms', 'cpu_time_ms', 'total_time_ms',
                                   'final_size_kb', 'throughput_genomes_per_hour']:
                        # Look for "field": value patterns
                        if field == 'best_strategy':
                            match = re.search(rf'"{field}":\s*"([^"]+)"', stdout)
                        else:
                            match = re.search(rf'"{field}":\s*([\d.]+)', stdout)
                        if match:
                            if field == 'best_strategy':
                                extracted[field] = match.group(1)
                            else:
                                try:
                                    extracted[field] = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                                except:
                                    pass
                    if extracted:
                        return extracted
                except:
                    pass
            return {}

        # Chunking
        if "chunking" in benchmarks and benchmarks["chunking"]["status"] == "success":
            chunking_results = get_results(benchmarks["chunking"])
            summary["key_metrics"]["adaptive_chunking"] = {
                "best_strategy": chunking_results.get("best_strategy"),
                "avg_time_ms": chunking_results.get("avg_time_ms")
            }

        # Difference computation
        if "difference_computation" in benchmarks and benchmarks["difference_computation"]["status"] == "success":
            diff_results = get_results(benchmarks["difference_computation"])
            summary["key_metrics"]["differential_encoding"] = {
                "encoding_time_ms": diff_results.get("encoding_time_ms"),
                "throughput_variants_per_sec": diff_results.get("throughput_variants_per_sec"),
                "compression_ratio": diff_results.get("compression_ratio")
            }

        # Hypervector encoding
        if "hypervector_encoding" in benchmarks and benchmarks["hypervector_encoding"]["status"] == "success":
            hv_results = get_results(benchmarks["hypervector_encoding"])
            summary["key_metrics"]["hypervector_projection"] = {
                "mlx_time_ms": hv_results.get("mlx_time_ms"),
                "cpu_time_ms": hv_results.get("cpu_time_ms"),
                "compression_ratio": hv_results.get("compression_ratio")
            }

        # End-to-end
        if "end_to_end" in benchmarks and benchmarks["end_to_end"]["status"] == "success":
            e2e_results = get_results(benchmarks["end_to_end"])
            summary["key_metrics"]["end_to_end_pipeline"] = {
                "total_time_ms": e2e_results.get("total_time_ms"),
                "final_size_kb": e2e_results.get("final_size_kb"),
                "throughput_genomes_per_hour": e2e_results.get("throughput_genomes_per_hour")
            }

        # Check for any failures
        failed = [name for name, result in benchmarks.items()
                  if result["status"] != "success"]

        if failed:
            summary["overall_status"] = "partial"
            summary["failed_benchmarks"] = failed

        self.results["summary"] = summary

        return summary

    def save_results(self):
        """Save results to JSON file"""
        output_file = self.output_dir / f"differential_encoding_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\nResults saved to: {output_file}")

        # Also save to standard location for pipeline
        standard_file = self.output_dir / "latest_results.json"
        with open(standard_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Latest results link: {standard_file}")

        return output_file

    def run_all(self):
        """Run complete benchmark suite"""
        logger.info("\n" + "="*70)
        logger.info("GenomeVault v2.0.0 - Differential Encoding Benchmark Suite")
        logger.info("="*70)
        logger.info(f"Mode: {'QUICK' if self.quick else 'FULL'}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("")

        start_time = time.time()

        # Run all benchmark phases
        phases = [
            ("Chunking", self.run_chunking_benchmark),
            ("Difference Computation", self.run_difference_computation_benchmark),
            ("Hypervector Encoding", self.run_hypervector_encoding_benchmark),
            ("End-to-End Pipeline", self.run_end_to_end_benchmark),
        ]

        results_summary = {}
        for phase_name, phase_func in phases:
            try:
                success = phase_func()
                results_summary[phase_name] = "✓" if success else "✗"
            except Exception as e:
                logger.error(f"Phase {phase_name} crashed: {e}")
                results_summary[phase_name] = "✗ (crashed)"

        # Generate summary
        summary = self.generate_summary()

        # Save results
        output_file = self.save_results()

        elapsed = time.time() - start_time

        # Final report
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK SUITE COMPLETE")
        logger.info("="*70)
        logger.info(f"Total time: {elapsed:.1f} seconds")
        logger.info("\nPhase Results:")
        for phase, status in results_summary.items():
            logger.info(f"  {status} {phase}")

        logger.info(f"\nOverall Status: {summary['overall_status'].upper()}")

        if summary.get("failed_benchmarks"):
            logger.warning(f"Failed benchmarks: {', '.join(summary['failed_benchmarks'])}")

        logger.info(f"\nResults file: {output_file}")
        logger.info("")

        return 0 if summary["overall_status"] in ["success", "partial"] else 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive differential encoding benchmarks"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: reduced iterations for faster results'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=RESULTS_DIR,
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Create runner and execute
    runner = DifferentialEncodingBenchmarkRunner(
        output_dir=args.output,
        quick=args.quick
    )

    return runner.run_all()


if __name__ == "__main__":
    sys.exit(main())
