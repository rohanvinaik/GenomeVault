"""
Automatic optimization selection based on hardware capabilities.

This module generates optimized pipeline configurations based on detected
hardware, ensuring maximum performance on any system architecture.

Usage:
    from genomevault.compute.optimization_selector import OptimizationSelector

    selector = OptimizationSelector()
    config = selector.generate_optimal_config()
    selector.write_config_file("pipeline_config.yaml")
"""

import yaml
from typing import Dict, Any
from pathlib import Path
import logging

from genomevault.compute.hardware_detector import HardwareDetector

logger = logging.getLogger(__name__)


class OptimizationSelector:
    """Generate optimal pipeline configuration based on hardware."""

    def __init__(self, hardware_detector: HardwareDetector = None):
        """
        Initialize optimization selector.

        Args:
            hardware_detector: Optional pre-initialized HardwareDetector
        """
        self.detector = hardware_detector or HardwareDetector()
        self.config = {}

    def generate_optimal_config(self) -> Dict[str, Any]:
        """
        Generate optimal pipeline configuration.

        Returns:
            Dict with complete pipeline configuration
        """
        # Detect hardware
        capabilities = self.detector.detect_all()
        recommendations = self.detector.recommend_optimizations()

        # Build configuration
        self.config = {
            "pipeline": {
                "name": "GenomeVault Enhanced Privacy Pipeline",
                "auto_configured": True,
                "hardware_detected": True
            },
            "hardware": {
                "cpu_cores": capabilities["cpu"]["cores"],
                "total_memory_gb": capabilities["memory"]["total_gb"],
                "gpu_backend": capabilities["gpu"]["recommended_backend"],
                "storage_type": capabilities["storage"]["type"]
            },
            "phase1": self._configure_phase1(recommendations["phase1"]),
            "phase2": self._configure_phase2(recommendations["phase2"]),
            "phase3": self._configure_phase3(recommendations["phase3"]),
            "command_line_args": self._generate_cli_args(recommendations)
        }

        return self.config

    def _configure_phase1(self, phase1_rec: Dict) -> Dict:
        """Configure Phase 1 optimizations."""
        return {
            "enabled": True,
            "description": "Immediate wins (30 min, 5.6 hours saved)",
            "optimizations": {
                "sambamba_sorting": {
                    "enabled": phase1_rec.get("use_sambamba", False),
                    "threads": phase1_rec.get("sambamba_threads", 8),
                    "memory": phase1_rec.get("sambamba_memory", "2G"),
                    "expected_speedup": "2-3×"
                },
                "parallel_bcftools": {
                    "enabled": phase1_rec.get("use_parallel_bcftools", False),
                    "threads": phase1_rec.get("bcftools_threads", 4),
                    "expected_speedup": "1.5-2×"
                },
                "metal_gpu_hdc": {
                    "enabled": phase1_rec.get("use_metal_gpu", False),
                    "backend": phase1_rec.get("metal_backend", "cpu"),
                    "expected_speedup": "43×" if phase1_rec.get("use_metal_gpu") else "1×"
                }
            }
        }

    def _configure_phase2(self, phase2_rec: Dict) -> Dict:
        """Configure Phase 2 optimizations."""
        return {
            "enabled": phase2_rec.get("use_amx", False) or phase2_rec.get("use_index_caching", True),
            "description": "High-impact (5 hours, 2.4 hours saved)",
            "optimizations": {
                "minimap2_index_caching": {
                    "enabled": True,  # Always enabled
                    "cache_dir": "~/.genomevault/minimap2_cache",
                    "expected_savings": "60 sec per reference"
                },
                "amx_alignment": {
                    "enabled": phase2_rec.get("use_amx", False),
                    "chip": phase2_rec.get("amx_chip", "N/A"),
                    "expected_speedup": phase2_rec.get("expected_speedup", "1×")
                }
            }
        }

    def _configure_phase3(self, phase3_rec: Dict) -> Dict:
        """Configure Phase 3 optimizations."""
        return {
            "enabled": phase3_rec.get("use_chromosome_parallel_sort", False) or
                      phase3_rec.get("use_parallel_vcf_parsing", False),
            "description": "Advanced (8 hours, 2.1 hours saved)",
            "optimizations": {
                "chromosome_parallel_sort": {
                    "enabled": phase3_rec.get("use_chromosome_parallel_sort", False),
                    "max_parallel": phase3_rec.get("max_parallel_chromosomes", 12),
                    "expected_speedup": phase3_rec.get("expected_speedup", "1×"),
                    "note": "Best for whole-genome data (24 chromosomes)"
                },
                "parallel_vcf_parsing": {
                    "enabled": phase3_rec.get("use_parallel_vcf_parsing", False),
                    "workers": phase3_rec.get("vcf_workers", 4),
                    "expected_speedup": "2-3×"
                }
            }
        }

    def _generate_cli_args(self, recommendations: Dict) -> Dict:
        """Generate command-line arguments for the pipeline."""
        phase1 = recommendations["phase1"]
        phase2 = recommendations["phase2"]
        phase3 = recommendations["phase3"]

        args = {
            "required": [
                "--output-dir", "benchmark_results/enhanced_privacy_k13_optimized_$(date +%Y%m%d_%H%M%S)",
                "--num-references", "12"
            ],
            "phase1": [],
            "phase2": [],
            "phase3": []
        }

        # Phase 1 args
        if phase1.get("use_sambamba"):
            args["phase1"].extend(["--use-sambamba"])
            args["phase1"].extend(["--sambamba-threads", str(phase1.get("sambamba_threads", 8))])
            args["phase1"].extend(["--sambamba-memory", phase1.get("sambamba_memory", "2G")])

        if phase1.get("use_parallel_bcftools"):
            args["phase1"].extend(["--parallel-bcftools"])
            args["phase1"].extend(["--bcftools-threads", str(phase1.get("bcftools_threads", 4))])

        if phase1.get("use_metal_gpu"):
            args["phase1"].extend(["--gpu-backend", "metal"])
        else:
            args["phase1"].extend(["--gpu-backend", "cpu"])

        args["phase1"].extend(["--threads", str(phase1.get("recommended_threads", 8))])

        # Phase 2 args
        args["phase2"].extend(["--enable-index-caching"])  # Always enabled

        if phase2.get("use_amx"):
            args["phase2"].extend(["--enable-amx"])

        # Phase 3 args
        if phase3.get("use_chromosome_parallel_sort"):
            args["phase3"].extend(["--use-chromosome-partitioned-sort"])
            args["phase3"].extend(["--max-parallel-chromosomes", str(phase3.get("max_parallel_chromosomes", 12))])

        if phase3.get("use_parallel_vcf_parsing"):
            args["phase3"].extend(["--use-parallel-vcf-parsing"])
            args["phase3"].extend(["--vcf-workers", str(phase3.get("vcf_workers", 4))])

        return args

    def write_config_file(self, output_path: str = "pipeline_config.yaml"):
        """
        Write configuration to YAML file.

        Args:
            output_path: Path to output YAML file
        """
        if not self.config:
            self.generate_optimal_config()

        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration written to: {output_path}")

    def print_deployment_commands(self):
        """Print ready-to-use deployment commands."""
        if not self.config:
            self.generate_optimal_config()

        cli_args = self.config["command_line_args"]

        print("\n" + "=" * 70)
        print("🚀 Deployment Commands")
        print("=" * 70)

        # Phase 1 command
        if self.config["phase1"]["enabled"]:
            print("\n⭐⭐⭐ Phase 1 (Deploy NOW - 30 min, 5.6 hours saved)")
            print("-" * 70)

            cmd_parts = ["python3 scripts/run_enhanced_privacy_pipeline_optimized.py"]
            cmd_parts.extend(cli_args["required"])
            cmd_parts.extend(cli_args["phase1"])

            cmd = " \\\n    ".join(cmd_parts)
            print(f"\n{cmd}\n")

        # Phase 2 command
        if self.config["phase2"]["enabled"]:
            print("\n⭐⭐ Phase 2 (Deploy after Phase 1 - 5 hours, 2.4 hours saved)")
            print("-" * 70)

            cmd_parts = ["python3 scripts/run_enhanced_privacy_pipeline_optimized.py"]
            cmd_parts.extend(cli_args["required"])
            cmd_parts.extend(cli_args["phase1"])
            cmd_parts.extend(cli_args["phase2"])

            cmd = " \\\n    ".join(cmd_parts)
            print(f"\n{cmd}\n")

        # Phase 3 command
        if self.config["phase3"]["enabled"]:
            print("\n⭐ Phase 3 (Optional - 8 hours, 2.1 hours saved)")
            print("-" * 70)

            cmd_parts = ["python3 scripts/run_enhanced_privacy_pipeline_optimized.py"]
            cmd_parts.extend(cli_args["required"])
            cmd_parts.extend(cli_args["phase1"])
            cmd_parts.extend(cli_args["phase2"])
            cmd_parts.extend(cli_args["phase3"])

            cmd = " \\\n    ".join(cmd_parts)
            print(f"\n{cmd}\n")

        print("=" * 70)

    def print_summary(self):
        """Print configuration summary."""
        if not self.config:
            self.generate_optimal_config()

        print("\n" + "=" * 70)
        print("📋 Optimization Configuration Summary")
        print("=" * 70)

        # Hardware summary
        hw = self.config["hardware"]
        print(f"\n🖥️  Detected Hardware:")
        print(f"  CPU Cores: {hw['cpu_cores']}")
        print(f"  Total Memory: {hw['total_memory_gb']} GB")
        print(f"  GPU Backend: {hw['gpu_backend']}")
        print(f"  Storage: {hw['storage_type']}")

        # Phase 1 summary
        p1 = self.config["phase1"]
        print(f"\n⭐⭐⭐ Phase 1: {p1['description']}")
        for opt_name, opt_config in p1["optimizations"].items():
            status = "✅ ENABLED" if opt_config["enabled"] else "❌ DISABLED"
            print(f"  {opt_name}: {status}")
            if opt_config["enabled"]:
                if "threads" in opt_config:
                    print(f"    Threads: {opt_config['threads']}")
                if "memory" in opt_config:
                    print(f"    Memory: {opt_config['memory']}")
                print(f"    Expected speedup: {opt_config['expected_speedup']}")

        # Phase 2 summary
        p2 = self.config["phase2"]
        print(f"\n⭐⭐ Phase 2: {p2['description']}")
        for opt_name, opt_config in p2["optimizations"].items():
            status = "✅ ENABLED" if opt_config["enabled"] else "❌ DISABLED"
            print(f"  {opt_name}: {status}")
            if opt_config["enabled"] and opt_name == "amx_alignment":
                if opt_config.get("chip") != "N/A":
                    print(f"    Chip: {opt_config['chip']}")
                print(f"    Expected speedup: {opt_config['expected_speedup']}")

        # Phase 3 summary
        p3 = self.config["phase3"]
        print(f"\n⭐ Phase 3: {p3['description']}")
        for opt_name, opt_config in p3["optimizations"].items():
            status = "✅ ENABLED" if opt_config["enabled"] else "❌ DISABLED"
            print(f"  {opt_name}: {status}")
            if opt_config["enabled"]:
                if "max_parallel" in opt_config:
                    print(f"    Max parallel: {opt_config['max_parallel']}")
                if "workers" in opt_config:
                    print(f"    Workers: {opt_config['workers']}")
                if "note" in opt_config:
                    print(f"    Note: {opt_config['note']}")

        print("\n" + "=" * 70)


def main():
    """Main function for standalone execution."""
    print("GenomeVault Optimization Auto-Configuration")
    print("=" * 70)
    print("Detecting hardware and generating optimal configuration...\n")

    selector = OptimizationSelector()

    # Generate configuration
    config = selector.generate_optimal_config()

    # Print hardware report
    selector.detector.print_report()

    # Print configuration summary
    selector.print_summary()

    # Print deployment commands
    selector.print_deployment_commands()

    # Write config file
    output_file = "genomevault_auto_config.yaml"
    selector.write_config_file(output_file)
    print(f"\n✅ Configuration saved to: {output_file}")
    print("\nYou can now deploy Phase 1 when ref1 completes sorting!\n")


if __name__ == "__main__":
    main()
