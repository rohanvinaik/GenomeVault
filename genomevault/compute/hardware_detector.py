"""
Hardware detection and capability assessment for optimization selection.

This module detects system hardware capabilities and recommends appropriate
optimizations for the GenomeVault pipeline.

Usage:
    from genomevault.compute.hardware_detector import HardwareDetector

    detector = HardwareDetector()
    capabilities = detector.detect_all()
    recommendations = detector.recommend_optimizations()
"""

import os
import platform
import subprocess
import shutil
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detect hardware capabilities for optimization selection."""

    def __init__(self):
        """Initialize hardware detector."""
        self.system = platform.system()
        self.machine = platform.machine()
        self.capabilities = {}

    def detect_all(self) -> Dict:
        """
        Detect all hardware capabilities.

        Returns:
            Dict with hardware capabilities:
            {
                "cpu": {...},
                "memory": {...},
                "gpu": {...},
                "storage": {...},
                "tools": {...}
            }
        """
        self.capabilities = {
            "cpu": self.detect_cpu(),
            "memory": self.detect_memory(),
            "gpu": self.detect_gpu(),
            "storage": self.detect_storage(),
            "tools": self.detect_tools(),
            "system": {
                "os": self.system,
                "machine": self.machine,
                "platform": platform.platform()
            }
        }

        return self.capabilities

    def detect_cpu(self) -> Dict:
        """Detect CPU capabilities."""
        cpu_info = {
            "cores": os.cpu_count() or 1,
            "architecture": self.machine,
            "is_apple_silicon": False,
            "has_amx": False,
            "has_avx2": False,
            "has_avx512": False
        }

        # Detect Apple Silicon
        if self.system == "Darwin" and self.machine == "arm64":
            cpu_info["is_apple_silicon"] = True
            cpu_info["has_amx"] = True  # All Apple Silicon has AMX

            # Detect specific chip
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                brand = result.stdout.strip()
                cpu_info["brand"] = brand

                # Parse chip generation (M1, M2, M3, M4)
                if "M1" in brand:
                    cpu_info["chip"] = "M1"
                    cpu_info["gpu_cores"] = 7 if "M1" in brand else 8
                elif "M2" in brand:
                    cpu_info["chip"] = "M2"
                    cpu_info["gpu_cores"] = 10
                elif "M3" in brand:
                    cpu_info["chip"] = "M3"
                    cpu_info["gpu_cores"] = 16
                elif "M4" in brand:
                    cpu_info["chip"] = "M4"
                    cpu_info["gpu_cores"] = 20

            except Exception as e:
                logger.warning(f"Could not detect Apple Silicon chip: {e}")

        # Detect x86_64 features
        elif self.machine in ["x86_64", "AMD64"]:
            try:
                # Check for AVX2/AVX512 support
                if self.system == "Linux":
                    with open("/proc/cpuinfo", "r") as f:
                        cpuinfo = f.read()
                        cpu_info["has_avx2"] = "avx2" in cpuinfo
                        cpu_info["has_avx512"] = "avx512" in cpuinfo

                elif self.system == "Darwin":
                    result = subprocess.run(
                        ["sysctl", "-a"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    sysctl = result.stdout
                    cpu_info["has_avx2"] = "avx2" in sysctl.lower()
                    cpu_info["has_avx512"] = "avx512" in sysctl.lower()

            except Exception as e:
                logger.warning(f"Could not detect x86_64 features: {e}")

        return cpu_info

    def detect_memory(self) -> Dict:
        """Detect memory (RAM) capabilities."""
        mem_info = {
            "total_gb": 0,
            "available_gb": 0,
            "recommended_sambamba_mem": "2G",
            "recommended_samtools_mem": "2G"
        }

        try:
            if self.system == "Darwin":
                # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                total_bytes = int(result.stdout.strip())
                mem_info["total_gb"] = round(total_bytes / (1024**3), 1)

            elif self.system == "Linux":
                # Linux
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_kb = int(line.split()[1])
                            mem_info["total_gb"] = round(total_kb / (1024**2), 1)
                            break

            # Set recommended memory based on total RAM
            total_gb = mem_info["total_gb"]
            if total_gb >= 64:
                mem_info["recommended_sambamba_mem"] = "8G"
                mem_info["recommended_samtools_mem"] = "4G"
            elif total_gb >= 32:
                mem_info["recommended_sambamba_mem"] = "4G"
                mem_info["recommended_samtools_mem"] = "2G"
            elif total_gb >= 16:
                mem_info["recommended_sambamba_mem"] = "2G"
                mem_info["recommended_samtools_mem"] = "1G"
            else:
                mem_info["recommended_sambamba_mem"] = "1G"
                mem_info["recommended_samtools_mem"] = "512M"

        except Exception as e:
            logger.warning(f"Could not detect memory: {e}")

        return mem_info

    def detect_gpu(self) -> Dict:
        """Detect GPU capabilities."""
        gpu_info = {
            "has_metal": False,
            "has_cuda": False,
            "has_opencl": False,
            "recommended_backend": "cpu"
        }

        # Check for Metal (Apple Silicon/macOS)
        if self.system == "Darwin":
            try:
                import mlx.core as mx
                device = mx.default_device()
                gpu_info["has_metal"] = "gpu" in str(device)

                if gpu_info["has_metal"]:
                    gpu_info["recommended_backend"] = "metal"
                    gpu_info["metal_device"] = str(device)

            except ImportError:
                logger.info("MLX not installed (Metal GPU unavailable)")
            except Exception as e:
                logger.warning(f"Could not detect Metal: {e}")

        # Check for CUDA (NVIDIA)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_info["has_cuda"] = True
                gpu_info["cuda_devices"] = result.stdout.strip().split("\n")
                gpu_info["recommended_backend"] = "cuda"

        except FileNotFoundError:
            pass  # nvidia-smi not found
        except Exception as e:
            logger.warning(f"Could not detect CUDA: {e}")

        # Check for OpenCL (Intel/AMD)
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                gpu_info["has_opencl"] = True
                gpu_info["opencl_devices"] = [
                    device.name for platform in platforms
                    for device in platform.get_devices()
                ]
        except ImportError:
            pass  # PyOpenCL not installed
        except Exception as e:
            logger.warning(f"Could not detect OpenCL: {e}")

        return gpu_info

    def detect_storage(self) -> Dict:
        """Detect storage capabilities."""
        storage_info = {
            "type": "unknown",
            "is_ssd": False,
            "is_nvme": False,
            "available_gb": 0
        }

        try:
            if self.system == "Darwin":
                # Check if APFS (implies SSD on macOS)
                result = subprocess.run(
                    ["diskutil", "info", "/"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                output = result.stdout.lower()

                if "solid state" in output or "apfs" in output:
                    storage_info["is_ssd"] = True
                    storage_info["type"] = "SSD"

                # Check for NVMe
                if "nvme" in output or "pcie" in output:
                    storage_info["is_nvme"] = True
                    storage_info["type"] = "NVMe"

            elif self.system == "Linux":
                # Check disk scheduler (none/noop = SSD)
                try:
                    with open("/sys/block/sda/queue/scheduler", "r") as f:
                        scheduler = f.read()
                        if "none" in scheduler or "noop" in scheduler:
                            storage_info["is_ssd"] = True
                            storage_info["type"] = "SSD"
                except FileNotFoundError:
                    pass

                # Check for NVMe devices
                nvme_devices = subprocess.run(
                    ["ls", "/dev/nvme*"],
                    capture_output=True,
                    text=True
                )
                if nvme_devices.returncode == 0:
                    storage_info["is_nvme"] = True
                    storage_info["type"] = "NVMe"

            # Get available space
            stat = os.statvfs("/")
            available_bytes = stat.f_bavail * stat.f_frsize
            storage_info["available_gb"] = round(available_bytes / (1024**3), 1)

        except Exception as e:
            logger.warning(f"Could not detect storage: {e}")

        return storage_info

    def detect_tools(self) -> Dict:
        """Detect installed bioinformatics tools."""
        tools_info = {
            "sambamba": False,
            "samtools": False,
            "bcftools": False,
            "minimap2": False,
            "pigz": False,
            "tabix": False
        }

        # Check each tool (use list to avoid dict modification during iteration)
        tool_names = list(tools_info.keys())
        for tool in tool_names:
            tools_info[tool] = shutil.which(tool) is not None

            # Get version if available
            if tools_info[tool]:
                try:
                    result = subprocess.run(
                        [tool, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    version_line = result.stdout.split("\n")[0]
                    tools_info[f"{tool}_version"] = version_line
                except Exception:
                    pass  # Version detection failed

        return tools_info

    def recommend_optimizations(self) -> Dict:
        """
        Recommend optimizations based on detected hardware.

        Returns:
            Dict with optimization recommendations:
            {
                "phase1": {...},
                "phase2": {...},
                "phase3": {...},
                "warnings": [...]
            }
        """
        if not self.capabilities:
            self.detect_all()

        recommendations = {
            "phase1": {},
            "phase2": {},
            "phase3": {},
            "warnings": [],
            "summary": {}
        }

        # Phase 1 recommendations
        recommendations["phase1"] = self._recommend_phase1()

        # Phase 2 recommendations
        recommendations["phase2"] = self._recommend_phase2()

        # Phase 3 recommendations
        recommendations["phase3"] = self._recommend_phase3()

        # Generate summary
        recommendations["summary"] = self._generate_summary()

        return recommendations

    def _recommend_phase1(self) -> Dict:
        """Recommend Phase 1 optimizations."""
        phase1 = {
            "use_sambamba": False,
            "use_parallel_bcftools": False,
            "use_metal_gpu": False,
            "recommended_threads": 1,
            "recommended_memory": "2G"
        }

        cpu = self.capabilities.get("cpu", {})
        mem = self.capabilities.get("memory", {})
        gpu = self.capabilities.get("gpu", {})
        tools = self.capabilities.get("tools", {})

        # Sambamba recommendation
        if tools.get("sambamba", False):
            phase1["use_sambamba"] = True
            phase1["sambamba_threads"] = min(cpu.get("cores", 1), 16)
            phase1["sambamba_memory"] = mem.get("recommended_sambamba_mem", "2G")
        else:
            phase1["warnings"] = ["sambamba not found - install with: conda install -c bioconda sambamba"]

        # BCFtools parallel recommendation
        if tools.get("bcftools", False):
            phase1["use_parallel_bcftools"] = True
            phase1["bcftools_threads"] = min(cpu.get("cores", 1) // 2, 8)
        else:
            phase1["warnings"] = phase1.get("warnings", []) + ["bcftools not found"]

        # Metal GPU recommendation
        if gpu.get("has_metal", False):
            phase1["use_metal_gpu"] = True
            phase1["metal_backend"] = "metal"
        else:
            phase1["metal_backend"] = "cpu"

        # Thread recommendation
        phase1["recommended_threads"] = min(cpu.get("cores", 1), 16)

        return phase1

    def _recommend_phase2(self) -> Dict:
        """Recommend Phase 2 optimizations."""
        phase2 = {
            "use_index_caching": True,  # Always recommended
            "use_amx": False,
            "warnings": []
        }

        cpu = self.capabilities.get("cpu", {})

        # AMX recommendation (Apple Silicon only)
        if cpu.get("is_apple_silicon", False):
            phase2["use_amx"] = True
            phase2["amx_chip"] = cpu.get("chip", "Unknown")
            phase2["expected_speedup"] = "2-3×"
        else:
            phase2["use_amx"] = False
            phase2["warnings"].append(
                "AMX not available (requires Apple Silicon M1/M2/M3/M4)"
            )

        return phase2

    def _recommend_phase3(self) -> Dict:
        """Recommend Phase 3 optimizations."""
        phase3 = {
            "use_chromosome_parallel_sort": False,
            "use_parallel_vcf_parsing": False,
            "warnings": []
        }

        cpu = self.capabilities.get("cpu", {})
        mem = self.capabilities.get("memory", {})
        storage = self.capabilities.get("storage", {})

        # Chromosome-parallel sorting recommendation
        cores = cpu.get("cores", 1)
        total_mem = mem.get("total_gb", 0)

        if cores >= 12 and total_mem >= 24:
            phase3["use_chromosome_parallel_sort"] = True
            phase3["max_parallel_chromosomes"] = min(cores, 24)
            phase3["expected_speedup"] = "2.5-3×"
        else:
            phase3["warnings"].append(
                f"Chromosome-parallel sorting not recommended: "
                f"need 12+ cores and 24+ GB RAM (have {cores} cores, {total_mem} GB RAM)"
            )

        # Parallel VCF parsing recommendation
        if cores >= 4 and total_mem >= 16:
            phase3["use_parallel_vcf_parsing"] = True
            phase3["vcf_workers"] = min(cores // 2, 7)  # Max 7 VCF files
        else:
            phase3["warnings"].append(
                f"Parallel VCF parsing not recommended: "
                f"need 4+ cores and 16+ GB RAM (have {cores} cores, {total_mem} GB RAM)"
            )

        return phase3

    def _generate_summary(self) -> Dict:
        """Generate optimization summary."""
        phase1 = self.capabilities.get("phase1", {})
        phase2 = self.capabilities.get("phase2", {})
        phase3 = self.capabilities.get("phase3", {})

        # Count enabled optimizations
        phase1_enabled = sum([
            phase1.get("use_sambamba", False),
            phase1.get("use_parallel_bcftools", False),
            phase1.get("use_metal_gpu", False)
        ])

        phase2_enabled = sum([
            phase2.get("use_index_caching", False),
            phase2.get("use_amx", False)
        ])

        phase3_enabled = sum([
            phase3.get("use_chromosome_parallel_sort", False),
            phase3.get("use_parallel_vcf_parsing", False)
        ])

        return {
            "phase1_optimizations": phase1_enabled,
            "phase2_optimizations": phase2_enabled,
            "phase3_optimizations": phase3_enabled,
            "total_optimizations": phase1_enabled + phase2_enabled + phase3_enabled,
            "estimated_speedup": self._estimate_speedup(phase1_enabled, phase2_enabled, phase3_enabled)
        }

    def _estimate_speedup(self, p1: int, p2: int, p3: int) -> str:
        """Estimate total speedup based on enabled optimizations."""
        speedups = {
            (3, 2, 2): "4.3×",  # All optimizations
            (3, 2, 0): "3.3×",  # Phase 1+2
            (3, 0, 0): "1.9×",  # Phase 1 only
            (0, 0, 0): "1×"     # No optimizations
        }

        # Find closest match
        key = (p1, p2, p3)
        if key in speedups:
            return speedups[key]

        # Estimate
        if p1 == 3 and p2 >= 1:
            return "2.5-3.5×"
        elif p1 == 3:
            return "1.8-2×"
        else:
            return "1.2-1.5×"

    def print_report(self):
        """Print hardware detection report."""
        if not self.capabilities:
            self.detect_all()

        print("=" * 70)
        print("GenomeVault Hardware Capability Report")
        print("=" * 70)

        # System info
        print("\n📋 System Information")
        print(f"  OS: {self.capabilities['system']['platform']}")
        print(f"  Architecture: {self.capabilities['system']['machine']}")

        # CPU info
        cpu = self.capabilities["cpu"]
        print("\n🖥️  CPU")
        print(f"  Cores: {cpu['cores']}")
        print(f"  Apple Silicon: {'✅' if cpu['is_apple_silicon'] else '❌'}")
        if cpu.get("chip"):
            print(f"  Chip: {cpu['chip']}")
        print(f"  AMX Support: {'✅' if cpu['has_amx'] else '❌'}")
        print(f"  AVX2: {'✅' if cpu['has_avx2'] else '❌'}")

        # Memory info
        mem = self.capabilities["memory"]
        print("\n💾 Memory")
        print(f"  Total RAM: {mem['total_gb']} GB")
        print(f"  Recommended sambamba mem: {mem['recommended_sambamba_mem']}")

        # GPU info
        gpu = self.capabilities["gpu"]
        print("\n🎮 GPU")
        print(f"  Metal (Apple): {'✅' if gpu['has_metal'] else '❌'}")
        print(f"  CUDA (NVIDIA): {'✅' if gpu['has_cuda'] else '❌'}")
        print(f"  Recommended backend: {gpu['recommended_backend']}")

        # Storage info
        storage = self.capabilities["storage"]
        print("\n💿 Storage")
        print(f"  Type: {storage['type']}")
        print(f"  SSD: {'✅' if storage['is_ssd'] else '❌'}")
        print(f"  NVMe: {'✅' if storage['is_nvme'] else '❌'}")
        print(f"  Available: {storage['available_gb']} GB")

        # Tools info
        tools = self.capabilities["tools"]
        print("\n🛠️  Bioinformatics Tools")
        print(f"  sambamba: {'✅' if tools['sambamba'] else '❌ (install recommended)'}")
        print(f"  samtools: {'✅' if tools['samtools'] else '❌'}")
        print(f"  bcftools: {'✅' if tools['bcftools'] else '❌'}")
        print(f"  minimap2: {'✅' if tools['minimap2'] else '❌'}")
        print(f"  pigz: {'✅' if tools['pigz'] else '❌'}")

        # Recommendations
        recommendations = self.recommend_optimizations()

        print("\n" + "=" * 70)
        print("🎯 Optimization Recommendations")
        print("=" * 70)

        # Phase 1
        p1 = recommendations["phase1"]
        print("\n⭐⭐⭐ Phase 1 (Immediate Wins - 30 min, 5.6 hours saved)")
        print(f"  Sambamba sorting: {'✅ ENABLED' if p1['use_sambamba'] else '❌ DISABLED'}")
        if p1['use_sambamba']:
            print(f"    Threads: {p1['sambamba_threads']}, Memory: {p1['sambamba_memory']}")
        print(f"  Parallel BCFtools: {'✅ ENABLED' if p1['use_parallel_bcftools'] else '❌ DISABLED'}")
        if p1['use_parallel_bcftools']:
            print(f"    Threads: {p1['bcftools_threads']}")
        print(f"  Metal GPU HDC: {'✅ ENABLED' if p1['use_metal_gpu'] else '❌ DISABLED (CPU fallback)'}")

        # Phase 2
        p2 = recommendations["phase2"]
        print("\n⭐⭐ Phase 2 (High-Impact - 5 hours, 2.4 hours saved)")
        print(f"  Minimap2 index caching: ✅ ENABLED (always recommended)")
        print(f"  AMX alignment: {'✅ ENABLED' if p2['use_amx'] else '❌ DISABLED'}")
        if p2['use_amx']:
            print(f"    Chip: {p2.get('amx_chip', 'Unknown')}")
            print(f"    Expected speedup: {p2.get('expected_speedup', 'N/A')}")

        # Phase 3
        p3 = recommendations["phase3"]
        print("\n⭐ Phase 3 (Advanced - 8 hours, 2.1 hours saved)")
        print(f"  Chromosome-parallel sort: {'✅ ENABLED' if p3['use_chromosome_parallel_sort'] else '❌ DISABLED'}")
        if p3['use_chromosome_parallel_sort']:
            print(f"    Max parallel: {p3['max_parallel_chromosomes']} chromosomes")
        print(f"  Parallel VCF parsing: {'✅ ENABLED' if p3['use_parallel_vcf_parsing'] else '❌ DISABLED'}")
        if p3['use_parallel_vcf_parsing']:
            print(f"    Workers: {p3['vcf_workers']}")

        # Warnings
        all_warnings = []
        for phase in ["phase1", "phase2", "phase3"]:
            warnings = recommendations[phase].get("warnings", [])
            if isinstance(warnings, list):
                all_warnings.extend(warnings)
            elif isinstance(warnings, str):
                all_warnings.append(warnings)

        if all_warnings:
            print("\n⚠️  Warnings:")
            for warning in all_warnings:
                print(f"  - {warning}")

        # Summary
        summary = recommendations["summary"]
        print("\n" + "=" * 70)
        print("📊 Summary")
        print("=" * 70)
        print(f"  Total optimizations enabled: {summary['total_optimizations']}")
        print(f"  Estimated speedup: {summary['estimated_speedup']}")
        print(f"  Recommended action: Deploy Phase 1 immediately")
        print("=" * 70)


if __name__ == "__main__":
    # Run hardware detection and print report
    detector = HardwareDetector()
    detector.print_report()
