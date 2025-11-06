"""
Backend Auto-Selection for GenomeVault

Automatically selects the best available hardware backend for HDC operations:
- Metal (Apple Silicon): 43× speedup for batch encoding
- CUDA (NVIDIA GPU): 10-50× speedup for batch operations
- CPU (fallback): Always available

Based on benchmark results from docs/optimization/APPLE_SILICON_BENCHMARK_RESULTS.md
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_optimal_backend(prefer_gpu: bool = True, batch_size: int = 1):
    """
    Auto-select the best available backend for GenomeVault operations.

    Args:
        prefer_gpu: If True, prefer GPU backends when available (default: True)
        batch_size: Expected batch size (GPU only beneficial for batch_size >= 10)

    Returns:
        Backend instance (MetalBackend, CUDABackend, or CPUBackend)

    Selection Logic:
    - Small operations (batch_size < 10): Always use CPU (GPU overhead too high)
    - Apple Silicon + batch >= 10: Use Metal (43× speedup proven)
    - NVIDIA GPU + batch >= 10: Use CUDA (10-50× speedup)
    - Otherwise: Use CPU (fastest for small operations)
    """

    # For small batches, always use CPU (GPU transfer overhead dominates)
    if batch_size < 10:
        logger.debug(f"Using CPU for small batch (size={batch_size})")
        from genomevault.compute.cpu_backend import CPUBackend
        return CPUBackend()

    # Try Metal first (Apple Silicon)
    if prefer_gpu:
        try:
            from genomevault.compute.metal_backend import MetalBackend
            backend = MetalBackend()
            logger.info(f"🍎 Metal GPU selected (batch_size={batch_size}, expected 43× speedup)")
            return backend
        except (ImportError, RuntimeError) as e:
            logger.debug(f"Metal not available: {e}")

    # Try CUDA next (NVIDIA GPU)
    if prefer_gpu:
        try:
            from genomevault.compute.cuda_backend import CUDABackend
            backend = CUDABackend()
            logger.info(f"🚀 CUDA GPU selected (batch_size={batch_size}, expected 10-50× speedup)")
            return backend
        except (ImportError, RuntimeError) as e:
            logger.debug(f"CUDA not available: {e}")

    # Fallback to CPU
    logger.info(f"Using CPU backend (batch_size={batch_size})")
    from genomevault.compute.cpu_backend import CPUBackend
    return CPUBackend()


def detect_available_backends():
    """
    Detect all available hardware acceleration backends.

    Returns:
        dict: {backend_name: is_available}
    """
    backends = {
        "cpu": True,  # Always available
        "metal": False,
        "cuda": False,
    }

    # Check Metal (Apple Silicon)
    try:
        from genomevault.compute.metal_backend import MetalBackend
        _ = MetalBackend()
        backends["metal"] = True
        logger.debug("Metal backend available")
    except Exception:
        pass

    # Check CUDA (NVIDIA)
    try:
        from genomevault.compute.cuda_backend import CUDABackend
        _ = CUDABackend()
        backends["cuda"] = True
        logger.debug("CUDA backend available")
    except Exception:
        pass

    return backends


def get_backend_info():
    """
    Get detailed information about available backends.

    Returns:
        dict: Backend information and capabilities
    """
    available = detect_available_backends()

    info = {
        "available_backends": available,
        "recommended": None,
        "capabilities": {},
    }

    # Determine recommended backend
    if available["metal"]:
        info["recommended"] = "metal"
        info["capabilities"]["metal"] = {
            "batch_speedup": "43×",
            "best_for": "Batch encoding (100+ samples)",
            "device": "Apple Silicon GPU",
        }
    elif available["cuda"]:
        info["recommended"] = "cuda"
        info["capabilities"]["cuda"] = {
            "batch_speedup": "10-50×",
            "best_for": "Large batch operations",
            "device": "NVIDIA GPU",
        }
    else:
        info["recommended"] = "cpu"
        info["capabilities"]["cpu"] = {
            "batch_speedup": "1×",
            "best_for": "Small operations, bundling",
            "device": "CPU",
        }

    return info


def print_backend_info():
    """Print available backend information."""
    info = get_backend_info()

    print("\n" + "=" * 70)
    print("  GENOMEVAULT HARDWARE BACKEND INFORMATION")
    print("=" * 70)

    print(f"\nAvailable Backends:")
    for name, available in info["available_backends"].items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"  {name.upper():<10} {status}")

    print(f"\nRecommended Backend: {info['recommended'].upper()}")

    if info["capabilities"]:
        print(f"\nCapabilities:")
        for backend, caps in info["capabilities"].items():
            if backend in info["available_backends"] and info["available_backends"][backend]:
                print(f"\n  {backend.upper()}:")
                for key, value in caps.items():
                    print(f"    {key}: {value}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Print backend info when run directly
    print_backend_info()
