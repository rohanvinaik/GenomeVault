"""Hardware backend detection and management."""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class AcceleratorType(Enum):
    """Types of hardware accelerators."""
    CPU = "cpu"
    CUDA = "cuda"  # NVIDIA GPUs
    METAL = "metal"  # Apple Silicon
    ROCM = "rocm"  # AMD GPUs
    ONEAPI = "oneapi"  # Intel GPUs
    TPU = "tpu"  # Google TPUs
    CLOUD_GPU = "cloud_gpu"  # Cloud-based GPUs
    FPGA = "fpga"  # Field-programmable gate arrays


@dataclass
class HardwareBackend:
    """Hardware backend information."""
    type: AcceleratorType
    name: str
    available: bool
    memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    preferred_precision: str = "float32"
    max_threads: Optional[int] = None
    
    def __str__(self) -> str:
        status = "✓" if self.available else "✗"
        mem_str = f", {self.memory_gb:.1f}GB" if self.memory_gb else ""
        return f"[{status}] {self.name} ({self.type.value}{mem_str})"


def detect_cpu_features() -> HardwareBackend:
    """Detect CPU capabilities."""
    import platform
    import multiprocessing
    
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        memory_gb = None
    
    return HardwareBackend(
        type=AcceleratorType.CPU,
        name=platform.processor() or "Unknown CPU",
        available=True,
        memory_gb=memory_gb,
        max_threads=multiprocessing.cpu_count(),
        preferred_precision="float32"
    )


def detect_cuda() -> Optional[HardwareBackend]:
    """Detect NVIDIA CUDA GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return HardwareBackend(
                type=AcceleratorType.CUDA,
                name=torch.cuda.get_device_name(0),
                available=True,
                memory_gb=torch.cuda.get_device_properties(0).total_memory / (1024**3),
                compute_capability=f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
                preferred_precision="float16" if torch.cuda.get_device_capability(0)[0] >= 7 else "float32"
            )
    except ImportError:
        pass
    
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            cp.cuda.Device(0).use()
            return HardwareBackend(
                type=AcceleratorType.CUDA,
                name=cp.cuda.Device(0).name.decode(),
                available=True,
                memory_gb=cp.cuda.Device(0).mem_info[1] / (1024**3),
                preferred_precision="float32"
            )
    except ImportError:
        pass
    
    return None


def detect_metal() -> Optional[HardwareBackend]:
    """Detect Apple Metal GPU."""
    try:
        import mlx.core as mx
        return HardwareBackend(
            type=AcceleratorType.METAL,
            name="Apple Silicon GPU",
            available=True,
            compute_capability="M1/M2/M3",
            preferred_precision="float32"
        )
    except ImportError:
        pass
    
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return HardwareBackend(
                type=AcceleratorType.METAL,
                name="Apple Metal Performance Shaders",
                available=True,
                preferred_precision="float32"
            )
    except ImportError:
        pass
    
    return None


def detect_rocm() -> Optional[HardwareBackend]:
    """Detect AMD ROCm GPUs."""
    try:
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            return HardwareBackend(
                type=AcceleratorType.ROCM,
                name="AMD GPU (ROCm)",
                available=True,
                preferred_precision="float32"
            )
    except ImportError:
        pass
    
    return None


def detect_tpu() -> Optional[HardwareBackend]:
    """Detect Google TPUs."""
    try:
        import jax
        devices = jax.devices()
        for device in devices:
            if 'tpu' in str(device).lower():
                return HardwareBackend(
                    type=AcceleratorType.TPU,
                    name=str(device),
                    available=True,
                    preferred_precision="bfloat16"
                )
    except ImportError:
        pass
    
    return None


def list_available_accelerators() -> List[HardwareBackend]:
    """List all available hardware accelerators."""
    accelerators = []
    
    # Always have CPU
    accelerators.append(detect_cpu_features())
    
    # Check GPUs
    cuda = detect_cuda()
    if cuda:
        accelerators.append(cuda)
    
    metal = detect_metal()
    if metal:
        accelerators.append(metal)
    
    rocm = detect_rocm()
    if rocm:
        accelerators.append(rocm)
    
    tpu = detect_tpu()
    if tpu:
        accelerators.append(tpu)
    
    return accelerators


def get_best_accelerator(
    preferred_type: Optional[AcceleratorType] = None,
    min_memory_gb: Optional[float] = None
) -> HardwareBackend:
    """
    Get the best available accelerator.
    
    Args:
        preferred_type: Preferred accelerator type
        min_memory_gb: Minimum required memory
        
    Returns:
        Best available hardware backend
    """
    accelerators = list_available_accelerators()
    
    # Filter by availability
    available = [a for a in accelerators if a.available]
    
    # Filter by memory if specified
    if min_memory_gb:
        available = [a for a in available if a.memory_gb and a.memory_gb >= min_memory_gb]
    
    # Prefer requested type
    if preferred_type:
        for acc in available:
            if acc.type == preferred_type:
                return acc
    
    # Priority order: TPU > CUDA > Metal > ROCm > CPU
    priority = [
        AcceleratorType.TPU,
        AcceleratorType.CUDA,
        AcceleratorType.METAL,
        AcceleratorType.ROCM,
        AcceleratorType.CPU
    ]
    
    for acc_type in priority:
        for acc in available:
            if acc.type == acc_type:
                return acc
    
    # Fallback to CPU
    return detect_cpu_features()


def get_accelerator_info() -> Dict[str, Any]:
    """Get detailed information about all accelerators."""
    accelerators = list_available_accelerators()
    
    info = {
        "available_count": sum(1 for a in accelerators if a.available),
        "accelerators": []
    }
    
    for acc in accelerators:
        acc_info = {
            "type": acc.type.value,
            "name": acc.name,
            "available": acc.available,
            "memory_gb": acc.memory_gb,
            "compute_capability": acc.compute_capability,
            "preferred_precision": acc.preferred_precision,
            "max_threads": acc.max_threads
        }
        info["accelerators"].append(acc_info)
    
    # Determine best accelerator
    best = get_best_accelerator()
    info["recommended"] = best.type.value
    
    return info