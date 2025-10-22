"""
Configuration Loader for GenomeVault Compute Backend

Loads and applies settings from compute.yaml and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from genomevault.compute.backend import ComputeBackend, initialize_backend

logger = logging.getLogger(__name__)


class ComputeConfig:
    """
    Configuration manager for compute backend settings

    Priority (highest to lowest):
    1. Environment variables
    2. Preset selection
    3. YAML config file
    4. Defaults
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Path to compute.yaml. If None, searches standard locations.
        """
        self.config_path = config_path or self._find_config()
        self.config = self._load_config()
        self._apply_env_overrides()
        self._apply_preset()

    def _find_config(self) -> Optional[Path]:
        """Search for compute.yaml in standard locations"""
        search_paths = [
            Path("genomevault/config/compute.yaml"),
            Path("config/compute.yaml"),
            Path(__file__).parent / "compute.yaml",
        ]

        for path in search_paths:
            if path.exists():
                logger.debug(f"Found config at: {path}")
                return path

        logger.warning("No compute.yaml found, using defaults")
        return None

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        if self.config_path is None or not self.config_path.exists():
            return self._default_config()

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded config from: {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if no file found"""
        return {
            'compute': {
                'default_backend': 'auto',
                'optimize_latency': False,
                'hdc_encoding': {
                    'single_sample': 'cpu',
                    'batch_threshold': 100,
                    'large_batch_warning': 1000,
                    'enable_faiss': True,
                    'faiss_threshold': 100000,
                },
                'similarity_search': {
                    'small_database_threshold': 10000,
                    'small_database_backend': 'cpu',
                    'large_database_backend': 'auto',
                    'faiss_cpu_threshold': 100000,
                },
                'zk_proofs': {
                    'backend': 'cpu',
                    'max_parallel': 4,
                    'allow_override': False,
                },
                'pir': {
                    'backend': 'cpu',
                    'enable_simd': True,
                    'sharding_threshold': 1000000,
                    'allow_override': False,
                },
            },
            'metal': {
                'use_unified_memory': True,
                'cache_kernels': True,
            },
            'cuda': {
                'device_id': 0,
                'use_pinned_memory': True,
                'memory_pool_mb': 1024,
            },
        }

    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Backend override
        env_backend = os.getenv('GENOMEVAULT_BACKEND')
        if env_backend:
            self.config['compute']['default_backend'] = env_backend.lower()
            logger.info(f"Backend override from env: {env_backend}")

        # Latency optimization
        env_latency = os.getenv('GENOMEVAULT_OPTIMIZE_LATENCY')
        if env_latency:
            self.config['compute']['optimize_latency'] = env_latency.lower() in ('true', '1', 'yes')
            logger.info(f"Latency optimization from env: {env_latency}")

        # CUDA device
        env_cuda_device = os.getenv('CUDA_VISIBLE_DEVICES')
        if env_cuda_device:
            try:
                self.config['cuda']['device_id'] = int(env_cuda_device.split(',')[0])
                logger.info(f"CUDA device from env: {env_cuda_device}")
            except (ValueError, IndexError):
                pass

    def _apply_preset(self):
        """Apply preset if specified"""
        preset_name = os.getenv('GENOMEVAULT_PRESET')
        if not preset_name:
            return

        if 'presets' not in self.config:
            logger.warning(f"Preset '{preset_name}' requested but no presets defined")
            return

        presets = self.config.get('presets', {})
        preset = presets.get(preset_name)

        if preset is None:
            logger.warning(f"Unknown preset: {preset_name}")
            return

        # Deep merge preset into config
        self._deep_merge(self.config['compute'], preset)
        logger.info(f"Applied preset: {preset_name}")

    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge update dict into base dict"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get_backend(self) -> ComputeBackend:
        """Get configured backend enum"""
        backend_str = self.config['compute']['default_backend'].lower()

        backend_map = {
            'auto': ComputeBackend.AUTO,
            'cpu': ComputeBackend.CPU,
            'metal': ComputeBackend.METAL,
            'cuda': ComputeBackend.CUDA,
        }

        backend = backend_map.get(backend_str)
        if backend is None:
            logger.warning(f"Unknown backend '{backend_str}', using AUTO")
            backend = ComputeBackend.AUTO

        return backend

    def initialize_backend(self) -> ComputeBackend:
        """Initialize backend based on configuration"""
        backend = self.get_backend()
        actual_backend = initialize_backend(backend)
        logger.info(f"Initialized backend: {actual_backend.value}")
        return actual_backend

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key

        Example:
            config.get('compute.hdc_encoding.batch_threshold')
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_hdc_batch_threshold(self) -> int:
        """Get batch threshold for HDC encoding"""
        return self.get('compute.hdc_encoding.batch_threshold', 100)

    def get_hdc_warning_threshold(self) -> int:
        """Get warning threshold for large CPU batches"""
        return self.get('compute.hdc_encoding.large_batch_warning', 1000)

    def should_use_faiss(self) -> bool:
        """Check if FAISS should be enabled"""
        return self.get('compute.hdc_encoding.enable_faiss', True)

    def get_faiss_threshold(self) -> int:
        """Get threshold for FAISS indexing"""
        return self.get('compute.hdc_encoding.faiss_threshold', 100000)

    def get_cuda_device_id(self) -> int:
        """Get CUDA device ID"""
        return self.get('cuda.device_id', 0)

    def is_latency_optimized(self) -> bool:
        """Check if optimizing for latency (CPU-only)"""
        return self.get('compute.optimize_latency', False)

    def print_config(self):
        """Print current configuration"""
        print("=" * 60)
        print("GenomeVault Compute Configuration")
        print("=" * 60)
        print(f"Config file: {self.config_path or 'defaults'}")
        print(f"\nBackend: {self.get_backend().value}")
        print(f"Optimize latency: {self.is_latency_optimized()}")
        print(f"\nHDC Settings:")
        print(f"  Batch threshold: {self.get_hdc_batch_threshold()}")
        print(f"  Warning threshold: {self.get_hdc_warning_threshold()}")
        print(f"  FAISS enabled: {self.should_use_faiss()}")
        print(f"  FAISS threshold: {self.get_faiss_threshold()}")
        print(f"\nZK Proofs: cpu (fixed)")
        print(f"PIR: cpu (fixed)")
        print("=" * 60)


# Global configuration instance
_global_config: Optional[ComputeConfig] = None


def get_config() -> ComputeConfig:
    """
    Get global configuration instance (singleton)

    Returns:
        ComputeConfig instance

    Example:
        >>> from genomevault.config.loader import get_config
        >>> config = get_config()
        >>> backend = config.get_backend()
    """
    global _global_config

    if _global_config is None:
        _global_config = ComputeConfig()

    return _global_config


def load_and_initialize() -> ComputeBackend:
    """
    Convenience function to load config and initialize backend

    Returns:
        Initialized ComputeBackend

    Example:
        >>> from genomevault.config.loader import load_and_initialize
        >>> backend = load_and_initialize()
        >>> print(f"Using: {backend.value}")
    """
    config = get_config()
    return config.initialize_backend()


if __name__ == "__main__":
    # Demo: Show current configuration
    config = get_config()
    config.print_config()

    # Initialize backend
    backend = config.initialize_backend()
    print(f"\n✓ Initialized backend: {backend.value}")
