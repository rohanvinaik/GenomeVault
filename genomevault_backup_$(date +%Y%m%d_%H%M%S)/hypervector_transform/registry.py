"""
Hypervector Registry for encoding version management.

Manages hypervector encoding versions for reproducibility and compatibility.
This is a core component for Stage 1 of the HDC implementation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import logging

from numpy.typing import NDArray
import numpy as np
import torch

from .hdc_encoder import (
    HypervectorConfig,
    HypervectorEncoder,
    ProjectionType,
)
from genomevault.core.constants import OmicsType

logger = logging.getLogger(__name__)


class HypervectorRegistry:
    """
    Manage hypervector encoding versions and parameters.

    Ensures reproducibility by tracking:
    - Encoding parameters (dimension, projection type)
    - Random seeds
    - Version history
    - Compatibility mappings
    """

    def __init__(self, registry_path: str = "./hypervector_registry.json"):
        """
        Initialize registry.

        Args:
            registry_path: Path to registry file
        """
        self.registry_path = Path(registry_path)
        self.versions = {}
        self.current_version = None
        self.compatibility_map = {}
        self.performance_metrics = {}

        # Load existing registry
        if self.registry_path.exists():
            self._load_registry()
        else:
            self._initialize_registry()

    def _initialize_registry(self):
        """Initialize new registry with default versions matching spec."""
        # Default version for Stage 1
        default_version = "v1.0.0"

        # Production-ready configurations
        configs = [
            (
                default_version,
                {
                    "dimension": 10000,
                    "projection_type": "sparse_random",
                    "seed": 42,
                    "sparsity": 0.1,
                },
                "Default HDC encoding parameters - Clinical tier",
                "clinical",
            ),
            (
                "mini_v1.0.0",
                {
                    "dimension": 5000,
                    "projection_type": "sparse_random",
                    "seed": 42,
                    "sparsity": 0.05,
                },
                "Mini tier for most-studied SNPs",
                "mini",
            ),
            (
                "full_v1.0.0",
                {
                    "dimension": 20000,
                    "projection_type": "sparse_random",
                    "seed": 42,
                    "sparsity": 0.1,
                },
                "Full tier for comprehensive analysis",
                "full",
            ),
        ]

        self.versions = {}
        for version, params, description, tier in configs:
            fingerprint = self._generate_fingerprint(params)
            self.versions[version] = {
                "params": params,
                "fingerprint": fingerprint,
                "created_at": datetime.now().isoformat(),
                "description": description,
                "tier": tier,
            }

        self.current_version = default_version
        self._save_registry()

        logger.info(f"Initialized hypervector registry with version {default_version}")

    def _load_registry(self):
        """Load registry from file."""
        try:
            with open(self.registry_path) as f:
                data = json.load(f)

            self.versions = data.get("versions", {})
            self.current_version = data.get("current_version")
            self.compatibility_map = data.get("compatibility_map", {})
            self.performance_metrics = data.get("performance_metrics", {})

            logger.info(f"Loaded registry with {len(self.versions)} versions")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self._initialize_registry()

    def _save_registry(self):
        """Save registry to file with backup."""
        data = {
            "versions": self.versions,
            "current_version": self.current_version,
            "compatibility_map": self.compatibility_map,
            "performance_metrics": self.performance_metrics,
            "last_updated": datetime.now().isoformat(),
        }

        # Create backup if file exists
        backup_path = None
        if self.registry_path.exists():
            backup_path = self.registry_path.with_suffix(".backup.json")
            self.registry_path.rename(backup_path)

        try:
            # Write with pretty formatting
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)

            # Remove backup on success
            if backup_path and backup_path.exists():
                backup_path.unlink()

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            # Restore backup
            if backup_path and backup_path.exists():
                backup_path.rename(self.registry_path)
            raise RuntimeError("Failed to save registry")

    def register_version(
        self,
        version: str,
        params: dict[str, Any],
        description: str = "",
        force: bool = False,
        performance_data: dict[str, float] | None = None,
    ):
        """
        Register new encoding version.

        Args:
            version: Version identifier (e.g., "v2.0.0")
            params: Encoding parameters
            description: Version description
            force: Overwrite if version exists
            performance_data: Optional performance metrics
        """
        if version in self.versions and not force:
            raise ValueError(f"Version {version} already exists")

        # Validate parameters
        required_params = ["dimension", "projection_type"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")

        # Validate dimension constraints
        if not 1000 <= params["dimension"] <= 100000:
            raise ValueError("Dimension must be between 1000 and 100000")

        # Add seed if not provided
        if "seed" not in params:
            params["seed"] = np.random.randint(2**32)

        # Generate fingerprint
        fingerprint = self._generate_fingerprint(params)

        # Determine tier based on dimension
        tier = "custom"
        if params["dimension"] <= 5000:
            tier = "mini"
        elif params["dimension"] <= 10000:
            tier = "clinical"
        elif params["dimension"] <= 20000:
            tier = "full"

        self.versions[version] = {
            "params": params,
            "fingerprint": fingerprint,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "tier": tier,
        }

        # Add performance data if provided
        if performance_data:
            self.performance_metrics[version] = performance_data

        self._save_registry()

        logger.info(f"Registered version {version} with fingerprint {fingerprint}")

    def create_encoder(self, version: str) -> HypervectorEncoder:
        """
        Create encoder from version parameters.

        Args:
            version: Version identifier

        Returns:
            Configured HypervectorEncoder
        """
        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        params = self.versions[version]["params"]

        # Create encoder configuration
        config = HypervectorConfig(
            dimension=params["dimension"],
            projection_type=ProjectionType(params["projection_type"]),
            seed=params.get("seed", 42),
            sparsity=params.get("sparsity", 0.1),
            normalize=params.get("normalize", True),
            quantize=params.get("quantize", False),
            quantization_bits=params.get("quantization_bits", 8),
        )

        # Create encoder
        encoder = HypervectorEncoder(config)

        # Tag encoder with version metadata
        encoder.version = version
        encoder.fingerprint = self.versions[version].get(
            "fingerprint", self._generate_fingerprint(params)
        )

        return encoder

    def get_encoder(self, version: str | None = None) -> HypervectorEncoder:
        """
        Get encoder with specific version parameters.

        Args:
            version: Version to use (current if None)

        Returns:
            Configured encoder
        """
        if version is None:
            version = self.current_version

        return self.create_encoder(version)

    def get_current_encoder(self) -> HypervectorEncoder:
        """
        Get encoder with current default version.

        Returns:
            Configured encoder with current version
        """
        if self.current_version is None:
            raise ValueError("No current version set")
        return self.create_encoder(self.current_version)

    def set_current_version(self, version: str):
        """Set current default version."""
        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        old_version = self.current_version
        self.current_version = version
        self._save_registry()

        logger.info(f"Changed current version from {old_version} to {version}")

    def add_compatibility(
        self,
        version1: str,
        version2: str,
        transform_function: str | None = None,
        bidirectional: bool = True,
    ):
        """
        Mark two versions as compatible.

        Args:
            version1: First version
            version2: Second version
            transform_function: Optional transform function name
            bidirectional: Whether compatibility is bidirectional
        """
        if version1 not in self.versions or version2 not in self.versions:
            raise ValueError("Both versions must exist")

        key = f"{version1}<->{version2}"
        reverse_key = f"{version2}<->{version1}"

        compatibility_info = {
            "compatible": True,
            "transform": transform_function,
            "bidirectional": bidirectional,
            "added_at": datetime.now().isoformat(),
        }

        self.compatibility_map[key] = compatibility_info

        if bidirectional:
            self.compatibility_map[reverse_key] = compatibility_info

        self._save_registry()

    def check_compatibility(self, version1: str, version2: str) -> dict[str, Any]:
        """
        Check if two versions are compatible.

        Returns:
            Compatibility information
        """
        # Check explicit mappings
        key1 = f"{version1}<->{version2}"
        key2 = f"{version2}<->{version1}"

        if key1 in self.compatibility_map:
            return self.compatibility_map[key1]
        elif key2 in self.compatibility_map:
            return self.compatibility_map[key2]
        else:
            # Check parameter compatibility
            params1 = self.versions[version1]["params"]
            params2 = self.versions[version2]["params"]

            # Dimensions must match for direct compatibility
            dimension_match = params1["dimension"] == params2["dimension"]

            # Projection types should match for best compatibility
            projection_match = params1["projection_type"] == params2["projection_type"]

            return {
                "compatible": dimension_match,
                "transform": None,
                "dimension_match": dimension_match,
                "projection_match": projection_match,
                "reason": "No explicit compatibility mapping",
                "recommendation": (
                    "Use VersionMigrator for conversion" if not dimension_match else None
                ),
            }

    def _generate_fingerprint(self, params: dict[str, Any]) -> str:
        """Generate unique fingerprint for parameters."""
        # Create stable string representation
        param_str = json.dumps(
            {
                "dimension": params.get("dimension"),
                "projection_type": params.get("projection_type"),
                "seed": params.get("seed"),
                "sparsity": params.get("sparsity", 0.1),
            },
            sort_keys=True,
        )

        # Generate SHA256 hash
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def list_versions(self) -> list[dict[str, Any]]:
        """List all registered versions with details."""
        versions = []

        for version, info in self.versions.items():
            version_info = {
                "version": version,
                "dimension": info["params"]["dimension"],
                "projection_type": info["params"]["projection_type"],
                "tier": info.get("tier", "custom"),
                "created_at": info["created_at"],
                "is_current": version == self.current_version,
                "fingerprint": info["fingerprint"],
                "description": info.get("description", ""),
            }

            # Add performance metrics if available
            if version in self.performance_metrics:
                version_info["performance"] = self.performance_metrics[version]

            versions.append(version_info)

        return sorted(versions, key=lambda x: x["created_at"], reverse=True)

    def export_version(self, version: str, filepath: str):
        """Export version configuration for sharing."""
        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        export_data = {
            "version": version,
            "params": self.versions[version]["params"],
            "fingerprint": self.versions[version]["fingerprint"],
            "description": self.versions[version].get("description", ""),
            "tier": self.versions[version].get("tier", "custom"),
            "exported_at": datetime.now().isoformat(),
            "registry_version": "1.0",
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported version {version} to {filepath}")

    def import_version(self, filepath: str, version: str | None = None, force: bool = False):
        """Import version configuration from file."""
        with open(filepath) as f:
            data = json.load(f)

        # Use provided version or from file
        if version is None:
            version = data["version"]

        # Validate registry version compatibility
        if data.get("registry_version", "1.0") != "1.0":
            logger.warning(f"Registry version mismatch: {data.get('registry_version')} != 1.0")

        self.register_version(
            version=version,
            params=data["params"],
            description=data.get("description", f"Imported from {filepath}"),
            force=force,
        )

    def add_version(
        self,
        version: str,
        dimension: int,
        projection_type: str = "sparse_random",
        seed: int = 42,
        sparsity: float = 0.1,
        description: str = "",
        tier: str | None = None,
    ):
        """
        Add a new version with specified parameters (convenience method).

        Args:
            version: Version identifier
            dimension: Hypervector dimension
            projection_type: Type of projection
            seed: Random seed
            sparsity: Sparsity level
            description: Version description
            tier: Optional tier designation
        """
        params = {
            "dimension": dimension,
            "projection_type": projection_type,
            "seed": seed,
            "sparsity": sparsity,
        }

        # Auto-determine tier if not provided
        if tier is None:
            if dimension <= 5000:
                tier = "mini"
            elif dimension <= 10000:
                tier = "clinical"
            elif dimension <= 20000:
                tier = "full"
            else:
                tier = "custom"

        self.register_version(version, params, description or f"{tier} tier configuration")

    def export_json(self, filepath: str):
        """
        Export entire registry to JSON file.

        Args:
            filepath: Path to export file
        """
        data = {
            "versions": self.versions,
            "current_version": self.current_version,
            "compatibility_map": self.compatibility_map,
            "performance_metrics": self.performance_metrics,
            "exported_at": datetime.now().isoformat(),
            "registry_version": "1.0",
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

        logger.info(f"Exported registry to {filepath}")

    def import_json(self, filepath: str, merge: bool = False):
        """
        Import registry from JSON file.

        Args:
            filepath: Path to import file
            merge: If True, merge with existing registry; if False, replace
        """
        with open(filepath) as f:
            data = json.load(f)

        if not merge:
            # Replace entire registry
            self.versions = data.get("versions", {})
            self.current_version = data.get("current_version")
            self.compatibility_map = data.get("compatibility_map", {})
            self.performance_metrics = data.get("performance_metrics", {})
        else:
            # Merge with existing
            self.versions.update(data.get("versions", {}))
            self.compatibility_map.update(data.get("compatibility_map", {}))
            self.performance_metrics.update(data.get("performance_metrics", {}))

        self._save_registry()
        logger.info(f"Imported registry from {filepath} (merge={merge})")

    def get_version_info(self, version: str) -> dict[str, Any]:
        """Get detailed information about a specific version."""
        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        info = self.versions[version].copy()
        info["version"] = version
        info["is_current"] = version == self.current_version

        # Add compatibility info
        compatible_versions = []
        for key in self.compatibility_map:
            if version in key:
                other_version = key.replace(version, "").replace("<->", "")
                compatible_versions.append(other_version)

        info["compatible_versions"] = compatible_versions

        return info

    def benchmark_version(self, version: str, num_samples: int = 100) -> dict[str, float]:
        """Benchmark a specific version's performance."""
        import time

        encoder = self.get_encoder(version)

        # Test encoding speed
        features = np.random.randn(1000)

        # Warm-up
        for _ in range(10):
            _ = encoder.encode(features, OmicsType.GENOMIC)

        # Benchmark
        start = time.time()
        for _ in range(num_samples):
            _ = encoder.encode(features, OmicsType.GENOMIC)
        elapsed = time.time() - start

        metrics = {
            "encoding_time_ms": (elapsed / num_samples) * 1000,
            "throughput_ops_per_sec": num_samples / elapsed,
            "dimension": encoder.config.dimension,
            "memory_per_vector_kb": encoder.config.dimension * 4 / 1024,  # float32
        }

        # Store metrics
        self.performance_metrics[version] = metrics
        self._save_registry()

        return metrics


class VersionMigrator:
    """
    Handle migrations between hypervector encoding versions.
    Implements Stage 1 requirement for version compatibility.
    """

    def __init__(self, registry: HypervectorRegistry):
        """
        Initialize migrator.

        Args:
            registry: Hypervector registry
        """
        self.registry = registry
        self.migration_functions = {
            "dimension_reduction": self._migrate_dimension_reduction,
            "dimension_expansion": self._migrate_dimension_expansion,
            "projection_change": self._migrate_projection_change,
        }

    def migrate_hypervector(
        self,
        hv: torch.Tensor,
        from_version: str,
        to_version: str,
        preserve_norm: bool = True,
    ) -> torch.Tensor:
        """
        Migrate hypervector between versions.

        Args:
            hv: Hypervector to migrate
            from_version: Source version
            to_version: Target version
            preserve_norm: Whether to preserve vector norm

        Returns:
            Migrated hypervector
        """
        # Check if migration needed
        if from_version == to_version:
            return hv

        # Check compatibility
        compat = self.registry.check_compatibility(from_version, to_version)

        if compat["compatible"] and compat.get("transform") is None:
            # Directly compatible
            return hv

        # Get parameters
        from_params = self.registry.versions[from_version]["params"]
        to_params = self.registry.versions[to_version]["params"]

        # Store original norm if preserving
        original_norm = torch.norm(hv) if preserve_norm else None

        # Determine migration type and execute
        if from_params["dimension"] != to_params["dimension"]:
            if from_params["dimension"] > to_params["dimension"]:
                result = self._migrate_dimension_reduction(hv, from_params, to_params)
            else:
                result = self._migrate_dimension_expansion(hv, from_params, to_params)
        elif from_params["projection_type"] != to_params["projection_type"]:
            result = self._migrate_projection_change(hv, from_params, to_params)
        else:
            # Same structure, just ensure compatibility
            result = hv.clone()

        # Restore norm if requested
        if preserve_norm and original_norm is not None:
            result = result / torch.norm(result) * original_norm

        logger.info(f"Migrated hypervector from {from_version} to {to_version}")
        return result

    def _migrate_dimension_reduction(
        self, hv: torch.Tensor, from_params: dict[str, Any], to_params: dict[str, Any]
    ) -> torch.Tensor:
        """Reduce hypervector dimension using stable projection."""
        from_dim = from_params["dimension"]
        to_dim = to_params["dimension"]

        # Use seed for reproducible projection
        rng = np.random.RandomState(to_params.get("seed", 42))

        # Create projection matrix using Johnson-Lindenstrauss
        projection: NDArray[np.float32] = (rng.randn(to_dim, from_dim) / np.sqrt(from_dim)).astype(
            np.float32
        )
        projection_tensor = torch.from_numpy(projection.astype(np.float32, copy=False))

        # Project to lower dimension
        reduced = torch.matmul(projection_tensor, hv)

        return reduced

    def _migrate_dimension_expansion(
        self, hv: torch.Tensor, from_params: dict[str, Any], to_params: dict[str, Any]
    ) -> torch.Tensor:
        """Expand hypervector dimension with controlled padding."""
        from_dim = from_params["dimension"]
        to_dim = to_params["dimension"]

        # Use seed for reproducible expansion
        rng = np.random.RandomState(to_params.get("seed", 42))

        # Create expanded vector
        expanded = torch.zeros(to_dim)

        # Copy original values
        expanded[:from_dim] = hv

        # Fill remaining dimensions with correlated noise
        # This preserves some structure while expanding
        if to_dim > from_dim:
            # Generate structured noise based on existing values
            noise_dim = to_dim - from_dim

            # Use random projection of original vector for correlation
            projection: NDArray[np.float32] = (rng.randn(noise_dim, from_dim) * 0.1).astype(
                np.float32
            )
            projection_tensor = torch.from_numpy(projection.astype(np.float32, copy=False))

            correlated_noise = torch.matmul(projection_tensor, hv)
            expanded[from_dim:] = correlated_noise

        return expanded

    def _migrate_projection_change(
        self, hv: torch.Tensor, from_params: dict[str, Any], to_params: dict[str, Any]
    ) -> torch.Tensor:
        """
        Change projection type (approximate).
        This is necessarily lossy as we cannot recover the original data.
        """
        to_type = to_params["projection_type"]

        if to_type == "binary":
            # Convert to binary representation
            return (hv > hv.median()).float() * 2 - 1

        elif to_type == "sparse_random":
            # Sparsify by soft thresholding
            sparsity = to_params.get("sparsity", 0.1)
            threshold = torch.quantile(torch.abs(hv), 1 - sparsity)
            sparse = hv.clone()
            sparse[torch.abs(sparse) < threshold] = 0
            return sparse

        elif to_type == "gaussian":
            # Normalize to gaussian-like distribution
            # Standardize
            mean = hv.mean()
            std = hv.std()
            return (hv - mean) / (std + 1e-8)

        else:
            # Unknown type, return as-is
            logger.warning(f"Unknown projection type: {to_type}")
            return hv

    def create_migration_report(
        self, from_version: str, to_version: str, test_vectors: int = 100
    ) -> dict[str, Any]:
        """Create detailed migration report between versions."""
        report = {
            "from_version": from_version,
            "to_version": to_version,
            "timestamp": datetime.now().isoformat(),
            "compatibility": self.registry.check_compatibility(from_version, to_version),
            "tests": {},
        }

        # Get encoders
        from_encoder = self.registry.get_encoder(from_version)
        to_encoder = self.registry.get_encoder(to_version)

        # Test migration quality
        similarities = []
        for _ in range(test_vectors):
            # Create random test vector
            original = torch.randn(from_encoder.config.dimension)
            original = original / torch.norm(original)

            # Migrate
            migrated = self.migrate_hypervector(original, from_version, to_version)

            # Migrate back if possible
            if from_encoder.config.dimension == to_encoder.config.dimension:
                back_migrated = self.migrate_hypervector(migrated, to_version, from_version)
                similarity = torch.nn.functional.cosine_similarity(
                    original.unsqueeze(0), back_migrated.unsqueeze(0)
                ).item()
                similarities.append(similarity)

        if similarities:
            report["tests"]["round_trip_similarity"] = {
                "mean": np.mean(similarities),
                "std": np.std(similarities),
                "min": np.min(similarities),
                "max": np.max(similarities),
            }

        return report
