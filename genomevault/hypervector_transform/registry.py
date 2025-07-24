"""
Hypervector Registry for encoding version management.

Manages hypervector encoding versions for reproducibility and compatibility.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from genomevault.hypervector_transform.encoding import HypervectorEncoder

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

        # Load existing registry
        if self.registry_path.exists():
            self._load_registry()
        else:
            self._initialize_registry()

    def _initialize_registry(self):
        """Initialize new registry with default version."""
        default_version = "v1.0.0"

        self.versions[default_version] = {
            "params": {
                "dimension": 10000,
                "projection_type": "gaussian",
                "seed": 42,
                "sparsity": 0.01,
            },
            "created_at": datetime.now().isoformat(),
            "description": "Default hypervector encoding parameters",
        }

        self.current_version = default_version
        self._save_registry()

        logger.info(f"Initialized hypervector registry with version {default_version}")

    def _load_registry(self):
        """Load registry from file."""
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)

            self.versions = data.get("versions", {})
            self.current_version = data.get("current_version")
            self.compatibility_map = data.get("compatibility_map", {})

            logger.info(f"Loaded registry with {len(self.versions)} versions")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self._initialize_registry()

    def _save_registry(self):
        """Save registry to file."""
        data = {
            "versions": self.versions,
            "current_version": self.current_version,
            "compatibility_map": self.compatibility_map,
            "last_updated": datetime.now().isoformat(),
        }

        # Create backup
        if self.registry_path.exists():
            backup_path = self.registry_path.with_suffix(".backup.json")
            self.registry_path.rename(backup_path)

        try:
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            # Restore backup
            if backup_path.exists():
                backup_path.rename(self.registry_path)
            raise

    def register_version(
        self, version: str, params: Dict[str, Any], description: str = "", force: bool = False
    ):
        """
        Register new encoding version.

        Args:
            version: Version identifier (e.g., "v2.0.0")
            params: Encoding parameters
            description: Version description
            force: Overwrite if version exists
        """
        if version in self.versions and not force:
            raise ValueError(f"Version {version} already exists")

        # Validate parameters
        required_params = ["dimension", "projection_type"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")

        # Add seed if not provided
        if "seed" not in params:
            params["seed"] = np.random.randint(2**32)

        # Generate fingerprint
        fingerprint = self._generate_fingerprint(params)

        self.versions[version] = {
            "params": params,
            "fingerprint": fingerprint,
            "created_at": datetime.now().isoformat(),
            "description": description,
        }

        self._save_registry()

        logger.info(f"Registered version {version} with fingerprint {fingerprint}")

    def get_encoder(self, version: Optional[str] = None) -> HypervectorEncoder:
        """
        Get encoder with specific version parameters.

        Args:
            version: Version to use (current if None)

        Returns:
            Configured encoder
        """
        if version is None:
            version = self.current_version

        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        params = self.versions[version]["params"]

        # Create encoder with versioned parameters
        encoder = HypervectorEncoder(dimension=params["dimension"], seed=params.get("seed", 42))

        # Apply additional parameters
        if "projection_type" in params:
            encoder.projection_type = params["projection_type"]

        if "sparsity" in params:
            encoder.sparsity = params["sparsity"]

        # Tag encoder with version
        encoder.version = version
        encoder.fingerprint = self.versions[version]["fingerprint"]

        return encoder

    def set_current_version(self, version: str):
        """Set current default version."""
        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        self.current_version = version
        self._save_registry()

        logger.info(f"Set current version to {version}")

    def add_compatibility(
        self, version1: str, version2: str, transform_function: Optional[str] = None
    ):
        """
        Mark two versions as compatible.

        Args:
            version1: First version
            version2: Second version
            transform_function: Optional transform function name
        """
        if version1 not in self.versions or version2 not in self.versions:
            raise ValueError("Both versions must exist")

        key = f"{version1}<->{version2}"

        self.compatibility_map[key] = {
            "compatible": True,
            "transform": transform_function,
            "added_at": datetime.now().isoformat(),
        }

        self._save_registry()

    def check_compatibility(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Check if two versions are compatible.

        Returns:
            Compatibility information
        """
        key1 = f"{version1}<->{version2}"
        key2 = f"{version2}<->{version1}"

        if key1 in self.compatibility_map:
            return self.compatibility_map[key1]
        elif key2 in self.compatibility_map:
            return self.compatibility_map[key2]
        else:
            # Check if parameters are compatible
            params1 = self.versions[version1]["params"]
            params2 = self.versions[version2]["params"]

            compatible = (
                params1["dimension"] == params2["dimension"]
                and params1["projection_type"] == params2["projection_type"]
            )

            return {
                "compatible": compatible,
                "transform": None,
                "reason": "No explicit compatibility mapping",
            }

    def _generate_fingerprint(self, params: Dict[str, Any]) -> str:
        """Generate unique fingerprint for parameters."""
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)

        return hashlib.sha256(sorted_params.encode()).hexdigest()[:16]

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all registered versions."""
        versions = []

        for version, info in self.versions.items():
            versions.append(
                {
                    "version": version,
                    "dimension": info["params"]["dimension"],
                    "projection_type": info["params"]["projection_type"],
                    "created_at": info["created_at"],
                    "is_current": version == self.current_version,
                    "fingerprint": info["fingerprint"],
                }
            )

        return sorted(versions, key=lambda x: x["created_at"], reverse=True)

    def export_version(self, version: str, filepath: str):
        """Export version configuration."""
        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        export_data = {
            "version": version,
            "params": self.versions[version]["params"],
            "fingerprint": self.versions[version]["fingerprint"],
            "exported_at": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported version {version} to {filepath}")

    def import_version(self, filepath: str, version: Optional[str] = None):
        """Import version configuration."""
        with open(filepath, "r") as f:
            data = json.load(f)

        if version is None:
            version = data["version"]

        self.register_version(
            version=version,
            params=data["params"],
            description=f"Imported from {filepath}",
            force=True,
        )


class VersionMigrator:
    """
    Handle migrations between hypervector encoding versions.
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

    def migrate_hypervector(self, hv: np.ndarray, from_version: str, to_version: str) -> np.ndarray:
        """
        Migrate hypervector between versions.

        Args:
            hv: Hypervector to migrate
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated hypervector
        """
        # Check compatibility
        compat = self.registry.check_compatibility(from_version, to_version)

        if compat["compatible"] and compat.get("transform") is None:
            # Directly compatible
            return hv

        # Get parameters
        from_params = self.registry.versions[from_version]["params"]
        to_params = self.registry.versions[to_version]["params"]

        # Determine migration type
        if from_params["dimension"] != to_params["dimension"]:
            if from_params["dimension"] > to_params["dimension"]:
                return self._migrate_dimension_reduction(hv, from_params, to_params)
            else:
                return self._migrate_dimension_expansion(hv, from_params, to_params)

        elif from_params["projection_type"] != to_params["projection_type"]:
            return self._migrate_projection_change(hv, from_params, to_params)

        else:
            # Re-encode with new parameters
            return self._migrate_re_encode(hv, from_params, to_params)

    def _migrate_dimension_reduction(
        self, hv: np.ndarray, from_params: Dict[str, Any], to_params: Dict[str, Any]
    ) -> np.ndarray:
        """Reduce hypervector dimension."""
        from_dim = from_params["dimension"]
        to_dim = to_params["dimension"]

        # Use PCA-like projection for dimension reduction
        # Preserve most important components

        # Generate stable projection matrix
        rng = np.random.RandomState(to_params.get("seed", 42))
        projection = rng.randn(to_dim, from_dim)
        projection /= np.sqrt(from_dim)

        # Project to lower dimension
        reduced = projection @ hv

        # Normalize
        reduced = reduced / (np.linalg.norm(reduced) + 1e-8)

        return reduced

    def _migrate_dimension_expansion(
        self, hv: np.ndarray, from_params: Dict[str, Any], to_params: Dict[str, Any]
    ) -> np.ndarray:
        """Expand hypervector dimension."""
        from_dim = from_params["dimension"]
        to_dim = to_params["dimension"]

        # Pad with structured noise
        rng = np.random.RandomState(to_params.get("seed", 42))

        # Keep original components
        expanded = np.zeros(to_dim)
        expanded[:from_dim] = hv

        # Add correlated noise to new dimensions
        noise = rng.randn(to_dim - from_dim)
        noise *= 0.1  # Small magnitude
        expanded[from_dim:] = noise

        # Normalize
        expanded = expanded / (np.linalg.norm(expanded) + 1e-8)

        return expanded

    def _migrate_projection_change(
        self, hv: np.ndarray, from_params: Dict[str, Any], to_params: Dict[str, Any]
    ) -> np.ndarray:
        """Change projection type (approximate)."""
        # This is lossy - we can't perfectly recover original data
        # Best effort: normalize to match new projection statistics

        if to_params["projection_type"] == "binary":
            # Convert to binary
            return (hv > 0).astype(np.float32)

        elif to_params["projection_type"] == "gaussian":
            # Already gaussian-like, just normalize
            return hv / (np.linalg.norm(hv) + 1e-8)

        elif to_params["projection_type"] == "sparse":
            # Sparsify by thresholding
            threshold = np.percentile(np.abs(hv), 90)
            sparse = hv.copy()
            sparse[np.abs(sparse) < threshold] = 0
            return sparse

        else:
            # Unknown projection, return as-is
            return hv

    def _migrate_re_encode(
        self, hv: np.ndarray, from_params: Dict[str, Any], to_params: Dict[str, Any]
    ) -> np.ndarray:
        """Re-encode with new parameters (lossy)."""
        # Can't recover original data, so just ensure compatibility
        # This is a placeholder - in practice, would need original data

        logger.warning(
            "Re-encoding migration is lossy - " "original data needed for perfect migration"
        )

        return hv


if __name__ == "__main__":
    # Example usage
    registry = HypervectorRegistry()

    # Register new version with different parameters
    registry.register_version(
        version="v2.0.0",
        params={"dimension": 20000, "projection_type": "sparse", "seed": 12345, "sparsity": 0.05},
        description="Sparse encoding with double dimension",
    )

    # List versions
    print("Registered versions:")
    for v in registry.list_versions():
        print(f"  {v['version']}: dim={v['dimension']}, type={v['projection_type']}")

    # Get encoders
    encoder_v1 = registry.get_encoder("v1.0.0")
    encoder_v2 = registry.get_encoder("v2.0.0")

    print(f"\nEncoder v1: dim={encoder_v1.dimension}, version={encoder_v1.version}")
    print(f"Encoder v2: dim={encoder_v2.dimension}, version={encoder_v2.version}")

    # Test migration
    migrator = VersionMigrator(registry)

    # Create test hypervector
    test_hv = np.random.randn(10000)
    test_hv = test_hv / np.linalg.norm(test_hv)

    # Migrate to v2
    migrated_hv = migrator.migrate_hypervector(test_hv, "v1.0.0", "v2.0.0")

    print(f"\nMigration test:")
    print(f"  Original shape: {test_hv.shape}")
    print(f"  Migrated shape: {migrated_hv.shape}")
    print(f"  Sparsity: {np.mean(migrated_hv == 0):.2%}")
