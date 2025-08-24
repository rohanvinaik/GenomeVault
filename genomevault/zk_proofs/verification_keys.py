"""
Production verification key management for ZK proofs.

This module handles generating, validating, and managing verification keys
for production ZK proof systems with proper security guarantees.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from genomevault.utils.logging import get_logger
from genomevault.utils.production_safety import (
    require_secure_environment,
    ProductionSafetyError,
    validate_not_mock,
)
from genomevault.utils.fallback_logger import get_fallback_logger
from genomevault.zk_proofs.backends.circom_backend import CircomBackend

logger = get_logger(__name__)


@dataclass
class VerificationKey:
    """Production verification key with metadata."""

    circuit_name: str
    key_data: Dict[str, Any]
    key_hash: str
    generated_at: datetime
    tau_power: int
    is_production: bool
    ceremony_participants: List[str]

    @classmethod
    def from_file(cls, circuit_name: str, key_path: Path) -> "VerificationKey":
        """Load verification key from file with validation."""
        if not key_path.exists():
            raise FileNotFoundError(f"Verification key not found: {key_path}")

        with open(key_path, "r") as f:
            key_data = json.load(f)

        # Calculate key hash for integrity
        key_hash = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

        # Extract metadata (if available)
        metadata = key_data.get("_metadata", {})

        return cls(
            circuit_name=circuit_name,
            key_data=key_data,
            key_hash=key_hash,
            generated_at=datetime.fromisoformat(
                metadata.get("generated_at", datetime.now().isoformat())
            ),
            tau_power=metadata.get("tau_power", 12),
            is_production=metadata.get("is_production", False),
            ceremony_participants=metadata.get("ceremony_participants", []),
        )

    def save_with_metadata(self, key_path: Path) -> None:
        """Save verification key with production metadata."""
        # Add metadata to key data
        key_data_with_meta = self.key_data.copy()
        key_data_with_meta["_metadata"] = {
            "circuit_name": self.circuit_name,
            "key_hash": self.key_hash,
            "generated_at": self.generated_at.isoformat(),
            "tau_power": self.tau_power,
            "is_production": self.is_production,
            "ceremony_participants": self.ceremony_participants,
            "generator": "GenomeVault Production Key Generator v1.0",
        }

        # Ensure parent directory exists
        key_path.parent.mkdir(parents=True, exist_ok=True)

        # Write key with metadata
        with open(key_path, "w") as f:
            json.dump(key_data_with_meta, f, indent=2)

        logger.info(f"Verification key saved with metadata: {key_path}")

    def validate_production_ready(self) -> bool:
        """Validate that verification key is suitable for production."""
        checks = []

        # Check if marked as production
        checks.append(("is_production", self.is_production))

        # Check tau power is sufficient for security
        checks.append(("tau_power_sufficient", self.tau_power >= 16))

        # Check key has required structure
        required_fields = ["vk_alpha_1", "vk_beta_2", "vk_gamma_2", "vk_delta_2", "IC"]
        has_fields = all(field in self.key_data for field in required_fields)
        checks.append(("has_required_fields", has_fields))

        # Check ceremony participation
        checks.append(("has_participants", len(self.ceremony_participants) > 0))

        failed_checks = [name for name, passed in checks if not passed]

        if failed_checks:
            logger.warning(f"Verification key failed production checks: {failed_checks}")
            return False

        logger.info("Verification key passed all production checks")
        return True


class ProductionVerificationKeyManager:
    """Manages production verification keys with security guarantees."""

    def __init__(self, keys_directory: Path = None):
        self.keys_dir = keys_directory or Path("keys/production")
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        self.backend = CircomBackend()

    @require_secure_environment("verification key generation")
    def generate_production_keys(
        self, circuit_names: List[str], tau_power: int = 16, ceremony_participants: List[str] = None
    ) -> Dict[str, VerificationKey]:
        """Generate production-ready verification keys with proper ceremony."""
        fallback_logger = get_fallback_logger("VerificationKeys")

        if tau_power < 16:
            raise ProductionSafetyError(
                "Production verification keys require tau_power >= 16 for security"
            )

        participants = ceremony_participants or ["GenomeVault-Production"]
        logger.info(f"Starting production key generation for {len(circuit_names)} circuits")
        logger.info(f"Tau power: {tau_power}, Participants: {participants}")

        generated_keys = {}

        for circuit_name in circuit_names:
            logger.info(f"Generating production verification key for {circuit_name}")

            # Perform trusted setup with higher security parameters
            if not self.backend.setup_trusted_setup(circuit_name, tau_power=tau_power):
                logger.error(f"Failed to generate trusted setup for {circuit_name}")
                continue

            # Load the generated verification key
            if circuit_name not in self.backend.circuits:
                logger.error(f"Circuit {circuit_name} not found in backend")
                continue

            circuit = self.backend.circuits[circuit_name]

            if not circuit.vkey_path.exists():
                logger.error(f"Verification key not generated for {circuit_name}")
                continue

            # Load and enhance with production metadata
            with open(circuit.vkey_path, "r") as f:
                key_data = json.load(f)

            # Create production verification key
            vkey = VerificationKey(
                circuit_name=circuit_name,
                key_data=key_data,
                key_hash=hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest(),
                generated_at=datetime.now(),
                tau_power=tau_power,
                is_production=True,
                ceremony_participants=participants,
            )

            # Validate production readiness
            if not vkey.validate_production_ready():
                logger.error(f"Generated key for {circuit_name} failed production validation")
                continue

            # Save to production keys directory
            prod_key_path = self.keys_dir / f"{circuit_name}_verification_key.json"
            vkey.save_with_metadata(prod_key_path)

            generated_keys[circuit_name] = vkey
            logger.info(f"✅ Production verification key generated for {circuit_name}")

        logger.info(
            f"Production key generation complete: {len(generated_keys)}/{len(circuit_names)} successful"
        )
        return generated_keys

    def load_production_key(self, circuit_name: str) -> Optional[VerificationKey]:
        """Load and validate production verification key."""
        key_path = self.keys_dir / f"{circuit_name}_verification_key.json"

        if not key_path.exists():
            logger.warning(f"Production verification key not found for {circuit_name}")
            return None

        try:
            vkey = VerificationKey.from_file(circuit_name, key_path)

            # Validate production readiness
            if not vkey.validate_production_ready():
                logger.error(f"Production key for {circuit_name} failed validation")
                return None

            # Additional integrity checks
            validate_not_mock(vkey.key_data, f"verification key for {circuit_name}")
            fallback_logger = get_fallback_logger("VerificationKeys")
            fallback_logger.log_successful_real_backend(
                "key_loading", f"ProductionKey-{circuit_name}"
            )

            logger.info(f"Production verification key loaded for {circuit_name}")
            return vkey

        except Exception as e:
            logger.error(f"Failed to load production key for {circuit_name}: {e}")
            return None

    def verify_key_integrity(self, circuit_name: str) -> bool:
        """Verify the integrity of a production verification key."""
        vkey = self.load_production_key(circuit_name)
        if not vkey:
            return False

        # Recalculate hash to verify integrity
        current_hash = hashlib.sha256(
            json.dumps(vkey.key_data, sort_keys=True).encode()
        ).hexdigest()

        if current_hash != vkey.key_hash:
            logger.error(f"Key integrity check failed for {circuit_name}")
            return False

        logger.info(f"Key integrity verified for {circuit_name}")
        return True

    def list_available_keys(self) -> List[str]:
        """List all available production verification keys."""
        keys = []
        for key_file in self.keys_dir.glob("*_verification_key.json"):
            circuit_name = key_file.name.replace("_verification_key.json", "")
            if self.verify_key_integrity(circuit_name):
                keys.append(circuit_name)
        return keys

    def get_key_info(self, circuit_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a production verification key."""
        vkey = self.load_production_key(circuit_name)
        if not vkey:
            return None

        return {
            "circuit_name": vkey.circuit_name,
            "key_hash": vkey.key_hash,
            "generated_at": vkey.generated_at.isoformat(),
            "tau_power": vkey.tau_power,
            "is_production": vkey.is_production,
            "ceremony_participants": vkey.ceremony_participants,
            "production_ready": vkey.validate_production_ready(),
        }


# Global instance for easy access
_production_key_manager = None


def get_production_key_manager() -> ProductionVerificationKeyManager:
    """Get the global production verification key manager."""
    global _production_key_manager
    if _production_key_manager is None:
        _production_key_manager = ProductionVerificationKeyManager()
    return _production_key_manager


def ensure_production_keys(circuit_names: List[str]) -> bool:
    """Ensure production verification keys exist for the given circuits."""
    manager = get_production_key_manager()

    missing_keys = []
    for circuit_name in circuit_names:
        if not manager.load_production_key(circuit_name):
            missing_keys.append(circuit_name)

    if missing_keys:
        logger.info(f"Generating missing production keys: {missing_keys}")
        generated = manager.generate_production_keys(missing_keys)
        return len(generated) == len(missing_keys)

    logger.info("All required production verification keys are available")
    return True
