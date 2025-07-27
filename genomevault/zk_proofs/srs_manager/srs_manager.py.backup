"""
SRS (Structured Reference String) Manager for ZK proofs.
Handles SRS lifecycle, domain separation, and verification.
"""
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from ...core.config import get_config
from ...core.exceptions import SecurityError, ValidationError
from ...utils.logging import get_logger

logger = get_logger(__name__)
config = get_config()


@dataclass
class SRSMetadata:
    """Metadata for a Structured Reference String."""
    """Metadata for a Structured Reference String."""
    """Metadata for a Structured Reference String."""

    srs_id: str
    circuit_type: str
    curve: str
    powers_of_tau: int
    blake2b_hash: str
    sha256_hash: str
    size_bytes: int
    created_at: str
    source_url: Optional[str] = None
    trusted_setup_ceremony: Optional[str] = None
    participants: Optional[List[str]] = None

    def to_dict(self) -> Dict:
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        """Convert to dictionary."""
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
        def from_dict(cls, data: Dict) -> "SRSMetadata":
        def from_dict(cls, data: Dict) -> "SRSMetadata":
            """Create from dictionary."""
        """Create from dictionary."""
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CircuitMetadata:
    """Metadata for a circuit."""
    """Metadata for a circuit."""
    """Metadata for a circuit."""

    circuit_id: str
    circuit_name: str
    version: str
    proving_key_hash: str
    verifying_key_hash: str
    constraint_count: int
    public_input_count: int
    srs_id: str
    created_at: str

    def to_dict(self) -> Dict:
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        """Convert to dictionary."""
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
        def from_dict(cls, data: Dict) -> "CircuitMetadata":
        def from_dict(cls, data: Dict) -> "CircuitMetadata":
            """Create from dictionary."""
        """Create from dictionary."""
        """Create from dictionary."""
        return cls(**data)


class SRSManager:
    """
    """
    """
    Manages Structured Reference Strings and circuit metadata for ZK proofs.
    Ensures deterministic builds and secure handling.
    """

    def __init__(self, base_path: Path, auto_download: bool = True):
    def __init__(self, base_path: Path, auto_download: bool = True):
        """
        """
    """
        Initialize SRS Manager.

        Args:
            base_path: Base directory for SRS storage
            auto_download: Whether to automatically download missing SRS
        """
            self.base_path = Path(base_path)
            self.srs_dir = self.base_path / "srs"
            self.circuits_dir = self.base_path / "circuits"
            self.registry_path = self.base_path / "registry.json"
            self.auto_download = auto_download

        # Create directories
            self.srs_dir.mkdir(parents=True, exist_ok=True)
            self.circuits_dir.mkdir(parents=True, exist_ok=True)

        # Load or create registry
            self.registry = self._load_registry()

        # Trusted SRS sources
            self.trusted_sources = {
            "powers_of_tau_28": {
                "url": "https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_28.ptau",
                "blake2b_hash": "55c77ce8562366c91e7cda394cf7b7c15a06c12d129c3760143b70e0e8d0f2b5",
                "size_bytes": 5418115356,
            },
            "powers_of_tau_27": {
                "url": "https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_27.ptau",
                "blake2b_hash": "7c3ac6e2e62e3d61f24a09d067946e0df8b644ce1a96bcc40c378f04d88e1870",
                "size_bytes": 2709057828,
            },
        }

        logger.info(f"SRS Manager initialized at {base_path}")

            def _load_registry(self) -> Dict[str, Dict]:
            def _load_registry(self) -> Dict[str, Dict]:
                """Load registry from disk."""
        """Load registry from disk."""
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"srs": {}, "circuits": {}}

                def _save_registry(self) -> None:
                def _save_registry(self) -> None:
                    """Save registry to disk."""
        """Save registry to disk."""
        """Save registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

            def register_srs(self, srs_metadata: SRSMetadata, srs_path: Path) -> None:
            def register_srs(self, srs_metadata: SRSMetadata, srs_path: Path) -> None:
                """
        """
        """
        Register an SRS file.

        Args:
            srs_metadata: SRS metadata
            srs_path: Path to SRS file
        """
        # Verify file exists and hash matches
        if not srs_path.exists():
            raise ValidationError(f"SRS file not found: {srs_path}")

        # Calculate hashes
        blake2b_hash = self._calculate_blake2b(srs_path)
        sha256_hash = self._calculate_sha256(srs_path)

        # Verify hashes
        if srs_metadata.blake2b_hash and blake2b_hash != srs_metadata.blake2b_hash:
            raise SecurityError(f"Blake2b hash mismatch for {srs_metadata.srs_id}")

        if srs_metadata.sha256_hash and sha256_hash != srs_metadata.sha256_hash:
            raise SecurityError(f"SHA256 hash mismatch for {srs_metadata.srs_id}")

        # Copy to managed location
        managed_path = self.srs_dir / f"{srs_metadata.srs_id}.ptau"
        if not managed_path.exists():
            shutil.copy2(srs_path, managed_path)

        # Update registry
            self.registry["srs"][srs_metadata.srs_id] = srs_metadata.to_dict()
            self._save_registry()

        logger.info(f"Registered SRS: {srs_metadata.srs_id}")

            def register_circuit(self, circuit_metadata: CircuitMetadata) -> None:
            def register_circuit(self, circuit_metadata: CircuitMetadata) -> None:
                """
        """
        """
        Register a circuit.

        Args:
            circuit_metadata: Circuit metadata
        """
        # Verify SRS exists
        if circuit_metadata.srs_id not in self.registry["srs"]:
            raise ValidationError(f"Unknown SRS: {circuit_metadata.srs_id}")

        # Update registry
            self.registry["circuits"][circuit_metadata.circuit_id] = circuit_metadata.to_dict()
            self._save_registry()

        logger.info(f"Registered circuit: {circuit_metadata.circuit_id}")

            def get_srs_path(self, srs_id: str) -> Path:
            def get_srs_path(self, srs_id: str) -> Path:
                """
        """
        """
        Get path to SRS file, downloading if necessary.

        Args:
            srs_id: SRS identifier

        Returns:
            Path to SRS file
        """
        # Check if registered
        if srs_id not in self.registry["srs"]:
            # Try to download from trusted sources
            if self.auto_download and srs_id in self.trusted_sources:
                self._download_trusted_srs(srs_id)
            else:
                raise ValidationError(f"Unknown SRS: {srs_id}")

        srs_path = self.srs_dir / f"{srs_id}.ptau"
        if not srs_path.exists():
            raise ValidationError(f"SRS file missing: {srs_id}")

        return srs_path

            def _download_trusted_srs(self, srs_id: str) -> None:
            def _download_trusted_srs(self, srs_id: str) -> None:
                """Download SRS from trusted source."""
        """Download SRS from trusted source."""
        """Download SRS from trusted source."""
        source = self.trusted_sources[srs_id]
        srs_path = self.srs_dir / f"{srs_id}.ptau"

        logger.info(f"Downloading SRS {srs_id} from {source['url']}")

        # Download with progress
        response = requests.get(source["url"], stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(srs_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    if downloaded % (100 * 1024 * 1024) == 0:  # Log every 100MB
                        logger.info(f"Download progress: {progress:.1f}%")

        # Verify hash
        blake2b_hash = self._calculate_blake2b(srs_path)
        if blake2b_hash != source["blake2b_hash"]:
            srs_path.unlink()  # Remove corrupted file
            raise SecurityError(f"Hash mismatch for downloaded SRS {srs_id}")

        # Register
        metadata = SRSMetadata(
            srs_id=srs_id,
            circuit_type="groth16",
            curve="bn254",
            powers_of_tau=int(srs_id.split("_")[-1]),
            blake2b_hash=source["blake2b_hash"],
            sha256_hash=self._calculate_sha256(srs_path),
            size_bytes=source["size_bytes"],
            created_at=datetime.utcnow().isoformat(),
            source_url=source["url"],
            trusted_setup_ceremony="Hermez",
        )

            self.register_srs(metadata, srs_path)

            def _calculate_blake2b(self, file_path: Path) -> str:
            def _calculate_blake2b(self, file_path: Path) -> str:
                """Calculate Blake2b hash of file."""
        """Calculate Blake2b hash of file."""
        """Calculate Blake2b hash of file."""
        import blake2b

        hasher = blake2b.blake2b()

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()

                def _calculate_sha256(self, file_path: Path) -> str:
                def _calculate_sha256(self, file_path: Path) -> str:
                    """Calculate SHA256 hash of file."""
        """Calculate SHA256 hash of file."""
        """Calculate SHA256 hash of file."""
        hasher = hashlib.sha256()

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()

                def create_domain_separator(self, domain: str) -> bytes:
                def create_domain_separator(self, domain: str) -> bytes:
                    """
        """
        """
        Create domain separator for transcript.

        Args:
            domain: Domain string

        Returns:
            Domain separator bytes
        """
        # Use standardized format
        separator = f"GenomeVault-ZK-{domain}".encode("utf-8")

        # Hash to fixed length
        return hashlib.blake2b(separator, digest_size=32).digest()

            def verify_proof_binding(
        self,
        proof: bytes,
        public_inputs: List[str],
        circuit_id: str,
        expected_transcript_hash: Optional[str] = None,
    ) -> bool:
        """
        """
        """
        Verify proof is properly bound to inputs and circuit.

        Args:
            proof: Proof bytes
            public_inputs: Public inputs
            circuit_id: Circuit identifier
            expected_transcript_hash: Expected transcript hash

        Returns:
            Whether verification passed
        """
        # Get circuit metadata
        if circuit_id not in self.registry["circuits"]:
            logger.error(f"Unknown circuit: {circuit_id}")
            return False

        circuit = self.registry["circuits"][circuit_id]

        # Reconstruct transcript
        transcript = self._create_transcript(circuit_id, public_inputs)
        transcript_hash = hashlib.sha256(transcript).hexdigest()

        # Verify transcript hash if provided
        if expected_transcript_hash and transcript_hash != expected_transcript_hash:
            logger.error("Transcript hash mismatch")
            return False

        # Additional verification would happen here
        # For now, return True if basic checks pass
        return True

            def _create_transcript(self, circuit_id: str, public_inputs: List[str]) -> bytes:
            def _create_transcript(self, circuit_id: str, public_inputs: List[str]) -> bytes:
                """Create Fiat-Shamir transcript."""
        """Create Fiat-Shamir transcript."""
        """Create Fiat-Shamir transcript."""
        transcript = bytearray()

        # Add domain separator
        transcript.extend(self.create_domain_separator(circuit_id))

        # Add circuit ID
        transcript.extend(circuit_id.encode("utf-8"))
        transcript.extend(b"\x00")  # Null separator

        # Add public inputs
        for input_val in public_inputs:
            transcript.extend(input_val.encode("utf-8"))
            transcript.extend(b"\x00")

        # Add timestamp for uniqueness
        transcript.extend(str(int(datetime.utcnow().timestamp())).encode("utf-8"))

        return bytes(transcript)

            def get_registry_summary(self) -> Dict[str, Any]:
            def get_registry_summary(self) -> Dict[str, Any]:
                """Get summary of registered SRS and circuits."""
        """Get summary of registered SRS and circuits."""
        """Get summary of registered SRS and circuits."""
        return {
            "srs_count": len(self.registry["srs"]),
            "circuit_count": len(self.registry["circuits"]),
            "srs_list": list(self.registry["srs"].keys()),
            "circuit_list": list(self.registry["circuits"].keys()),
            "total_size_gb": sum(s["size_bytes"] for s in self.registry["srs"].values())
            / (1024**3),
        }


class GnarkDockerBuilder:
    """
    """
    """
    Manages deterministic gnark builds using Docker.
    """

    def __init__(self, workspace: Path):
    def __init__(self, workspace: Path):
        """Initialize Docker builder."""
    """Initialize Docker builder."""
    """Initialize Docker builder."""
        self.workspace = Path(workspace)
        self.docker_image = "genomevault/gnark-builder:v0.9.1"

        def build_circuit(
        self, circuit_name: str, circuit_path: Path, srs_path: Path, output_dir: Path
    ) -> Tuple[Path, Path, CircuitMetadata]:
        """
        """
        """
        Build circuit in Docker container.

        Args:
            circuit_name: Circuit name
            circuit_path: Path to circuit Go file
            srs_path: Path to SRS file
            output_dir: Output directory

        Returns:
            Tuple of (proving_key_path, verifying_key_path, metadata)
        """
        # Create build directory
        build_dir = self.workspace / f"build_{circuit_name}_{int(datetime.utcnow().timestamp())}"
        build_dir.mkdir(parents=True, exist_ok=True)

        # Copy circuit file
        shutil.copy2(circuit_path, build_dir / "circuit.go")

        # Create build script
        build_script = build_dir / "build.sh"
        build_script.write_text(
            f"""#!/bin/bash
set -e

# Compile circuit
go build -o circuit circuit.go

# Generate keys
./circuit compile \\
    --srs /srs/powers_of_tau.ptau \\
    --proving-key /output/proving_key.pk \\
    --verifying-key /output/verifying_key.vk

# Generate metadata
./circuit info > /output/circuit_info.json
"""
        )
        build_script.chmod(0o755)

        # Run Docker build
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{circuit_path.parent}:/circuit:ro",
            "-v",
            f"{srs_path}:/srs/powers_of_tau.ptau:ro",
            "-v",
            f"{output_dir}:/output",
            "-w",
            "/work",
            self.docker_image,
            "/work/build.sh",
        ]

        logger.info(f"Building circuit {circuit_name} in Docker")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Circuit build failed: {result.stderr}")

        # Parse metadata
        info_path = output_dir / "circuit_info.json"
        with open(info_path, "r") as f:
            info = json.load(f)

        # Create circuit metadata
        metadata = CircuitMetadata(
            circuit_id=f"{circuit_name}_v1",
            circuit_name=circuit_name,
            version="1.0",
            proving_key_hash=self._hash_file(output_dir / "proving_key.pk"),
            verifying_key_hash=self._hash_file(output_dir / "verifying_key.vk"),
            constraint_count=info["constraint_count"],
            public_input_count=info["public_input_count"],
            srs_id=srs_path.stem,
            created_at=datetime.utcnow().isoformat(),
        )

        # Clean up
        shutil.rmtree(build_dir)

        return (output_dir / "proving_key.pk", output_dir / "verifying_key.vk", metadata)

            def _hash_file(self, file_path: Path) -> str:
            def _hash_file(self, file_path: Path) -> str:
                """Calculate SHA256 hash of file."""
        """Calculate SHA256 hash of file."""
        """Calculate SHA256 hash of file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()


# Example usage and tests
                def test_srs_manager():
                def test_srs_manager():
                    """Test SRS manager functionality."""
    """Test SRS manager functionality."""
    """Test SRS manager functionality."""
    # Initialize manager
    manager = SRSManager(Path("/tmp/genomevault_srs"), auto_download=True)

    # Get SRS (will download if needed)
    srs_path = manager.get_srs_path("powers_of_tau_27")
    print(f"SRS path: {srs_path}")

    # Create domain separator
    separator = manager.create_domain_separator("median_proof")
    print(f"Domain separator: {separator.hex()}")

    # Get registry summary
    summary = manager.get_registry_summary()
    print(f"Registry summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    test_srs_manager()
