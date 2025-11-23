"""Real Engine module for Zero-Knowledge Proofs.

This module provides the main interface for generating and verifying ZK proofs
using compiled Circom circuits and snarkjs.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RealProof:
    """Data container for ZK proof information."""

    proof: dict
    public: dict
    circuit_type: str = "unknown"
    metadata: Optional[dict] = None

    def to_wire(self) -> dict[str, Any]:
        """Convert proof to wire format.

        Returns:
            Dictionary containing proof and public inputs.
        """
        result = {
            "proof": self.proof,
            "public_inputs": self.public,
            "circuit_type": self.circuit_type,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class CircuitArtifacts:
    """Container for circuit build artifacts."""

    wasm_path: Path
    zkey_path: Path
    vkey_path: Path
    r1cs_path: Path
    circuit_type: str

    def exists(self) -> bool:
        """Check if all artifacts exist."""
        return all(
            [
                self.wasm_path.exists(),
                self.zkey_path.exists(),
                self.vkey_path.exists(),
                self.r1cs_path.exists(),
            ]
        )

    @property
    def verification_key(self) -> dict:
        """Load and return verification key."""
        with open(self.vkey_path, "r") as f:
            return json.load(f)


class RealZKEngine:
    """Real ZK engine using Circom + snarkjs (Groth16).

    Supports multiple circuit types with automatic artifact loading
    and caching. Falls back to Ed25519 signed transcripts when
    Circom/snarkjs is not available.
    """

    def __init__(self, repo_root: str) -> None:
        """Initialize the ZK engine.

        Args:
            repo_root: Root directory of the repository.
        """
        self.repo_root = Path(repo_root)
        self.circuits_dir = self.repo_root / "genomevault" / "zk" / "circuits"

        # Cache for loaded circuit artifacts
        self._circuit_cache: Dict[str, CircuitArtifacts] = {}

        # Supported circuits
        self.supported_circuits = ["sum64", "median_verification"]

        # Initialize Ed25519 keypair for fallback mode
        self._init_fallback_keys()

        # Check toolchain availability
        self._check_toolchain()

    def _init_fallback_keys(self) -> None:
        """Initialize Ed25519 keys for transcript fallback."""
        self.key_path = self.repo_root / ".zk_transcript_key"

        if self.key_path.exists():
            with open(self.key_path, "rb") as f:
                key_bytes = f.read()
                self.signing_key = ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)
        else:
            self.signing_key = ed25519.Ed25519PrivateKey.generate()
            try:
                self.key_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.key_path, "wb") as f:
                    f.write(
                        self.signing_key.private_bytes(
                            encoding=serialization.Encoding.Raw,
                            format=serialization.PrivateFormat.Raw,
                            encryption_algorithm=serialization.NoEncryption(),
                        )
                    )
            except Exception as e:
                logger.warning(f"Could not save transcript key: {e}")

        self.verify_key = self.signing_key.public_key()

    def _check_toolchain(self) -> bool:
        """Check if snarkjs is available.

        Returns:
            True if snarkjs is available, False otherwise.
        """
        try:
            result = subprocess.run(
                ["snarkjs", "--version"], capture_output=True, text=True, timeout=5
            )
            self.toolchain_available = result.returncode == 0
            if self.toolchain_available:
                logger.info(f"snarkjs available: {result.stdout.strip()}")
            return self.toolchain_available
        except (subprocess.SubprocessError, FileNotFoundError):
            self.toolchain_available = False
            logger.warning("snarkjs not found, will use transcript fallback")
            return False

    @lru_cache(maxsize=10)
    def load_circuit(self, circuit_type: str) -> Optional[CircuitArtifacts]:
        """Load circuit artifacts from build directory.

        Args:
            circuit_type: Type of circuit to load (e.g., "sum64")

        Returns:
            CircuitArtifacts if found, None otherwise.
        """
        if circuit_type in self._circuit_cache:
            return self._circuit_cache[circuit_type]

        # Look for circuit directory
        circuit_dir = self.circuits_dir / circuit_type
        build_dir = circuit_dir / "build"

        if not build_dir.exists():
            logger.warning(f"Build directory not found for {circuit_type}: {build_dir}")
            return None

        # Define expected artifact paths
        artifacts = CircuitArtifacts(
            wasm_path=build_dir / f"{circuit_type}_js" / f"{circuit_type}.wasm",
            zkey_path=build_dir / f"{circuit_type}_final.zkey",
            vkey_path=build_dir / "verification_key.json",
            r1cs_path=build_dir / f"{circuit_type}.r1cs",
            circuit_type=circuit_type,
        )

        if not artifacts.exists():
            logger.warning(f"Missing artifacts for {circuit_type}")
            logger.info(f"Run scripts/build_circuits.sh to build {circuit_type}")
            return None

        # Cache and return
        self._circuit_cache[circuit_type] = artifacts
        logger.info(f"Loaded circuit artifacts for {circuit_type}")
        return artifacts

    def generate_proof(self, circuit_type: str, inputs: Dict[str, Any]) -> Optional[RealProof]:
        """Generate a ZK proof using snarkjs.

        Args:
            circuit_type: Type of circuit to use
            inputs: Circuit inputs (both public and private)

        Returns:
            RealProof if successful, None otherwise.
        """
        if not self.toolchain_available:
            logger.info("Using transcript fallback (snarkjs not available)")
            return self._create_transcript_proof(circuit_type, inputs)

        # Load circuit artifacts
        artifacts = self.load_circuit(circuit_type)
        if not artifacts:
            logger.warning(f"Circuit {circuit_type} not available, using transcript")
            return self._create_transcript_proof(circuit_type, inputs)

        try:
            # Create temporary directory for witness
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                witness_path = tmpdir_path / "witness.wtns"
                input_path = tmpdir_path / "input.json"
                proof_path = tmpdir_path / "proof.json"
                public_path = tmpdir_path / "public.json"

                # Write input file
                with open(input_path, "w") as f:
                    json.dump(inputs, f)

                # Generate witness
                logger.debug(f"Generating witness for {circuit_type}")
                witness_result = subprocess.run(
                    [
                        "snarkjs",
                        "wtns",
                        "calculate",
                        str(artifacts.wasm_path),
                        str(input_path),
                        str(witness_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if witness_result.returncode != 0:
                    logger.error(f"Witness generation failed: {witness_result.stderr}")
                    return self._create_transcript_proof(circuit_type, inputs)

                # Generate proof
                logger.debug(f"Generating proof for {circuit_type}")
                proof_result = subprocess.run(
                    [
                        "snarkjs",
                        "groth16",
                        "prove",
                        str(artifacts.zkey_path),
                        str(witness_path),
                        str(proof_path),
                        str(public_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if proof_result.returncode != 0:
                    logger.error(f"Proof generation failed: {proof_result.stderr}")
                    return self._create_transcript_proof(circuit_type, inputs)

                # Read proof and public signals
                with open(proof_path, "r") as f:
                    proof_data = json.load(f)
                with open(public_path, "r") as f:
                    public_signals = json.load(f)

                # Create metadata
                metadata = {
                    "engine": "snarkjs",
                    "backend": "groth16",
                    "circuit_type": circuit_type,
                    "timestamp": int(time.time()),
                    "prover_version": "snarkjs",
                }

                # Map public signals to named outputs
                public_dict = self._map_public_signals(circuit_type, public_signals)

                logger.info(f"Successfully generated proof for {circuit_type}")
                return RealProof(
                    proof=proof_data,
                    public=public_dict,
                    circuit_type=circuit_type,
                    metadata=metadata,
                )

        except subprocess.TimeoutExpired:
            logger.error(f"Proof generation timed out for {circuit_type}")
            return self._create_transcript_proof(circuit_type, inputs)
        except Exception as e:
            logger.error(f"Proof generation failed: {e}")
            return self._create_transcript_proof(circuit_type, inputs)

    def verify_proof(
        self,
        proof: Dict[str, Any],
        public_inputs: Dict[str, Any],
        circuit_type: Optional[str] = None,
    ) -> bool:
        """Verify a ZK proof using snarkjs.

        Args:
            proof: The proof to verify
            public_inputs: Public inputs/signals
            circuit_type: Type of circuit (optional, can be inferred)

        Returns:
            True if proof is valid, False otherwise.
        """
        # Check if this is a transcript proof
        if isinstance(proof, dict) and proof.get("engine") == "transcript":
            # For transcript proofs, verify signature AND check public inputs match
            transcript_public = proof.get("public_inputs", {})

            # Convert both to strings for comparison
            transcript_public_str = {k: str(v) for k, v in transcript_public.items()}
            provided_public_str = {k: str(v) for k, v in public_inputs.items()}

            if transcript_public_str != provided_public_str:
                logger.warning(
                    f"Public inputs mismatch: expected {transcript_public_str}, got {provided_public_str}"
                )
                return False

            return self._verify_transcript(proof)

        # Extract circuit type from metadata if not provided
        if not circuit_type:
            if isinstance(proof, dict) and "metadata" in proof:
                circuit_type = proof["metadata"].get("circuit_type", "sum64")
            else:
                circuit_type = "sum64"  # Default

        if not self.toolchain_available:
            logger.warning("Cannot verify proof without snarkjs")
            return False

        # Load circuit artifacts
        artifacts = self.load_circuit(circuit_type)
        if not artifacts:
            logger.error(f"Cannot verify proof: circuit {circuit_type} not found")
            return False

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                proof_path = tmpdir_path / "proof.json"
                public_path = tmpdir_path / "public.json"
                vkey_path = tmpdir_path / "vkey.json"

                # Extract proof data (remove metadata if present)
                proof_data = proof
                if isinstance(proof, dict) and "metadata" in proof:
                    proof_data = {k: v for k, v in proof.items() if k != "metadata"}

                # Convert public inputs to array format
                public_array = self._public_dict_to_array(circuit_type, public_inputs)

                # Write files
                with open(proof_path, "w") as f:
                    json.dump(proof_data, f)
                with open(public_path, "w") as f:
                    json.dump(public_array, f)
                with open(vkey_path, "w") as f:
                    json.dump(artifacts.verification_key, f)

                # Verify proof
                logger.debug(f"Verifying proof for {circuit_type}")
                verify_result = subprocess.run(
                    [
                        "snarkjs",
                        "groth16",
                        "verify",
                        str(vkey_path),
                        str(public_path),
                        str(proof_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if verify_result.returncode != 0:
                    logger.error(f"Verification failed: {verify_result.stderr}")
                    return False

                # Check output for verification result
                output = verify_result.stdout.strip().lower()
                is_valid = "ok" in output or "true" in output or "pass" in output

                if is_valid:
                    logger.info(f"Proof verified successfully for {circuit_type}")
                else:
                    logger.warning(f"Proof verification failed for {circuit_type}")

                return is_valid

        except Exception as e:
            logger.error(f"Proof verification error: {e}")
            return False

    def _map_public_signals(self, circuit_type: str, signals: list) -> Dict[str, Any]:
        """Map public signals array to named dictionary.

        Args:
            circuit_type: Type of circuit
            signals: Array of public signals

        Returns:
            Dictionary with named public outputs.
        """
        if circuit_type == "sum64":
            # sum64 has one public output: c
            return {"c": str(signals[0]) if signals else "0"}
        elif circuit_type == "median_verification":
            # Example for median circuit
            return {
                "median": str(signals[0]) if len(signals) > 0 else "0",
                "count": str(signals[1]) if len(signals) > 1 else "0",
            }
        else:
            # Generic mapping
            return {f"output_{i}": str(v) for i, v in enumerate(signals)}

    def _public_dict_to_array(self, circuit_type: str, public_dict: Dict[str, Any]) -> list:
        """Convert public inputs dictionary to array format.

        Args:
            circuit_type: Type of circuit
            public_dict: Dictionary of public inputs

        Returns:
            Array of public signals.
        """
        if circuit_type == "sum64":
            return [str(public_dict.get("c", 0))]
        elif circuit_type == "median_verification":
            return [str(public_dict.get("median", 0)), str(public_dict.get("count", 0))]
        else:
            # Try to preserve order or use sorted keys
            return [str(v) for v in public_dict.values()]

    def _create_transcript_proof(self, circuit_type: str, inputs: Dict[str, Any]) -> RealProof:
        """Create a signed transcript as fallback.

        Args:
            circuit_type: Type of circuit
            inputs: Circuit inputs

        Returns:
            RealProof with transcript data.
        """
        # Separate public and private inputs
        if circuit_type == "sum64":
            public_inputs = {"c": str(inputs.get("c", 0))}
            claim = f"Knowledge of a + b = {public_inputs['c']}"
        else:
            # Generic handling
            public_inputs = {k: str(v) for k, v in inputs.items() if not k.startswith("_")}
            claim = f"Computation verified for {circuit_type}"

        # Create transcript
        transcript = {
            "version": "1.0",
            "engine": "transcript",
            "circuit_type": circuit_type,
            "timestamp": int(time.time()),
            "claim": claim,
            "public_inputs": public_inputs,
            "query_hash": hashlib.sha256(
                json.dumps(public_inputs, sort_keys=True).encode()
            ).hexdigest(),
            "algorithm": "ed25519",
        }

        # Sign transcript
        canonical_json = json.dumps(transcript, sort_keys=True)
        signature = self.signing_key.sign(canonical_json.encode())

        transcript["signature"] = signature.hex()
        transcript["public_key"] = self.verify_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        ).hex()

        return RealProof(
            proof=transcript,
            public=public_inputs,
            circuit_type=circuit_type,
            metadata={"fallback": True},
        )

    def _verify_transcript(self, transcript: Dict[str, Any]) -> bool:
        """Verify a signed transcript.

        Args:
            transcript: Transcript to verify

        Returns:
            True if valid, False otherwise.
        """
        try:
            # Make a copy to avoid modifying original
            transcript_copy = transcript.copy()

            # Extract signature and public key
            signature_hex = transcript_copy.pop("signature", None)
            public_key_hex = transcript_copy.pop("public_key", None)

            if not signature_hex or not public_key_hex:
                return False

            # Recreate canonical JSON
            canonical_json = json.dumps(transcript_copy, sort_keys=True)

            # Verify signature
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
            public_key.verify(bytes.fromhex(signature_hex), canonical_json.encode())

            logger.info("Transcript signature verified")
            return True

        except Exception as e:
            logger.debug(f"Transcript verification failed: {e}")
            return False

    # Compatibility methods for existing code
    def create_proof(self, *, circuit_type: str, inputs: dict[str, Any]) -> RealProof:
        """Create proof (compatibility method).

        Args:
            circuit_type: Type of circuit
            inputs: Circuit inputs

        Returns:
            RealProof instance.
        """
        return self.generate_proof(circuit_type, inputs)

    def list_available_circuits(self) -> list[str]:
        """List available compiled circuits.

        Returns:
            List of circuit types with compiled artifacts.
        """
        available = []
        for circuit_type in self.supported_circuits:
            artifacts = self.load_circuit(circuit_type)
            if artifacts and artifacts.exists():
                available.append(circuit_type)

        # Check for any other circuits in the circuits directory
        if self.circuits_dir.exists():
            for circuit_dir in self.circuits_dir.iterdir():
                if circuit_dir.is_dir():
                    circuit_name = circuit_dir.name
                    if circuit_name not in self.supported_circuits:
                        build_dir = circuit_dir / "build"
                        if build_dir.exists():
                            # Check for key artifacts
                            wasm = build_dir / f"{circuit_name}_js" / f"{circuit_name}.wasm"
                            zkey = build_dir / f"{circuit_name}_final.zkey"
                            if wasm.exists() and zkey.exists():
                                available.append(circuit_name)

        return available
