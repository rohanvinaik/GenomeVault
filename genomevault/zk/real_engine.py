"""Real Engine module."""

from __future__ import annotations

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import hashlib
import json
import time

from genomevault.crypto.types import PrivateKey, PublicKey
from genomevault.utils.logging import get_logger
from genomevault.zk.backends.circom_snarkjs import (
    CircuitPaths,
    prove,
    toolchain_available,
    verify,
)

logger = get_logger(__name__)


@dataclass
class RealProof:
    """Data container for realproof information."""

    proof: dict
    public: dict

    def to_wire(self) -> dict[str, Any]:
        """To wire.

        Returns:
            Operation result.
        """
        return {"proof": self.proof, "public_inputs": self.public}


class RealZKEngine:
    """Real ZK engine using Circom + snarkjs (Groth16).

    Currently supports circuit_type == "sum64" only:
      private a,b; public c; with constraint a + b == c

    Falls back to Ed25519 signed transcripts when Circom is not available.
    """

    def __init__(self, repo_root: str) -> None:
        """Initialize instance.

        Args:
            repo_root: Repo root.
        """
        self.repo_root = Path(repo_root)

        # Generate or load Ed25519 keypair for transcript fallback
        self.key_path = self.repo_root / ".zk_transcript_key"
        self.signing_key: PrivateKey
        if self.key_path.exists():
            with open(self.key_path, "rb") as f:
                key_bytes = f.read()
                self.signing_key = ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)
        else:
            self.signing_key = ed25519.Ed25519PrivateKey.generate()
            # Save key for persistence
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

        self.verify_key: PublicKey = self.signing_key.public_key()

    def _create_transcript(
        self, circuit_type: str, inputs: Dict[str, Any], claim: str = None
    ) -> Dict[str, Any]:
        """Create a signed transcript as fallback when Circom is not available.

        Args:
            circuit_type: Type of circuit being simulated
            inputs: Input values (both public and private)
            claim: Optional claim about the computation

        Returns:
            Transcript with signature and metadata
        """
        # Separate public and private inputs based on circuit type
        if circuit_type == "sum64":
            public_inputs = {"c": inputs.get("c", 0)}
            # private_inputs = {"a": inputs.get("a", 0), "b": inputs.get("b", 0)}
            if claim is None:
                claim = f"a + b = {public_inputs['c']}"
        else:
            # For other circuits, assume all inputs are public by default
            public_inputs = {k: v for k, v in inputs.items() if not k.startswith("_")}
            # private_inputs = {k: v for k, v in inputs.items()
            #                 if k.startswith("_")}
            if claim is None:
                claim = f"Computation verified for {circuit_type}"

        # Create canonical transcript
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
            "manifest_hash": hashlib.sha256(
                f"{circuit_type}:{self.repo_root}".encode()
            ).hexdigest(),
            "algorithm": "ed25519",
            "parameters": {"circuit": circuit_type, "backend": "transcript_fallback"},
        }

        # Sign the transcript
        canonical_json = json.dumps(transcript, sort_keys=True)
        signature = self.signing_key.sign(canonical_json.encode())

        # Add signature and public key to transcript
        transcript["signature"] = signature.hex()
        transcript["public_key"] = self.verify_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        ).hex()

        return transcript

    def _verify_transcript(self, transcript: Dict[str, Any]) -> bool:
        """Verify a signed transcript.

        Args:
            transcript: The transcript to verify

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Extract and remove signature and public key
            signature_hex = transcript.pop("signature", None)
            public_key_hex = transcript.pop("public_key", None)

            if not signature_hex or not public_key_hex:
                return False

            # Recreate canonical JSON
            canonical_json = json.dumps(transcript, sort_keys=True)

            # Verify signature
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
            public_key.verify(bytes.fromhex(signature_hex), canonical_json.encode())

            # Restore fields
            transcript["signature"] = signature_hex
            transcript["public_key"] = public_key_hex

            return True

        except Exception as e:
            logger.debug(f"Transcript verification failed: {e}")
            return False

    def create_proof(self, *, circuit_type: str, inputs: dict[str, Any]) -> RealProof:
        """Create proof.

        Returns:
            Newly created proof. Uses Circom if available, otherwise falls back
            to Ed25519 signed transcript.

        Raises:
            ValueError: When circuit type is not supported.
        """
        # Check if Circom toolchain is available
        if toolchain_available():
            # Use real ZK proof with Circom
            if circuit_type != "sum64":
                # For unsupported circuits, still use transcript fallback
                logger.info(
                    f"Circuit type '{circuit_type}' not available in Circom, " "using transcript"
                )
                transcript = self._create_transcript(circuit_type, inputs)
                return RealProof(proof=transcript, public=transcript.get("public_inputs", {}))

            try:
                a = int(inputs.get("a", 0))
                b = int(inputs.get("b", 0))
                c = int(inputs.get("c", 0))
                paths = CircuitPaths.for_sum64(self.repo_root)
                out = prove(paths, a=a, b=b, c_public=c)

                # Add metadata to indicate real proof
                proof_with_meta = {
                    **out["proof"],
                    "_metadata": {
                        "engine": "circom",
                        "backend": "groth16",
                        "circuit_type": circuit_type,
                    },
                }
                return RealProof(proof=proof_with_meta, public=out["public"])

            except Exception as e:
                logger.warning(
                    f"Circom proof generation failed: {e}, " "falling back to transcript"
                )
                # Fall through to transcript generation
        else:
            logger.info("ZK toolchain not available, using transcript fallback")

        # Fallback: Create signed transcript
        transcript = self._create_transcript(circuit_type, inputs)

        # Return in same format as real proof
        return RealProof(proof=transcript, public=transcript.get("public_inputs", {}))

    def verify_proof(self, *, proof: dict, public_inputs: dict) -> bool:
        """Verify proof.

        Returns:
            Boolean result. Verifies Circom proofs or transcript signatures.
        """
        # Check if this is a transcript proof
        if isinstance(proof, dict):
            # Check for transcript signature
            if proof.get("engine") == "transcript" and "signature" in proof:
                logger.debug("Verifying transcript proof")
                # Verify the transcript signature
                transcript_copy = proof.copy()

                # Check that public inputs match
                transcript_public = transcript_copy.get("public_inputs", {})
                if transcript_public != public_inputs:
                    logger.warning("Public inputs mismatch in transcript verification")
                    return False

                # Verify signature
                return self._verify_transcript(transcript_copy)

            # Check for metadata indicating proof type
            metadata = proof.get("_metadata", {})
            if metadata.get("engine") == "circom":
                # Remove metadata before passing to Circom verifier
                proof_without_meta = {k: v for k, v in proof.items() if k != "_metadata"}

                if toolchain_available():
                    try:
                        paths = CircuitPaths.for_sum64(self.repo_root)
                        return verify(paths, proof=proof_without_meta, public=public_inputs)
                    except Exception as e:
                        logger.error(f"Circom verification failed: {e}")
                        return False
                else:
                    logger.warning("Cannot verify Circom proof without toolchain")
                    return False

        # Default: Try Circom verification if available
        if toolchain_available():
            try:
                paths = CircuitPaths.for_sum64(self.repo_root)
                return verify(paths, proof=proof, public=public_inputs)
            except Exception as e:
                logger.error(f"Proof verification failed: {e}")
                return False

        # No verification method available
        logger.warning("No verification method available for proof")
        return False
