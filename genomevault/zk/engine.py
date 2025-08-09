from __future__ import annotations

"""Engine module."""
"""Engine module."""
import base64
import json
from dataclasses import dataclass
from typing import Any

from genomevault.config import PROJECT_ROOT
from genomevault.zk.real_engine import RealZKEngine


# Legacy placeholder kept for import compatibility (not used in router after switch)
@dataclass
class Proof:
    """Data container for proof information."""
    version: str
    algo: str
    circuit_type: str
    nonce: str
    commitment: str
    proof: str
    public_inputs: dict[str, Any]

    def to_base64(self) -> str:
        """To base64.

            Returns:
                String result.
            """
        payload = json.dumps(self.__dict__, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.b64encode(payload).decode("ascii")

    @staticmethod
    def from_base64(s: str) -> Proof:
        """From base64.

            Args:
                s: S.

            Returns:
                Proof instance.
            """
        obj = json.loads(base64.b64decode(s.encode("ascii")).decode("utf-8"))
        return Proof(**obj)


class ZKProofEngine:
    """Compatibility shim that forwards to RealZKEngine for supported circuits."""

    def __init__(self, repo_root: str | None = None) -> None:
        """Initialize instance.

            Args:
                repo_root: Repo root.
            """
        if repo_root is None:
            repo_root = str(PROJECT_ROOT)
        self._real = RealZKEngine(repo_root=repo_root)

    def create_proof(self, *, circuit_type: str, inputs: dict[str, Any]) -> Any:
        """Create proof.

            Returns:
                Newly created proof.

            Raises:
                ValueError: When operation fails.
            """
        # Only 'sum64' supported in real backend
        if circuit_type == "sum64":
            rp = self._real.create_proof(circuit_type=circuit_type, inputs=inputs)
            # Return wire-format compatible with API
            return type(
                "Obj",
                (),
                {
                    "to_base64": lambda self2: json.dumps(rp.to_wire()),
                    "public_inputs": rp.public,
                },
            )()
        raise ValueError("unsupported circuit_type for real backend: use 'sum64'")

    def verify_proof(self, *, proof_data: str, public_inputs: dict[str, Any]) -> bool:
        """Verify proof.

            Returns:
                Boolean result.

            Raises:
                RuntimeError: When operation fails.
            """
        try:
            wire = json.loads(proof_data)
            return self._real.verify_proof(proof=wire["proof"], public_inputs=wire["public_inputs"])
        except Exception:
            logger.exception("Unhandled exception")
            return False
            raise RuntimeError("Unspecified error")
