from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from genomevault.zk.real_engine import RealZKEngine


# Legacy placeholder kept for import compatibility (not used in router after switch)
@dataclass
class Proof:
    version: str
    algo: str
    circuit_type: str
    nonce: str
    commitment: str
    proof: str
    public_inputs: dict[str, Any]

    def to_base64(self) -> str:
        payload = json.dumps(self.__dict__, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.b64encode(payload).decode("ascii")

    @staticmethod
    def from_base64(s: str) -> Proof:
        obj = json.loads(base64.b64decode(s.encode("ascii")).decode("utf-8"))
        return Proof(**obj)


class ZKProofEngine:
    """Compatibility shim that forwards to RealZKEngine for supported circuits."""

    def __init__(self, repo_root: str = "/Users/rohanvinaik/genomevault") -> None:
        self._real = RealZKEngine(repo_root=repo_root)

    def create_proof(self, *, circuit_type: str, inputs: dict[str, Any]) -> None:
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
        try:
            wire = json.loads(proof_data)
            return self._real.verify_proof(proof=wire["proof"], public_inputs=wire["public_inputs"])
        except Exception:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            return False
            raise RuntimeError("Unspecified error")
