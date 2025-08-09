from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genomevault.zk.backends.circom_snarkjs import (
    CircuitPaths,
    prove,
    toolchain_available,
    verify,
)


@dataclass
class RealProof:
    proof: dict
    public: dict

    def to_wire(self) -> dict[str, Any]:
        return {"proof": self.proof, "public_inputs": self.public}


class RealZKEngine:
    """Real ZK engine using Circom + snarkjs (Groth16).

    Currently supports circuit_type == "sum64" only:
      private a,b; public c; with constraint a + b == c
    """

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root)

    def create_proof(self, *, circuit_type: str, inputs: dict[str, Any]) -> RealProof:
        if circuit_type != "sum64":
            raise ValueError("unsupported circuit_type; only 'sum64' is available in this build")
        if not toolchain_available():
            raise RuntimeError("ZK toolchain not available (circom/snarkjs/node)")
        a = int(inputs.get("a"))
        b = int(inputs.get("b"))
        c = int(inputs.get("c"))
        paths = CircuitPaths.for_sum64(self.repo_root)
        out = prove(paths, a=a, b=b, c_public=c)
        return RealProof(proof=out["proof"], public=out["public"])

    def verify_proof(self, *, proof: dict, public_inputs: dict) -> bool:
        if not toolchain_available():
            return False
        paths = CircuitPaths.for_sum64(self.repo_root)
        return verify(paths, proof=proof, public=public_inputs)
