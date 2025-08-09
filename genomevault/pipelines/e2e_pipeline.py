from __future__ import annotations

"""E2E Pipeline module."""
"""E2E Pipeline module."""
import numpy as np

from genomevault.governance.ledger import Ledger
from genomevault.hypervector.engine import HypervectorEngine
from genomevault.pir.client import PIRClient
from genomevault.zk.engine import ProofEngine


def run_e2e(seed=123):
    """Run e2e.

        Args:
            seed: Seed.

        Returns:
            Operation result.
        """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(8, 64)).astype("int8")
    hv = HypervectorEngine()
    encoded = hv.encode(data=X.tolist(), dimension=256)
    pir = PIRClient()
    qres = pir.query(index=0, key="demo")
    zk = ProofEngine()
    proof = zk.create_proof(payload={"shape": len(getattr(encoded, "vector", []))})
    led = Ledger()
    tx = led.record(event="e2e_ok", meta={"proof_id": getattr(proof, "id", "N/A")})
    return {"encoded": encoded, "pir": qres, "proof": proof, "tx": tx}
