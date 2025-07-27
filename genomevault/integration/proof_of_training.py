from __future__ import annotations

"""
Proof-of-Training (PoT) attestation pipeline.

- Hashes model snapshots and training logs per config cadence.
- Emits signed attestations (placeholder signer) and optional on-chain submit via client adapter.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from hashlib import blake2b
from typing import Any, Dict, List, Optional


def _b2(b: bytes) -> str:
    return blake2b(b, digest_size=32).hexdigest()


@dataclass
class PoTConfig:
    snapshot_every_steps: int
    signer_id: str


@dataclass
class SnapshotAttestation:
    step: int
    weights_hash: str
    log_hash: str
    signer_id: str
    signature: str
    ts: float


class ProofOfTraining:
    def __init__(self, cfg: PoTConfig, artifact_dir: str = ".gv_artifacts/pot") -> None:
        self.cfg = cfg
        self.dir = artifact_dir
        os.makedirs(self.dir, exist_ok=True)

    @staticmethod
    def _hash_file(path: str) -> str:
        h = blake2b(digest_size=32)
        with open(path, "rb") as f:
            while True:
                b = f.read(8192)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    @staticmethod
    def _sign(signer_id: str, payload: Dict[str, Any]) -> str:
        # Placeholder signature: BLAKE2b(canonical_json + signer_id)
        msg = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return _b2(msg + signer_id.encode())

    def attest_snapshot(self, step: int, weights_path: str, log_path: str) -> SnapshotAttestation:
        w_h = self._hash_file(weights_path)
        l_h = self._hash_file(log_path)
        payload = {
            "step": step,
            "weights_hash": w_h,
            "log_hash": l_h,
            "signer_id": self.cfg.signer_id,
        }
        sig = self._sign(self.cfg.signer_id, payload)
        att = SnapshotAttestation(
            step=step,
            weights_hash=w_h,
            log_hash=l_h,
            signer_id=self.cfg.signer_id,
            signature=sig,
            ts=time.time(),
        )
        # Persist
        out = os.path.join(self.dir, f"attestation_step_{step}.json")
        with open(out, "w") as f:
            json.dump(asdict(att), f, indent=2, sort_keys=True)
        return att

    def maybe_attest(
        self, step: int, weights_path: str, log_path: str
    ) -> Optional[SnapshotAttestation]:
        if step % self.cfg.snapshot_every_steps != 0:
            return None
        return self.attest_snapshot(step, weights_path, log_path)
