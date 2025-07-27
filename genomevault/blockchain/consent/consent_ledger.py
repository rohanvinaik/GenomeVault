from __future__ import annotations

"""
Consent Ledger with signed consent records and content-addressed IDs.

- Stores records in JSON Lines (append-only) and an index mapping consent_hash -> record path & offset.
- Provides verify_signature hook (app expects to plug in org PKI); default is detached signature over canonical JSON.
- Exposes bind_to_public_inputs() utility to add consent_hash to ZK public inputs.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from hashlib import blake2b
from typing import Any, Dict, Optional

LEDGER_DIR = os.environ.get("GV_CONSENT_LEDGER_DIR", ".gv_artifacts/consent")
LEDGER_LOG = os.path.join(LEDGER_DIR, "ledger.jsonl")
LEDGER_IDX = os.path.join(LEDGER_DIR, "index.json")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _b2(content: bytes) -> str:
    return blake2b(content, digest_size=32).hexdigest()


@dataclass(frozen=True)
class ConsentRecord:
    subject_id: str  # pseudonymous subject ID
    dataset_id: str
    policy_id: str
    policy_version: str
    issued_at: float
    expires_at: Optional[float]
    signer_id: str  # who signed (e.g., org key id)
    signature: str  # hex-encoded signature over canonical content


class ConsentLedger:
    def __init__(self, dir_path: str = LEDGER_DIR) -> None:
        self.dir = dir_path
        _ensure_dir(self.dir)
        self.log_path = os.path.join(self.dir, "ledger.jsonl")
        self.idx_path = os.path.join(self.dir, "index.json")
        self._load_index()

    def _load_index(self) -> None:
        if os.path.exists(self.idx_path):
            with open(self.idx_path, "r") as f:
                self.idx: Dict[str, Dict[str, Any]] = json.load(f)
        else:
            self.idx = {}

    def _save_index(self) -> None:
        with open(self.idx_path, "w") as f:
            json.dump(self.idx, f, indent=2, sort_keys=True)

    # Signature verification hook (replace with org PKI verification as needed)
    @staticmethod
    def verify_signature(record: ConsentRecord) -> bool:
        # Default: treat signature as BLAKE2b(content + signer_id) hex; replace with real PKI in prod.
        content = _canonical(
            {
                "subject_id": record.subject_id,
                "dataset_id": record.dataset_id,
                "policy_id": record.policy_id,
                "policy_version": record.policy_version,
                "issued_at": record.issued_at,
                "expires_at": record.expires_at,
                "signer_id": record.signer_id,
            }
        )
        expected = _b2(content + record.signer_id.encode())
        return expected == record.signature

    def issue(self, rec: ConsentRecord) -> str:
        if not self.verify_signature(rec):
            raise ValueError("Invalid consent signature")
        payload = asdict(rec)
        consent_hash = _b2(_canonical(payload))
        line = json.dumps({"consent_hash": consent_hash, "record": payload}) + "\n"
        with open(self.log_path, "a") as f:
            offset = f.tell()
            f.write(line)
        self.idx[consent_hash] = {
            "path": self.log_path,
            "offset": offset,
            "revoked": False,
            "ts": time.time(),
        }
        self._save_index()
        return consent_hash

    def revoke(self, consent_hash: str, reason: str) -> None:
        meta = self.idx.get(consent_hash)
        if not meta:
            raise KeyError("Unknown consent hash")
        meta["revoked"] = True
        meta["revocation_reason"] = reason
        meta["revoked_ts"] = time.time()
        self._save_index()

    def get(self, consent_hash: str) -> ConsentRecord:
        meta = self.idx.get(consent_hash)
        if not meta:
            raise KeyError("Unknown consent hash")
        with open(meta["path"], "r") as f:
            f.seek(meta["offset"])
            line = f.readline()
        obj = json.loads(line)["record"]
        return ConsentRecord(**obj)


def bind_to_public_inputs(public_inputs: dict, consent_hash: str) -> dict:
    """Attach consent_hash to public inputs for ZK verification; returns new dict."""
    out = dict(public_inputs)
    out["consent_hash"] = consent_hash
    return out
