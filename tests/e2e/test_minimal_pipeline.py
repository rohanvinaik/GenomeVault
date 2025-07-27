import os

import numpy as np

from genomevault.blockchain.consent.consent_ledger import (
    ConsentLedger,
    ConsentRecord,
    bind_to_public_inputs,
)
from genomevault.zk_proofs.srs_manager.srs_manager import SRSManager, SRSMetadata


def test_srs_manager_roundtrip(tmp_path):
    os.environ["GV_ZK_ARTIFACT_DIR"] = str(tmp_path / "zk")
    mgr = SRSManager()
    srs_id = mgr.register_srs(
        b"dummy_srs",
        SRSMetadata(curve="bls12-381", size=4096, toolchain="gnark", toolchain_version="vX"),
    )
    srs = mgr.get_srs(srs_id)
    assert srs == b"dummy_srs"
    vk_hash = mgr.register_vk("median_v1", b"vk_bytes", srs_id)
    got_vk_hash, got_srs_id, vk = mgr.get_vk("median_v1")
    assert vk == b"vk_bytes" and got_srs_id == srs_id and got_vk_hash == vk_hash


def test_consent_ledger(tmp_path):
    os.environ["GV_CONSENT_LEDGER_DIR"] = str(tmp_path / "consent")
    led = ConsentLedger()
    rec = ConsentRecord(
        subject_id="subj-1",
        dataset_id="ds-1",
        policy_id="pol-1",
        policy_version="v1",
        issued_at=1234567890.0,
        expires_at=1234567900.0,
        signer_id="test-signer",
        signature=led.verify_signature.__func__(
            ConsentRecord(
                subject_id="subj-1",
                dataset_id="ds-1",
                policy_id="pol-1",
                policy_version="v1",
                issued_at=1234567890.0,
                expires_at=1234567900.0,
                signer_id="test-signer",
                signature="",
            )
        ),
    )
    # Create a valid signature
    from hashlib import blake2b

    content = b'{"dataset_id":"ds-1","expires_at":1234567900.0,"issued_at":1234567890.0,"policy_id":"pol-1","policy_version":"v1","signer_id":"test-signer","subject_id":"subj-1"}'
    sig = blake2b(content + b"test-signer", digest_size=32).hexdigest()

    rec = ConsentRecord(
        subject_id="subj-1",
        dataset_id="ds-1",
        policy_id="pol-1",
        policy_version="v1",
        issued_at=1234567890.0,
        expires_at=1234567900.0,
        signer_id="test-signer",
        signature=sig,
    )

    ch = led.issue(rec)
    assert ch is not None
    got = led.get(ch)
    assert got.subject_id == "subj-1"

    # Test bind_to_public_inputs
    pub_inputs = {"query": "test"}
    bound = bind_to_public_inputs(pub_inputs, ch)
    assert bound["consent_hash"] == ch
    assert bound["query"] == "test"
