from genomevault.zk.engine import ProofEngine

def test_proof_verify_roundtrip():
    zk = ProofEngine()
    payload = {"x": 3, "y": 4}
    pf = zk.create_proof(payload=payload)
    assert zk.verify_proof(proof=pf, payload=payload) is True


def test_proof_verify_fail_on_tamper():
    zk = ProofEngine()
    payload = {"x": 3, "y": 4}
    pf = zk.create_proof(payload=payload)
    bad = {"x": 3, "y": 5}
    assert zk.verify_proof(proof=pf, payload=bad) is False
