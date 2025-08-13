from genomevault.zk.engine import ProofEngine


def test_proof_verify_roundtrip():
    """Test proof verify roundtrip.
    Returns:
        Result of the operation."""
    zk = ProofEngine()
    payload = {"x": 3, "y": 4}
    pf = zk.create_proof(payload=payload)
    assert zk.verify_proof(proof=pf, payload=payload) is True


def test_proof_verify_fail_on_tamper():
    """Test proof verify fail on tamper.
    Returns:
        Result of the operation."""
    zk = ProofEngine()
    payload = {"x": 3, "y": 4}
    pf = zk.create_proof(payload=payload)
    bad = {"x": 3, "y": 5}
    assert zk.verify_proof(proof=pf, payload=bad) is False
