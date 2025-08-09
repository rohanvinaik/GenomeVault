from __future__ import annotations


from genomevault.zk_proofs.circuits.clinical_circuits import prove, verify


class TestZKProofs:
    def test_prove(self):
        """Test proof generation."""
        payload = {"glucose": 100, "resistance": 0.5}
        proof = prove(payload)
        assert isinstance(proof, dict)
        assert "proof" in proof
        assert "public" in proof
        assert proof["public"]["commitment"] == "deadbeef"

    def test_verify_valid(self):
        """Test verification of valid proof."""
        proof = {"proof": "MOCK", "public": {"commitment": "deadbeef"}}
        assert verify(proof) is True

    def test_verify_invalid(self):
        """Test verification of invalid proof."""
        assert verify({}) is False
        assert verify({"public": {}}) is False
        assert verify(None) is False
