from pathlib import Path

from genomevault.zk.real_engine import RealZKEngine

from ._toolcheck import require_toolchain

REPO_ROOT = Path("/Users/rohanvinaik/genomevault")


def test_sum64_proof_roundtrip():
    require_toolchain()
    eng = RealZKEngine(repo_root=str(REPO_ROOT))
    proof = eng.create_proof(circuit_type="sum64", inputs={"a": 7, "b": 5, "c": 12})
    assert proof.public and "0" in proof.public  # public signals array; index 0 is c
    ok = eng.verify_proof(proof=proof.proof, public_inputs=proof.public)
    assert ok
