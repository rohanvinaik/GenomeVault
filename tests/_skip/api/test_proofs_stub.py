from fastapi.testclient import TestClient

from genomevault.api.app import app

client = TestClient(app)


def test_proofs_stubs():
    """Test proofs stubs."""
    r = client.post("/proofs/create")
    assert r.status_code == 501
    r2 = client.post("/proofs/verify")
    assert r2.status_code == 501
