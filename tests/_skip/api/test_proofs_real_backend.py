from fastapi.testclient import TestClient


from genomevault.api.app import app
from tests.zk._toolcheck import require_toolchain

client = TestClient(app)


def test_api_proofs_sum64_endpoints():
    require_toolchain()
    r = client.post(
        "/proofs/create",
        json={"circuit_type": "sum64", "inputs": {"a": 3, "b": 9, "c": 12}},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    v = client.post(
        "/proofs/verify",
        json={"proof": payload["proof"], "public_inputs": payload["public_inputs"]},
    )
    assert v.status_code == 200, v.text
    assert v.json()["valid"] is True
