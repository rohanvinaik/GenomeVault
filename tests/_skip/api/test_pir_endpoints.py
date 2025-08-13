from fastapi.testclient import TestClient
from hashlib import sha256
import base64

from genomevault.api.app import app

client = TestClient(app)


def test_pir_query_endpoint_returns_expected_item():
    # Dataset in router: [b"alpha", b"bravo", b"charlie", b"delta"]
    idx = 2
    exp = base64.b64encode(sha256(b"charlie").digest()).decode("ascii")
    r = client.post("/pir/query", json={"index": idx})
    assert r.status_code == 200
    assert r.json()["item_base64"] == exp
