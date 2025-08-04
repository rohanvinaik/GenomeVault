from fastapi.testclient import TestClient

from genomevault.api.app import app

client = TestClient(app)


def test_federated_aggregate_endpoint():
    payload = {
        "updates": [
            {"client_id": "a", "weights": [1.0, 1.0], "num_examples": 2},
            {"client_id": "b", "weights": [3.0, 5.0], "num_examples": 1},
        ]
    }
    r = client.post("/federated/aggregate", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert j["client_count"] == 2 and j["total_examples"] == 3
