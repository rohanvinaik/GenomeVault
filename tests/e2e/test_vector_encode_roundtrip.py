from fastapi.testclient import TestClient

from genomevault.api.app import app
from genomevault.core.constants import HYPERVECTOR_DIMENSIONS

client = TestClient(app)


def test_vector_encode_roundtrip_e2e():
    dim = HYPERVECTOR_DIMENSIONS["base"]
    r = client.post(
        "/vectors/encode",
        json={
            "data": {"genomic": [1.0, 2.0, 3.0]},
            "dimension": str(dim),
            "compression_tier": "mini",
        },
    )
    assert r.status_code == 200
    vid = r.json()["vector_id"]
    r2 = client.get("/vectors/similarity", params={"vector_id1": vid, "vector_id2": vid})
    assert r2.status_code == 200
    assert 0.99 <= r2.json()["similarity"] <= 1.0
