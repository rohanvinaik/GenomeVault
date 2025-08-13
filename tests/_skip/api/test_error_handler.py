from fastapi.testclient import TestClient

from genomevault.api.app import app

client = TestClient(app)


def test_error_handler_returns_400_for_domain_error():
    """Test error handler returns 400 for domain error."""
    r = client.post(
        "/vectors/encode",
        json={
            "data": {"genomic": [1, 2]},
            "dimension": "12345",
            "compression_tier": "mini",
        },
    )
    assert r.status_code == 400
    j = r.json()
    assert "code" in j and j["code"] in ("ProjectionError", "GenomeVaultError")
