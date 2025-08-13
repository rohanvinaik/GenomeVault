from fastapi.testclient import TestClient

from genomevault.api.main import app


def test_health_and_status():
    """Test health and status.
    Returns:
        Result of the operation."""
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        r = client.get("/status")
        assert r.status_code == 200
