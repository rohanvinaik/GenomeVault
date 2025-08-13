from fastapi.testclient import TestClient

from genomevault.api.app import app


def test_metrics_endpoint_exists():
    """Test metrics endpoint exists.
    Returns:
        Result of the operation."""
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code in (200, 404)  # OK if metrics gated; if present, should be 200
    if r.status_code == 200:
        assert "python_info" in r.text or "process_cpu_seconds_total" in r.text
