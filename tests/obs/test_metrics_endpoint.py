from fastapi.testclient import TestClient

from genomevault.api.app import app

client = TestClient(app)


def test_metrics_endpoint_exposes_prometheus_text():
    """Test metrics endpoint exposes prometheus text.
    Returns:
        Result of the operation."""
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    # should include at least our counter name or comment if lib missing
    assert (
        "genomevault_http_requests_total" in r.text or "prometheus_client not installed" in r.text
    )
