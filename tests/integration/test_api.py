"""Integration tests for API endpoints."""

from fastapi.testclient import TestClient


def test_health_endpoint():
    """Test health check endpoint."""
    from genomevault.api.main import app

    client = TestClient(app)

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint():
    """Test root endpoint."""
    from genomevault.api.main import app

    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_encode_endpoint():
    """Test encoding endpoint."""
    from genomevault.api.main import app

    client = TestClient(app)

    payload = {"data": [1.0, 2.0, 3.0, 4.0, 5.0], "dimension": 1000, "seed": 42}

    response = client.post("/api/v1/encode", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert "encoded_dimension" in result
    assert "num_samples" in result
    assert result["num_samples"] == 5


def test_encode_info_endpoint():
    """Test encoding info endpoint."""
    from genomevault.api.main import app

    client = TestClient(app)

    response = client.get("/api/v1/encode/info")
    assert response.status_code == 200

    result = response.json()
    assert "supported_dimensions" in result
    assert "encoding_methods" in result
