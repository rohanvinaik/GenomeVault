from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, Client

from genomevault.api.routers import encode, health


@pytest.fixture
def app():
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(encode.router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    transport = ASGITransport(app=app)
    with Client(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.integration
class TestAPI:
    
    @pytest.mark.unit
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.integration
    def test_encode(self, client):
        """Test encode endpoint."""
        payload = {"data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "seed": 42}
        response = client.post("/encode", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert result["dim"] == 10_000
        assert result["n"] == 2
