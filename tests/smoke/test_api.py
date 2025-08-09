from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from genomevault.api.routers import encode, health

# Create test app
app = FastAPI()
app.include_router(health.router)
app.include_router(encode.router)
client = TestClient(app)


class TestAPI:
    def test_health(self):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_encode(self):
        """Test encode endpoint."""
        payload = {"data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "seed": 42}
        response = client.post("/encode", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert result["dim"] == 10_000
        assert result["n"] == 2
