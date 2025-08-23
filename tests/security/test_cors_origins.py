from fastapi import FastAPI
from fastapi.testclient import TestClient

from genomevault.security import register_security


def test_cors_respects_configured_origins():
    allowed = "https://allowed.example"
    app = FastAPI()
    register_security(app, allow_origins=[allowed])
    client = TestClient(app)

    resp_allowed = client.get("/", headers={"Origin": allowed})
    assert resp_allowed.headers.get("access-control-allow-origin") == allowed

    resp_denied = client.get("/", headers={"Origin": "https://denied.example"})
    assert "access-control-allow-origin" not in resp_denied.headers
