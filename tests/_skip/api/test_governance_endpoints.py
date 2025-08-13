from fastapi.testclient import TestClient

from genomevault.api.app import app

client = TestClient(app)


def test_consent_flow_and_dsar_endpoints():
    # grant
    r = client.post(
        "/governance/consent/grant",
        json={"subject_id": "s1", "scope": "research", "ttl_days": 1},
    )
    assert r.status_code == 200 and r.json().get("ok") is True
    # check active
    r = client.get("/governance/consent/check", params={"subject_id": "s1", "scope": "research"})
    assert r.status_code == 200 and r.json()["active"] is True
    # dsar export
    r = client.post("/governance/dsar/export", json={"subject_id": "s1"})
    assert r.status_code == 200 and r.json()["redacted"] is True
    # revoke
    r = client.post("/governance/consent/revoke", json={"subject_id": "s1", "scope": "research"})
    assert r.status_code == 200
    r = client.get("/governance/consent/check", params={"subject_id": "s1", "scope": "research"})
    assert r.status_code == 200 and r.json()["active"] in (False,)
