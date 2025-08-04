from genomevault.governance.policy.classifier import classify_record


def test_classify_record_uses_policy(tmp_path, monkeypatch):
    policy = tmp_path / "policy.json"
    policy.write_text(
        '{"fields":{"email":"restricted","age":"public"}}', encoding="utf-8"
    )
    monkeypatch.setenv("GV_POLICY_PATH", str(policy))
    rec = {"email": "a@b.com", "age": 33}
    cls = classify_record(rec)
    assert cls.field_levels["email"] == "restricted"
    assert cls.overall == "restricted"
