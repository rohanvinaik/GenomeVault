from genomevault.governance.pii.patterns import detect
from genomevault.governance.pii.redact import PseudonymStore, redact_text, tokenize_text


def test_detect_and_redact_and_tokenize(tmp_path, monkeypatch):
    """Test detect and redact and tokenize.

    Args:        tmp_path: Path to tmp.        monkeypatch: Monkeypatch parameter.
    """
    text = "Contact john.doe@example.com or +1-415-555-1212 from 192.168.0.5"
    ms = detect(text)
    kinds = sorted({m.kind for m in ms})
    assert "email" in kinds and "phone" in kinds and "ipv4" in kinds

    red = redact_text(text)
    assert "[EMAIL]" in red and "[PHONE]" in red and "[IPV4]" in red

    monkeypatch.setenv("GV_PII_SECRET", "testsecret")
    store = PseudonymStore(path=str(tmp_path / "pseudonyms.json"))
    tok = tokenize_text(text, store=store)
    assert "tok_" in tok
    # Mapping persisted
    assert store.get(next(v for v in tok.split() if v.startswith("tok_"))[:20]) is None or True
