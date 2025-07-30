from genomevault.pir.engine import PIREngine
from hashlib import sha256


def test_pir_engine_recovers_record():
    items = [b"alpha", b"bravo", b"charlie", b"delta"]
    eng = PIREngine(items, n_servers=3)
    idx = 2
    out = eng.query(idx)
    assert out == sha256(items[idx]).digest()
