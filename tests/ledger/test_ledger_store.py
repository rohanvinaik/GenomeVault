from genomevault.ledger.store import InMemoryLedger


def test_ledger_append_and_verify():
    L = InMemoryLedger()
    e1 = L.append({"a": 1})
    e2 = L.append({"b": 2})
    assert e2.index == 1 and L.verify_chain() is True
    assert len(L.entries()) == 2
