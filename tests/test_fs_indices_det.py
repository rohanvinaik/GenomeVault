from genomevault.crypto.rng import xof_uint_mod
from genomevault.crypto.transcript import Transcript


def derive_indices(trace_root_hex: str, cons_root_hex: str, n: int, domain: int):
    """Derive indices.
        Args:        trace_root_hex: Parameter value.        cons_root_hex: Parameter value.    \
            n: Number or count value.        domain: Parameter value.
        Returns:
            Result of the operation.    """
    t = Transcript()
    t.append("trace", bytes.fromhex(trace_root_hex))
    t.append("cons", bytes.fromhex(cons_root_hex))
    out = set()
    c = 0
    while len(out) < n:
        out.add(xof_uint_mod(b"QIDX" + c.to_bytes(4, "big"), t.digest(), domain))
        c += 1
    return sorted(out)


def test_indices_stable():
    """Test indices stable.
    Returns:
        Result of the operation."""
    a = derive_indices("11" * 32, "22" * 32, 16, 8192)
    b = derive_indices("11" * 32, "22" * 32, 16, 8192)
    assert a == b
