#!/usr/bin/env python3
"""Test the Fiat-Shamir transcript module."""

from genomevault.crypto.transcript import Transcript
from genomevault.crypto.serialization import be_int


def test_transcript():
    """Test Fiat-Shamir transcript functionality."""

    print("=" * 50)

    # Create a transcript
    t = Transcript()

    # Test appending messages
    pass  # Debug print removed

    t.append("commitment", b"\xaa" * 32)
    digest1 = t.digest()
    print(f"  After commitment: {digest1.hex()[:16]}...")

    t.append("public_input", be_int(12345, 4))
    digest2 = t.digest()
    print(f"  After public_input: {digest2.hex()[:16]}...")

    assert digest1 != digest2
    print("  ✓ Digest changes with new messages")

    # Test domain separation
    t2 = Transcript()
    t2.append("public_input", be_int(12345, 4))  # Same value, different label order
    t2.append("commitment", b"\xaa" * 32)

    assert t.digest() != t2.digest()
    print("  ✓ Order matters (non-commutative)")

    # Test length prefixing prevents ambiguity
    t3 = Transcript()
    t3.append("ab", b"cd")

    t4 = Transcript()
    t4.append("a", b"bcd")

    assert t3.digest() != t4.digest()
    print("  ✓ Length prefixing prevents concatenation ambiguity")

    # Test challenge generation
    pass  # Debug print removed

    challenge1 = t.challenge("alpha", 32)
    assert len(challenge1) == 32
    print(f"  Challenge 1: {challenge1.hex()[:16]}...")

    challenge2 = t.challenge("beta", 32)
    assert len(challenge2) == 32
    print(f"  Challenge 2: {challenge2.hex()[:16]}...")

    assert challenge1 != challenge2
    print("  ✓ Different challenges for different labels")

    # Challenges should be deterministic given same transcript state
    t5 = Transcript()
    t5.append("commitment", b"\xaa" * 32)
    t5.append("public_input", be_int(12345, 4))

    challenge5 = t5.challenge("alpha", 32)
    assert challenge1 == challenge5
    print("  ✓ Challenges are deterministic")

    # Test round counter prevents challenge replay
    challenge3 = t.challenge("alpha", 32)  # Same label as challenge1
    assert challenge3 != challenge1
    print("  ✓ Round counter prevents challenge replay")

    # Test variable-length challenges
    pass  # Debug print removed

    short_challenge = t.challenge("short", 16)
    long_challenge = t.challenge("long", 64)

    assert len(short_challenge) == 16
    assert len(long_challenge) == 64
    print(f"  16-byte challenge: {short_challenge.hex()}")
    print(f"  64-byte challenge: {long_challenge.hex()[:32]}...")
    print("  ✓ Supports variable-length challenges")

    # Test typical proof protocol flow
    pass  # Debug print removed

    # Prover's transcript
    prover = Transcript()

    # Step 1: Commit
    commitment = b"\xcc" * 32
    prover.append("commit", commitment)

    # Step 2: Add public inputs
    public_x = be_int(42, 8)
    public_y = be_int(7, 8)
    prover.append("x", public_x)
    prover.append("y", public_y)

    # Step 3: Get challenge
    prover_challenge = prover.challenge("challenge", 32)

    # Step 4: Compute and append response
    response = b"\xdd" * 32  # In real protocol, this would be computed
    prover.append("response", response)

    # Verifier's transcript (should match)
    verifier = Transcript()
    verifier.append("commit", commitment)
    verifier.append("x", public_x)
    verifier.append("y", public_y)

    verifier_challenge = verifier.challenge("challenge", 32)

    assert prover_challenge == verifier_challenge
    print("  ✓ Prover and verifier derive same challenge")

    verifier.append("response", response)

    # Final digests should match
    assert prover.digest() == verifier.digest()
    print("  ✓ Final transcripts match")

    # But different if response is wrong
    verifier2 = Transcript()
    verifier2.append("commit", commitment)
    verifier2.append("x", public_x)
    verifier2.append("y", public_y)
    _ = verifier2.challenge("challenge", 32)
    verifier2.append("response", b"\xee" * 32)  # Wrong response

    assert prover.digest() != verifier2.digest()
    print("  ✓ Different response produces different transcript")

    pass  # Debug print removed


if __name__ == "__main__":
    test_transcript()
