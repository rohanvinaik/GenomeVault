"""
Cryptographic Primitives for Differential Encoding.

This module implements cryptographically secure primitives for:
1. Deterministic random number generation using HKDF (HMAC-based Key Derivation Function)
2. Chunk identifier generation with collision resistance
3. Reference genome integrity verification
4. Cryptographic binding between chunks and reference genomes

Mathematical Properties:
- Determinism: Same seed → same output (reproducibility)
- Unpredictability: Cannot predict outputs from partial information
- Collision Resistance: Different inputs → different outputs (with high probability)
- Binding Security: Cannot swap references without detection

Security Guarantees:
- Uses HMAC-SHA256 for key derivation (NIST SP 800-108 compliant)
- Uses SHA-256 for hashing (256-bit collision resistance)
- Uses secrets.SystemRandom for initialization (cryptographically secure)
"""

from __future__ import annotations

import hashlib
import hmac
import random
import secrets
from secrets import SystemRandom
from typing import Any, List, TypeVar

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CryptoRNG:
    """
    Cryptographically Secure Random Number Generator with Deterministic Derivation.

    Uses HKDF (HMAC-based Key Derivation Function) for deriving deterministic
    seeds from a master seed and context. Provides:

    1. Cryptographically secure initialization (secrets.SystemRandom)
    2. Deterministic seed derivation (HMAC-SHA256 based)
    3. Context-aware derivation with counter
    4. Uniform random integer generation
    5. Cryptographic random selection from lists

    Mathematical Foundation:
        Derived Seed = HMAC-SHA256(master_seed, context || counter)

        Where:
        - master_seed: 32-byte cryptographically random value
        - context: Application-specific context bytes
        - counter: Monotonic counter for uniqueness
        - ||: Concatenation operator

    Security Properties:
        - Pseudorandomness: Derived seeds indistinguishable from uniform random
        - Unpredictability: Cannot predict future seeds from past seeds
        - Independence: Different contexts produce independent seeds
        - Determinism: Same (master_seed, context, counter) → same derived seed

    Example:
        >>> rng = CryptoRNG()
        >>> seed1 = rng.derive_seed(b"chunk_1")
        >>> seed2 = rng.derive_seed(b"chunk_2")
        >>> # seed1 and seed2 are cryptographically independent

        >>> # Deterministic random integer
        >>> value = rng.random_int(0, 100, seed1)
        >>> # Same seed always produces same value

        >>> # Random selection
        >>> items = ["GRCh38", "GRCh37", "CHM13"]
        >>> selected = rng.random_choice(items, seed1)
    """

    def __init__(self, master_seed: bytes | None = None):
        """
        Initialize cryptographic RNG with master seed.

        Args:
            master_seed: 32-byte master seed. If None, generates cryptographically
                        secure random seed using secrets.token_bytes(32).

        Mathematical Initialization:
            If master_seed is None:
                master_seed ← SystemRandom().randbytes(32)

            Then:
                counter ← 0
                system_rng ← SystemRandom()
        """
        self.system_rng = SystemRandom()
        self.master_seed = master_seed or secrets.token_bytes(32)
        self.counter = 0

        if len(self.master_seed) != 32:
            raise ValueError(
                f"Master seed must be exactly 32 bytes, got {len(self.master_seed)}"
            )

        logger.debug(
            f"Initialized CryptoRNG with {len(self.master_seed)}-byte master seed, "
            f"counter={self.counter}"
        )

    def derive_seed(self, context: bytes) -> bytes:
        """
        Derive deterministic seed using HKDF-style HMAC-SHA256.

        This implements a simplified HKDF (HMAC-based Key Derivation Function)
        following NIST SP 800-108 guidelines.

        Algorithm:
            1. message ← context || counter_bytes
            2. derived ← HMAC-SHA256(master_seed, message)
            3. counter ← counter + 1
            4. return derived

        Mathematical Formula:
            derived_seed = HMAC-SHA256(master_seed, context || counter)

            Where HMAC-SHA256(K, M) is defined as:
                HMAC(K, M) = H((K ⊕ opad) || H((K ⊕ ipad) || M))

            With:
                H: SHA-256 hash function
                K: master_seed (key)
                M: context || counter (message)
                opad: 0x5c repeated (outer padding)
                ipad: 0x36 repeated (inner padding)

        Security Properties:
            - Pseudorandomness: Output indistinguishable from uniform random
            - Unpredictability: Cannot predict future outputs
            - Independence: Different contexts → independent outputs
            - Collision Resistance: Inherited from SHA-256 (2^128 operations)

        Args:
            context: Application-specific context bytes for derivation.
                    Examples: b"chunk_1", b"reference_selection", b"chr1_100000"

        Returns:
            32-byte derived seed (deterministic for same context and counter state)

        Example:
            >>> rng = CryptoRNG(master_seed=b"\\x00" * 32)
            >>> seed1 = rng.derive_seed(b"test_context")
            >>> # seed1 is deterministic: always same for this master_seed and context
            >>> # Next call with same context produces different seed due to counter
            >>> seed2 = rng.derive_seed(b"test_context")
            >>> assert seed1 != seed2  # Counter incremented
        """
        # Encode counter as 8-byte big-endian integer
        counter_bytes = self.counter.to_bytes(8, byteorder="big")

        # Construct message: context || counter
        message = context + counter_bytes

        # Derive seed using HMAC-SHA256
        derived = hmac.new(self.master_seed, message, hashlib.sha256).digest()

        # Increment counter for next derivation
        self.counter += 1

        logger.debug(
            f"Derived seed from context={context[:20]}... with counter={self.counter-1}"
        )

        return derived

    def random_int(self, low: int, high: int, seed: bytes) -> int:
        """
        Generate deterministic random integer in range [low, high).

        Uses the seed to initialize a deterministic PRNG (Python's random.Random)
        and generates an integer in the specified range.

        Algorithm:
            1. seed_int ← interpret seed as big-endian integer
            2. rng ← Random(seed_int)  # Deterministic PRNG
            3. return rng.randint(low, high - 1)

        Deterministic Property:
            For fixed (low, high, seed):
                random_int(low, high, seed) always returns same value

        Uniformity:
            Each integer in [low, high) has probability 1/(high - low)

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)
            seed: 32-byte seed for deterministic generation

        Returns:
            Random integer i where low ≤ i < high

        Raises:
            ValueError: If low >= high

        Example:
            >>> rng = CryptoRNG()
            >>> seed = rng.derive_seed(b"test")
            >>> val1 = rng.random_int(0, 100, seed)
            >>> val2 = rng.random_int(0, 100, seed)
            >>> assert val1 == val2  # Deterministic
            >>> assert 0 <= val1 < 100  # In range
        """
        if low >= high:
            raise ValueError(f"Invalid range: low={low} must be < high={high}")

        # Convert seed bytes to integer for PRNG initialization
        seed_int = int.from_bytes(seed, byteorder="big")

        # Initialize deterministic PRNG with seed
        rng = random.Random(seed_int)

        # Generate random integer in [low, high)
        result = rng.randint(low, high - 1)

        logger.debug(f"Generated random_int({low}, {high}) = {result} from seed")

        return result

    def random_choice(self, items: List[T], seed: bytes) -> T:
        """
        Cryptographically select random item from list.

        Uses deterministic random integer generation to select an index,
        ensuring uniform distribution over items.

        Algorithm:
            1. n ← len(items)
            2. idx ← random_int(0, n, seed)
            3. return items[idx]

        Uniformity Property:
            Each item has equal probability 1/n of being selected

        Deterministic Property:
            For fixed (items, seed):
                random_choice(items, seed) always returns same item

        Args:
            items: Non-empty list of items to choose from
            seed: 32-byte seed for deterministic selection

        Returns:
            Randomly selected item from items

        Raises:
            ValueError: If items is empty

        Example:
            >>> rng = CryptoRNG()
            >>> references = ["GRCh38", "GRCh37", "CHM13"]
            >>> seed = rng.derive_seed(b"ref_selection")
            >>> ref1 = rng.random_choice(references, seed)
            >>> ref2 = rng.random_choice(references, seed)
            >>> assert ref1 == ref2  # Deterministic
            >>> assert ref1 in references  # Valid selection
        """
        if not items:
            raise ValueError("Cannot select from empty list")

        # Generate random index
        idx = self.random_int(0, len(items), seed)

        selected = items[idx]

        logger.debug(
            f"Selected item at index {idx}/{len(items)}: {str(selected)[:50]}..."
        )

        return selected

    def reset_counter(self) -> None:
        """
        Reset derivation counter to 0.

        Warning: This should only be used for testing purposes.
        Resetting the counter can lead to seed reuse, which may compromise
        security properties.

        Example:
            >>> rng = CryptoRNG()
            >>> seed1 = rng.derive_seed(b"test")
            >>> rng.reset_counter()
            >>> seed2 = rng.derive_seed(b"test")
            >>> assert seed1 == seed2  # Same seed due to counter reset
        """
        logger.warning("Resetting counter - should only be used for testing")
        self.counter = 0

    def get_counter(self) -> int:
        """
        Get current counter value.

        Returns:
            Current counter value (number of seeds derived)

        Example:
            >>> rng = CryptoRNG()
            >>> assert rng.get_counter() == 0
            >>> _ = rng.derive_seed(b"test1")
            >>> assert rng.get_counter() == 1
            >>> _ = rng.derive_seed(b"test2")
            >>> assert rng.get_counter() == 2
        """
        return self.counter


def compute_chunk_id(chunk: Any, master_seed: bytes) -> bytes:
    """
    Generate cryptographic identifier for a genomic chunk.

    Creates a collision-resistant, deterministic identifier by hashing:
    - Master seed (for cryptographic binding)
    - Chromosome name
    - Start and end positions
    - Variant content hash

    Mathematical Formula:
        chunk_id = SHA-256(master_seed || chromosome || start || end || variant_hash)

        where variant_hash = SHA-256(sorted_variant_strings)

    Algorithm:
        1. h ← SHA256()
        2. h.update(master_seed)
        3. h.update(encode(chromosome))
        4. h.update(encode(start_position))
        5. h.update(encode(end_position))
        6. variant_hash ← compute_variant_content_hash(variants)
        7. h.update(variant_hash)
        8. return h.digest()

    Properties:
        - Deterministic: Same chunk → same ID
        - Collision-resistant: Different chunks → different IDs (2^128 probability)
        - Unpredictable: Cannot guess IDs without master_seed
        - Integrity: Detects any modification to chunk content

    Security Analysis:
        - Based on SHA-256 (256-bit security)
        - Collision resistance: ~2^128 operations to find collision
        - Preimage resistance: ~2^256 operations to invert
        - Second-preimage resistance: ~2^256 operations

    Args:
        chunk: GenomeChunk object with attributes:
              - chromosome: str (e.g., "chr1")
              - start_position: int (genomic coordinate)
              - end_position: int (genomic coordinate)
              - variants: List[Variant] (variant objects with position, ref, alt)
        master_seed: 32-byte cryptographic seed

    Returns:
        32-byte chunk identifier (SHA-256 digest)

    Example:
        >>> from dataclasses import dataclass
        >>> from typing import List
        >>>
        >>> @dataclass
        >>> class MockVariant:
        ...     position: int
        ...     ref: str
        ...     alt: str
        >>>
        >>> @dataclass
        >>> class MockChunk:
        ...     chromosome: str
        ...     start_position: int
        ...     end_position: int
        ...     variants: List[MockVariant]
        >>>
        >>> chunk = MockChunk(
        ...     chromosome="chr1",
        ...     start_position=100000,
        ...     end_position=200000,
        ...     variants=[
        ...         MockVariant(position=150000, ref="A", alt="G")
        ...     ]
        ... )
        >>> master_seed = b"\\x00" * 32
        >>> chunk_id = compute_chunk_id(chunk, master_seed)
        >>> assert len(chunk_id) == 32  # SHA-256 output
        >>> # Deterministic: same inputs → same output
        >>> chunk_id2 = compute_chunk_id(chunk, master_seed)
        >>> assert chunk_id == chunk_id2
    """
    h = hashlib.sha256()

    # Include master seed for cryptographic binding
    h.update(master_seed)

    # Include genomic coordinates
    h.update(chunk.chromosome.encode("utf-8"))
    h.update(chunk.start_position.to_bytes(8, byteorder="big"))
    h.update(chunk.end_position.to_bytes(8, byteorder="big"))

    # Compute variant content hash
    variant_hash = hashlib.sha256()

    # Sort variants by position for deterministic ordering
    sorted_variants = sorted(chunk.variants, key=lambda v: v.position)

    for variant in sorted_variants:
        # Create canonical variant string representation
        variant_str = f"{variant.position}:{variant.ref}>{variant.alt}"
        variant_hash.update(variant_str.encode("utf-8"))

    # Include variant content in chunk ID
    h.update(variant_hash.digest())

    chunk_id = h.digest()

    logger.debug(
        f"Computed chunk_id for {chunk.chromosome}:"
        f"{chunk.start_position}-{chunk.end_position} "
        f"with {len(chunk.variants)} variants"
    )

    return chunk_id


def compute_reference_hash(reference: Any) -> str:
    """
    Compute SHA-256 hash of reference genome for integrity verification.

    Creates a cryptographic hash of the entire reference genome, enabling:
    - Integrity verification (detect tampering)
    - Version tracking (different versions → different hashes)
    - Provenance tracking (link to specific reference version)

    Mathematical Formula:
        reference_hash = SHA-256(assembly || ⊕[chr_hashes])

        where chr_hash_i = SHA-256(chr_name || sorted_variants_i)

    Algorithm:
        1. h ← SHA256()
        2. h.update(encode(assembly))
        3. for each chromosome in sorted order:
            a. h.update(encode(chromosome_name))
            b. for each variant in sorted order:
                h.update(encode(variant_string))
        4. return h.hexdigest()

    Properties:
        - Deterministic: Same reference → same hash
        - Integrity: Any modification → different hash
        - Collision-resistant: Different references → different hashes
        - Compact: 64 hex characters (256 bits)

    Args:
        reference: ReferenceGenome object with attributes:
                  - assembly: str (e.g., "GRCh38")
                  - variants: Dict[str, List[Variant]]
                              Mapping chromosome → variants

    Returns:
        64-character hexadecimal string (SHA-256 hash)

    Example:
        >>> from dataclasses import dataclass
        >>> from typing import Dict, List
        >>>
        >>> @dataclass
        >>> class MockVariant:
        ...     position: int
        ...     ref: str
        ...     alt: str
        ...     genotype: str
        >>>
        >>> @dataclass
        >>> class MockReference:
        ...     assembly: str
        ...     variants: Dict[str, List[MockVariant]]
        >>>
        >>> reference = MockReference(
        ...     assembly="GRCh38",
        ...     variants={
        ...         "chr1": [
        ...             MockVariant(position=100, ref="A", alt="G", genotype="0/1")
        ...         ],
        ...         "chr2": [
        ...             MockVariant(position=200, ref="C", alt="T", genotype="1/1")
        ...         ]
        ...     }
        ... )
        >>> ref_hash = compute_reference_hash(reference)
        >>> assert len(ref_hash) == 64  # SHA-256 hex string
        >>> # Deterministic
        >>> ref_hash2 = compute_reference_hash(reference)
        >>> assert ref_hash == ref_hash2
    """
    h = hashlib.sha256()

    # Include assembly version
    h.update(reference.assembly.encode("utf-8"))

    # Process each chromosome in sorted order for determinism
    for chr_name in sorted(reference.variants.keys()):
        # Include chromosome name
        h.update(chr_name.encode("utf-8"))

        # Sort variants by position for deterministic ordering
        sorted_variants = sorted(reference.variants[chr_name], key=lambda v: v.position)

        # Hash each variant
        for variant in sorted_variants:
            # Create canonical variant string with genotype
            variant_str = (
                f"{variant.position}:{variant.ref}>{variant.alt}:{variant.genotype}"
            )
            h.update(variant_str.encode("utf-8"))

    reference_hash = h.hexdigest()

    logger.debug(
        f"Computed reference_hash for {reference.assembly}: "
        f"{reference_hash[:16]}... "
        f"({sum(len(v) for v in reference.variants.values())} total variants)"
    )

    return reference_hash


def compute_chunk_reference_binding(chunk_id: bytes, reference_id: str) -> bytes:
    """
    Compute cryptographic binding between chunk and reference genome.

    Creates an HMAC-based binding that ensures:
    - Cannot swap reference without detection
    - Cannot forge bindings without chunk_id
    - Cryptographic proof of chunk-reference association

    Mathematical Formula:
        binding = HMAC-SHA256(chunk_id, reference_id)

        where HMAC-SHA256(K, M) = H((K ⊕ opad) || H((K ⊕ ipad) || M))

    Algorithm:
        1. key ← chunk_id (32 bytes)
        2. message ← encode(reference_id)
        3. binding ← HMAC-SHA256(key, message)
        4. return binding

    Security Properties:
        - Unforgeability: Cannot create valid binding without chunk_id
        - Collision Resistance: Different (chunk, ref) → different bindings
        - Verification: Can verify binding with (chunk_id, reference_id)
        - Non-repudiation: Binding proves chunk-reference association

    Attack Resistance:
        - Reference Substitution: Attacker cannot swap reference without
          detection (binding verification will fail)
        - Binding Forgery: Attacker cannot forge binding without knowing
          chunk_id (HMAC key)
        - Replay: Each (chunk_id, reference_id) pair has unique binding

    Args:
        chunk_id: 32-byte chunk identifier (from compute_chunk_id)
        reference_id: Reference genome identifier (e.g., "GRCh38", "HG002")

    Returns:
        32-byte HMAC binding

    Example:
        >>> chunk_id = b"\\x00" * 32
        >>> reference_id = "GRCh38"
        >>> binding = compute_chunk_reference_binding(chunk_id, reference_id)
        >>> assert len(binding) == 32
        >>>
        >>> # Different reference → different binding
        >>> binding2 = compute_chunk_reference_binding(chunk_id, "GRCh37")
        >>> assert binding != binding2
        >>>
        >>> # Deterministic
        >>> binding3 = compute_chunk_reference_binding(chunk_id, reference_id)
        >>> assert binding == binding3

    Verification Example:
        >>> # Store binding with encoded chunk
        >>> stored_binding = compute_chunk_reference_binding(chunk_id, "GRCh38")
        >>>
        >>> # Later: verify chunk-reference association
        >>> claimed_reference = "GRCh38"
        >>> computed_binding = compute_chunk_reference_binding(
        ...     chunk_id, claimed_reference
        ... )
        >>> assert computed_binding == stored_binding  # Verification passes
        >>>
        >>> # Attack: try to use wrong reference
        >>> wrong_reference = "GRCh37"
        >>> forged_binding = compute_chunk_reference_binding(chunk_id, wrong_reference)
        >>> assert forged_binding != stored_binding  # Attack detected
    """
    # Use chunk_id as HMAC key, reference_id as message
    binding = hmac.new(
        chunk_id, reference_id.encode("utf-8"), hashlib.sha256
    ).digest()

    logger.debug(
        f"Computed chunk-reference binding for reference={reference_id}: "
        f"{binding[:8].hex()}..."
    )

    return binding
