"""
Test script to demonstrate the fixed Merkle proof verification.

This shows that the variant membership proof now properly:
1. Uses direction bits instead of assuming parity
2. Verifies against the expected root
"""

from typing import List, Tuple
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import directly to avoid module initialization issues
import importlib.util

spec = importlib.util.spec_from_file_location(
    "catalytic_proof", os.path.join(os.path.dirname(__file__), "catalytic_proof.py")
)
catalytic_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalytic_module)
CatalyticProofEngine = catalytic_module.CatalyticProofEngine

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class MerkleTree:
    """Simple Merkle tree implementation for testing."""

    def __init__(self, leaves: List[bytes]):
        """Build a Merkle tree from leaves."""
        self.leaves = leaves
        self.tree = self._build_tree(leaves)
        self.root = self.tree[-1][0] if self.tree else b""

    def _build_tree(self, leaves: List[bytes]) -> List[List[bytes]]:
        """Build complete Merkle tree."""
        if not leaves:
            return []

        tree = [leaves]
        current_level = leaves

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    left = current_level[i]
                    right = current_level[i + 1]
                else:
                    # Odd number of nodes - duplicate the last one
                    left = current_level[i]
                    right = current_level[i]

                parent = hashlib.sha256(left + right).digest()
                next_level.append(parent)

            tree.append(next_level)
            current_level = next_level

        return tree

    def get_proof(self, index: int) -> Tuple[List[bytes], List[int]]:
        """
        Get Merkle proof for leaf at index.

        Returns:
            (siblings, directions) where directions[i] = 0 if node is on left, 1 if on right
        """
        if index >= len(self.leaves):
            raise ValueError(f"Index {index} out of range")

        siblings = []
        directions = []

        for level in self.tree[:-1]:  # Exclude root level
            if index % 2 == 0:
                # Node is on the left, sibling is on the right
                sibling_index = index + 1
                direction = 0  # We are on the left
            else:
                # Node is on the right, sibling is on the left
                sibling_index = index - 1
                direction = 1  # We are on the right

            if sibling_index < len(level):
                siblings.append(level[sibling_index])
            else:
                # No sibling, duplicate current node
                siblings.append(level[index])

            directions.append(direction)
            index = index // 2  # Move to parent index

        return siblings, directions


def test_merkle_proof_verification():
    """Test that Merkle proof verification works correctly."""

    logger.info("Testing Merkle Proof Verification Fix")
    logger.info("=" * 60)

    # Create some test variants
    variants = [
        {"chr": "chr1", "pos": 12345, "ref": "A", "alt": "G"},
        {"chr": "chr1", "pos": 23456, "ref": "C", "alt": "T"},
        {"chr": "chr2", "pos": 34567, "ref": "G", "alt": "A"},
        {"chr": "chr2", "pos": 45678, "ref": "T", "alt": "C"},
        {"chr": "chr3", "pos": 56789, "ref": "A", "alt": "T"},
        {"chr": "chr3", "pos": 67890, "ref": "C", "alt": "G"},
        {"chr": "chr4", "pos": 78901, "ref": "G", "alt": "C"},
        {"chr": "chr4", "pos": 89012, "ref": "T", "alt": "A"},
    ]

    # Hash each variant
    variant_hashes = []
    for v in variants:
        variant_str = f"{v['chr']}:{v['pos']}:{v['ref']}:{v['alt']}"
        variant_hash = hashlib.sha256(variant_str.encode()).digest()
        variant_hashes.append(variant_hash)

    # Build Merkle tree
    tree = MerkleTree(variant_hashes)
    root_hex = tree.root.hex()

    logger.info(f"Built Merkle tree with {len(variants)} variants")
    logger.info(f"Root hash: {root_hex[:16]}...")

    # Test proof for variant at index 2
    test_index = 2
    test_variant = variants[test_index]
    siblings, directions = tree.get_proof(test_index)

    logger.info(f"\nTesting proof for variant {test_index}: {test_variant}")
    logger.info(f"Proof path length: {len(siblings)}")
    logger.info(f"Direction bits: {directions}")

    # Initialize catalytic proof engine
    engine = CatalyticProofEngine(
        clean_space_limit=512 * 1024,
        catalytic_space_size=10 * 1024 * 1024,
    )

    # Test 1: Correct proof with proper root verification
    logger.info("\n1. Testing CORRECT proof with matching root:")
    try:
        proof = engine.generate_catalytic_proof(
            circuit_name="variant_presence",
            public_inputs={
                "variant_hash": variant_hashes[test_index].hex(),
                "reference_hash": hashlib.sha256(b"GRCh38").hexdigest(),
                "commitment_root": root_hex,  # Correct root
            },
            private_inputs={
                "variant_data": test_variant,
                "merkle_siblings": [s.hex() for s in siblings],
                "merkle_directions": directions,  # Proper direction bits
                "witness_randomness": os.urandom(32).hex(),
            },
        )

        # Check if verification passed
        import json

        proof_data = json.loads(proof.proof_data.decode()[:256])
        if proof_data.get("verification_passed"):
            logger.info("✓ Proof generated successfully with correct root verification!")
        else:
            logger.error("✗ Proof verification failed even with correct inputs!")
    except Exception as e:
        logger.error(f"✗ Failed to generate proof: {e}")

    # Test 2: Incorrect root should fail
    logger.info("\n2. Testing proof with INCORRECT root (should fail):")
    fake_root = hashlib.sha256(b"fake_root").hexdigest()

    try:
        proof = engine.generate_catalytic_proof(
            circuit_name="variant_presence",
            public_inputs={
                "variant_hash": variant_hashes[test_index].hex(),
                "reference_hash": hashlib.sha256(b"GRCh38").hexdigest(),
                "commitment_root": fake_root,  # Wrong root!
            },
            private_inputs={
                "variant_data": test_variant,
                "merkle_siblings": [s.hex() for s in siblings],
                "merkle_directions": directions,
                "witness_randomness": os.urandom(32).hex(),
            },
        )

        # Check if verification failed as expected
        import json

        proof_data = json.loads(proof.proof_data.decode()[:256])
        if not proof_data.get("verification_passed"):
            logger.info("✓ Proof correctly detected root mismatch!")
            logger.info(f"  Computed: {proof_data.get('computed_root', '')[:16]}...")
            logger.info(f"  Expected: {proof_data.get('expected_root', '')[:16]}...")
        else:
            logger.error("✗ Proof incorrectly passed with wrong root!")
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")

    # Test 3: Wrong direction bits should fail
    logger.info("\n3. Testing proof with WRONG direction bits (should fail):")
    wrong_directions = [1 - d for d in directions]  # Flip all directions

    try:
        proof = engine.generate_catalytic_proof(
            circuit_name="variant_presence",
            public_inputs={
                "variant_hash": variant_hashes[test_index].hex(),
                "reference_hash": hashlib.sha256(b"GRCh38").hexdigest(),
                "commitment_root": root_hex,
            },
            private_inputs={
                "variant_data": test_variant,
                "merkle_siblings": [s.hex() for s in siblings],
                "merkle_directions": wrong_directions,  # Wrong directions!
                "witness_randomness": os.urandom(32).hex(),
            },
        )

        # Check if verification failed as expected
        import json

        proof_data = json.loads(proof.proof_data.decode()[:256])
        if not proof_data.get("verification_passed"):
            logger.info("✓ Proof correctly detected wrong direction bits!")
        else:
            logger.error("✗ Proof incorrectly passed with wrong directions!")
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")

    # Test 4: Missing direction bits should raise error
    logger.info("\n4. Testing proof WITHOUT direction bits (should raise error):")
    try:
        proof = engine.generate_catalytic_proof(
            circuit_name="variant_presence",
            public_inputs={
                "variant_hash": variant_hashes[test_index].hex(),
                "reference_hash": hashlib.sha256(b"GRCh38").hexdigest(),
                "commitment_root": root_hex,
            },
            private_inputs={
                "variant_data": test_variant,
                "merkle_proof": [s.hex() for s in siblings],  # Old style without directions
                # No merkle_directions provided!
                "witness_randomness": os.urandom(32).hex(),
            },
        )
        logger.error("✗ Should have raised an error for missing direction bits!")
    except ValueError as e:
        if "direction bits" in str(e):
            logger.info(f"✓ Correctly raised error: {e}")
        else:
            logger.error(f"✗ Wrong error: {e}")
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Merkle proof verification tests completed!")


if __name__ == "__main__":
    test_merkle_proof_verification()
