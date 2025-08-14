"""Privacy-preserving genomic proof system.

This module provides high-level APIs for genomic data verification
without revealing the underlying genomic sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from genomevault.crypto import merkle


@dataclass
class GenomicCommitment:
    """Commitment to genomic data using Merkle tree."""

    root: bytes
    size: int
    metadata: Dict[str, Any]

    def to_hex(self) -> str:
        """Return hex representation of root."""
        return self.root.hex()


@dataclass
class SNPProof:
    """Proof of a specific SNP at a genomic position."""

    position: int
    nucleotide: str
    proof_path: List[Tuple[bytes, bool]]
    commitment_root: bytes

    def verify(self) -> bool:
        """Verify this proof against the commitment."""
        base_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
        if self.nucleotide not in base_to_int:
            return False

        return merkle.verify([base_to_int[self.nucleotide]], self.proof_path, self.commitment_root)


@dataclass
class MultiSNPProof:
    """Batch proof for multiple SNPs."""

    proofs: List[SNPProof]
    commitment_root: bytes

    def verify_all(self) -> Dict[int, bool]:
        """Verify all proofs and return results by position."""
        results = {}
        for proof in self.proofs:
            results[proof.position] = proof.verify()
        return results


class GenomicProver:
    """High-level API for genomic proofs."""

    def __init__(self):
        self.base_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.int_to_base = {v: k for k, v in self.base_to_int.items()}

    def commit_genome(
        self, genome_sequence: str, metadata: Optional[Dict[str, Any]] = None
    ) -> GenomicCommitment:
        """Create a privacy-preserving commitment to a genome.

        Args:
            genome_sequence: String of nucleotides (A, C, G, T)
            metadata: Optional metadata (not included in commitment)

        Returns:
            GenomicCommitment with Merkle root
        """
        # Convert genome to integer representation
        genome_ints = []
        for base in genome_sequence.upper():
            if base in self.base_to_int:
                genome_ints.append(self.base_to_int[base])
            else:
                # Handle unknown bases as 'N' -> -1 (will be masked)
                genome_ints.append(255)  # Special value for unknown

        # Create Merkle tree leaves
        leaves = [merkle.leaf_bytes([val]) for val in genome_ints]
        tree = merkle.build(leaves)

        # Store tree for later proof generation (in practice, would be stored securely)
        self._last_tree = tree
        self._last_genome = genome_ints

        return GenomicCommitment(
            root=tree["root"], size=len(genome_sequence), metadata=metadata or {}
        )

    def prove_snp(self, position: int, commitment: GenomicCommitment) -> Optional[SNPProof]:
        """Generate proof for a SNP at a specific position.

        Args:
            position: 0-based position in genome
            commitment: The genomic commitment

        Returns:
            SNPProof if position is valid, None otherwise
        """
        if not hasattr(self, "_last_tree") or not hasattr(self, "_last_genome"):
            raise ValueError("No genome committed yet")

        if position < 0 or position >= len(self._last_genome):
            return None

        # Get the nucleotide at this position
        base_int = self._last_genome[position]
        if base_int == 255:
            return None  # Unknown base

        nucleotide = self.int_to_base[base_int]

        # Generate Merkle proof path
        proof_path = merkle.path(self._last_tree, position)

        return SNPProof(
            position=position,
            nucleotide=nucleotide,
            proof_path=proof_path,
            commitment_root=commitment.root,
        )

    def prove_snps_batch(
        self, positions: List[int], commitment: GenomicCommitment
    ) -> MultiSNPProof:
        """Generate batch proof for multiple SNPs.

        Args:
            positions: List of 0-based positions
            commitment: The genomic commitment

        Returns:
            MultiSNPProof containing individual proofs
        """
        proofs = []
        for pos in positions:
            proof = self.prove_snp(pos, commitment)
            if proof:
                proofs.append(proof)

        return MultiSNPProof(proofs=proofs, commitment_root=commitment.root)

    def verify_disease_marker(
        self, marker_position: int, marker_nucleotide: str, proof: SNPProof
    ) -> bool:
        """Verify a disease marker matches expected value.

        Args:
            marker_position: Expected position of marker
            marker_nucleotide: Expected nucleotide (A, C, G, or T)
            proof: The SNP proof to verify

        Returns:
            True if marker matches and proof is valid
        """
        # Check position matches
        if proof.position != marker_position:
            return False

        # Check nucleotide matches
        if proof.nucleotide != marker_nucleotide:
            return False

        # Verify cryptographic proof
        return proof.verify()


class ClinicalVerifier:
    """Verifier for clinical/research use cases."""

    def __init__(self):
        self.base_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}

    def verify_snp_proof(self, proof: SNPProof) -> bool:
        """Verify a single SNP proof.

        Args:
            proof: The SNP proof to verify

        Returns:
            True if proof is valid
        """
        return proof.verify()

    def verify_disease_panel(
        self, proofs: MultiSNPProof, disease_markers: Dict[int, str]
    ) -> Dict[str, Any]:
        """Verify a panel of disease markers.

        Args:
            proofs: Multi-SNP proof object
            disease_markers: Dict of position -> expected nucleotide

        Returns:
            Verification results with statistics
        """
        results = {
            "valid_proofs": 0,
            "invalid_proofs": 0,
            "matching_markers": 0,
            "mismatched_markers": 0,
            "details": {},
        }

        for proof in proofs.proofs:
            is_valid = proof.verify()

            if is_valid:
                results["valid_proofs"] += 1
            else:
                results["invalid_proofs"] += 1

            # Check if this position is in disease panel
            if proof.position in disease_markers:
                expected = disease_markers[proof.position]
                matches = proof.nucleotide == expected

                if matches:
                    results["matching_markers"] += 1
                else:
                    results["mismatched_markers"] += 1

                results["details"][proof.position] = {
                    "expected": expected,
                    "actual": proof.nucleotide,
                    "matches": matches,
                    "proof_valid": is_valid,
                }

        results["risk_score"] = (
            results["matching_markers"]
            / (results["matching_markers"] + results["mismatched_markers"])
            if (results["matching_markers"] + results["mismatched_markers"]) > 0
            else 0.0
        )

        return results


# Example usage functions
def example_privacy_preserving_diagnosis():
    """Example: Privacy-preserving disease diagnosis."""

    # Simulate a genome with known disease markers
    genome = "ACGTACGTACGTACGT" * 100  # 1600 bases

    # Known disease markers (positions and expected nucleotides)
    disease_markers = {
        5: "C",  # Position 5 should be C
        42: "G",  # Position 42 should be G
        156: "T",  # Position 156 should be T
    }

    # Patient side: Create commitment and proofs
    prover = GenomicProver()
    commitment = prover.commit_genome(genome)

    # Generate proofs only for relevant positions
    positions = list(disease_markers.keys())
    multi_proof = prover.prove_snps_batch(positions, commitment)

    # Verifier side: Check disease markers without seeing genome
    verifier = ClinicalVerifier()
    results = verifier.verify_disease_panel(multi_proof, disease_markers)

    return {
        "commitment": commitment.to_hex()[:16] + "...",
        "genome_size": commitment.size,
        "verification_results": results,
    }


def example_snp_verification():
    """Example: Verify single SNP without revealing genome."""

    # Create a small test genome
    genome = "ACGTACGTACGTACGT"

    # Commit to genome
    prover = GenomicProver()
    commitment = prover.commit_genome(genome)

    # Prove SNP at position 5
    position = 5
    proof = prover.prove_snp(position, commitment)

    # Verify the proof
    verifier = ClinicalVerifier()
    is_valid = verifier.verify_snp_proof(proof)

    return {
        "genome_commitment": commitment.to_hex()[:16] + "...",
        "position": position,
        "nucleotide": proof.nucleotide,
        "proof_valid": is_valid,
        "proof_size_bytes": sum(32 for _ in proof.proof_path),  # Each hash is 32 bytes
    }
