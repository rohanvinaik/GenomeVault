"""
Training Attestation Smart Contract for GenomeVault

This contract records cryptographic proofs of ML model training on-chain,
enabling immutable audit trails for clinical AI/ML systems.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class AttestationStatus(Enum):
    """Status of a training attestation"""

    PENDING = "pending"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    REVOKED = "revoked"


@dataclass
class TrainingAttestation:
    """On-chain attestation of model training"""

    attestation_id: str
    model_hash: str
    dataset_hash: str
    training_start: int
    training_end: int
    snapshot_merkle_root: str
    proof_hash: str
    submitter: str
    timestamp: int
    status: AttestationStatus
    metadata: dict[str, Any]


@dataclass
class VerificationRecord:
    """Record of attestation verification"""

    verification_id: str
    attestation_id: str
    verifier: str
    timestamp: int
    result: bool
    evidence_hash: str
    notes: str


class TrainingAttestationContract:
    """
    Smart contract for recording and verifying ML model training attestations.

    This contract enables:
    1. Recording training proofs on-chain
    2. Multi-party verification of attestations
    3. Dispute resolution for contested models
    4. Audit trail for regulatory compliance
    """

    def __init__(self, contract_address: str, chain_id: int):
        self.contract_address = contract_address
        self.chain_id = chain_id
        self.attestations: dict[str, TrainingAttestation] = {}
        self.verifications: dict[str, list[VerificationRecord]] = {}
        self.model_to_attestations: dict[str, list[str]] = {}
        self.pending_attestations: list[str] = []
        self.dispute_threshold = 3  # Number of negative verifications to trigger dispute

        # Contract state
        self.owner = None
        self.authorized_verifiers: list[str] = []
        self.paused = False

        logger.info(f"Training attestation contract deployed at {contract_address}")

    def initialize(self, owner: str, initial_verifiers: list[str]):
        """Initialize contract with owner and initial verifiers"""
        if self.owner is not None:
            raise Exception("Contract already initialized")

        self.owner = owner
        self.authorized_verifiers = initial_verifiers

        logger.info(
            f"Contract initialized with owner {owner} and {len(initial_verifiers)} verifiers"
        )

    def submit_attestation(
        self,
        model_hash: str,
        dataset_hash: str,
        training_start: int,
        training_end: int,
        snapshot_merkle_root: str,
        proof_hash: str,
        submitter: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Submit a new training attestation to the blockchain.

        Args:
            model_hash: Hash of the final trained model
            dataset_hash: Hash of the training dataset
            training_start: Unix timestamp of training start
            training_end: Unix timestamp of training end
            snapshot_merkle_root: Merkle root of training snapshots
            proof_hash: Hash of the ZK proof
            submitter: Address of the attestation submitter
            metadata: Additional metadata (model type, hyperparameters, etc.)

        Returns:
            Attestation ID
        """
        if self.paused:
            raise Exception("Contract is paused")

        # Generate attestation ID
        attestation_data = f"{model_hash}{dataset_hash}{training_start}{training_end}{submitter}"
        attestation_id = hashlib.sha256(attestation_data.encode()).hexdigest()[:16]

        # Check for duplicate
        if attestation_id in self.attestations:
            raise Exception(f"Attestation {attestation_id} already exists")

        # Create attestation
        attestation = TrainingAttestation(
            attestation_id=attestation_id,
            model_hash=model_hash,
            dataset_hash=dataset_hash,
            training_start=training_start,
            training_end=training_end,
            snapshot_merkle_root=snapshot_merkle_root,
            proof_hash=proof_hash,
            submitter=submitter,
            timestamp=int(time.time()),
            status=AttestationStatus.PENDING,
            metadata=metadata or {},
        )

        # Store attestation
        self.attestations[attestation_id] = attestation
        self.pending_attestations.append(attestation_id)

        # Index by model hash
        if model_hash not in self.model_to_attestations:
            self.model_to_attestations[model_hash] = []
        self.model_to_attestations[model_hash].append(attestation_id)

        # Emit event (in real blockchain, this would be an actual event)
        self._emit_event(
            "AttestationSubmitted",
            {
                "attestation_id": attestation_id,
                "model_hash": model_hash,
                "submitter": submitter,
                "timestamp": attestation.timestamp,
            },
        )

        logger.info(f"Attestation {attestation_id} submitted for model {model_hash[:8]}...")

        return attestation_id

    def verify_attestation(
        self,
        attestation_id: str,
        verifier: str,
        verification_result: bool,
        evidence_hash: str,
        notes: str = "",
    ) -> str:
        """
        Submit verification for an attestation.

        Args:
            attestation_id: ID of attestation to verify
            verifier: Address of the verifier
            verification_result: True if verification passed
            evidence_hash: Hash of verification evidence
            notes: Optional verification notes

        Returns:
            Verification ID
        """
        if self.paused:
            raise Exception("Contract is paused")

        if verifier not in self.authorized_verifiers:
            raise Exception(f"Verifier {verifier} not authorized")

        if attestation_id not in self.attestations:
            raise Exception(f"Attestation {attestation_id} not found")

        attestation = self.attestations[attestation_id]
        if attestation.status != AttestationStatus.PENDING:
            raise Exception("Attestation not in pending status")

        # Create verification record
        verification_data = f"{attestation_id}{verifier}{verification_result}{evidence_hash}"
        verification_id = hashlib.sha256(verification_data.encode()).hexdigest()[:16]

        verification = VerificationRecord(
            verification_id=verification_id,
            attestation_id=attestation_id,
            verifier=verifier,
            timestamp=int(time.time()),
            result=verification_result,
            evidence_hash=evidence_hash,
            notes=notes,
        )

        # Store verification
        if attestation_id not in self.verifications:
            self.verifications[attestation_id] = []
        self.verifications[attestation_id].append(verification)

        # Update attestation status based on verifications
        self._update_attestation_status(attestation_id)

        # Emit event
        self._emit_event(
            "VerificationSubmitted",
            {
                "verification_id": verification_id,
                "attestation_id": attestation_id,
                "verifier": verifier,
                "result": verification_result,
            },
        )

        logger.info(
            f"Verification {verification_id} submitted for attestation {attestation_id}: "
            f"{'PASSED' if verification_result else 'FAILED'}"
        )

        return verification_id

    def get_attestation(self, attestation_id: str) -> dict[str, Any] | None:
        """Get attestation details"""
        if attestation_id not in self.attestations:
            return None

        attestation = self.attestations[attestation_id]
        result = asdict(attestation)
        result["status"] = attestation.status.value

        # Add verification summary
        if attestation_id in self.verifications:
            verifications = self.verifications[attestation_id]
            result["verification_count"] = len(verifications)
            result["positive_verifications"] = sum(1 for v in verifications if v.result)
            result["negative_verifications"] = sum(1 for v in verifications if not v.result)
        else:
            result["verification_count"] = 0
            result["positive_verifications"] = 0
            result["negative_verifications"] = 0

        return result

    def get_attestations_for_model(self, model_hash: str) -> list[dict[str, Any]]:
        """Get all attestations for a specific model"""
        if model_hash not in self.model_to_attestations:
            return []

        attestations = []
        for attestation_id in self.model_to_attestations[model_hash]:
            attestation_data = self.get_attestation(attestation_id)
            if attestation_data:
                attestations.append(attestation_data)

        return attestations

    def get_verifications(self, attestation_id: str) -> list[dict[str, Any]]:
        """Get all verifications for an attestation"""
        if attestation_id not in self.verifications:
            return []

        return [asdict(v) for v in self.verifications[attestation_id]]

    def dispute_attestation(self, attestation_id: str, disputer: str, evidence_hash: str) -> bool:
        """
        Dispute an attestation.

        Args:
            attestation_id: ID of attestation to dispute
            disputer: Address raising the dispute
            evidence_hash: Hash of dispute evidence

        Returns:
            True if dispute was recorded
        """
        if attestation_id not in self.attestations:
            raise Exception(f"Attestation {attestation_id} not found")

        attestation = self.attestations[attestation_id]
        if attestation.status == AttestationStatus.REVOKED:
            raise Exception("Cannot dispute revoked attestation")

        # Record dispute
        attestation.status = AttestationStatus.DISPUTED

        # Emit event
        self._emit_event(
            "AttestationDisputed",
            {
                "attestation_id": attestation_id,
                "disputer": disputer,
                "evidence_hash": evidence_hash,
                "timestamp": int(time.time()),
            },
        )

        logger.warning(f"Attestation {attestation_id} disputed by {disputer}")

        return True

    def add_verifier(self, verifier: str, added_by: str) -> bool:
        """Add a new authorized verifier (only owner)"""
        if added_by != self.owner:
            raise Exception("Only owner can add verifiers")

        if verifier in self.authorized_verifiers:
            return False

        self.authorized_verifiers.append(verifier)

        self._emit_event(
            "VerifierAdded",
            {"verifier": verifier, "added_by": added_by, "timestamp": int(time.time())},
        )

        return True

    def remove_verifier(self, verifier: str, removed_by: str) -> bool:
        """Remove an authorized verifier (only owner)"""
        if removed_by != self.owner:
            raise Exception("Only owner can remove verifiers")

        if verifier not in self.authorized_verifiers:
            return False

        self.authorized_verifiers.remove(verifier)

        self._emit_event(
            "VerifierRemoved",
            {
                "verifier": verifier,
                "removed_by": removed_by,
                "timestamp": int(time.time()),
            },
        )

        return True

    def pause_contract(self, paused_by: str):
        """Pause contract operations (only owner)"""
        if paused_by != self.owner:
            raise Exception("Only owner can pause contract")

        self.paused = True

        self._emit_event("ContractPaused", {"paused_by": paused_by, "timestamp": int(time.time())})

    def unpause_contract(self, unpaused_by: str):
        """Unpause contract operations (only owner)"""
        if unpaused_by != self.owner:
            raise Exception("Only owner can unpause contract")

        self.paused = False

        self._emit_event(
            "ContractUnpaused",
            {"unpaused_by": unpaused_by, "timestamp": int(time.time())},
        )

    def _update_attestation_status(self, attestation_id: str):
        """Update attestation status based on verifications"""
        attestation = self.attestations[attestation_id]
        verifications = self.verifications.get(attestation_id, [])

        if not verifications:
            return

        positive = sum(1 for v in verifications if v.result)
        negative = sum(1 for v in verifications if not v.result)

        # Check for dispute threshold
        if negative >= self.dispute_threshold:
            attestation.status = AttestationStatus.DISPUTED
            self._emit_event(
                "AttestationAutoDisputed",
                {"attestation_id": attestation_id, "negative_verifications": negative},
            )
        # Check for verification threshold (simple majority)
        elif positive > negative and positive >= len(self.authorized_verifiers) // 2:
            attestation.status = AttestationStatus.VERIFIED
            if attestation_id in self.pending_attestations:
                self.pending_attestations.remove(attestation_id)
            self._emit_event(
                "AttestationVerified",
                {"attestation_id": attestation_id, "positive_verifications": positive},
            )

    def _emit_event(self, event_name: str, data: dict[str, Any]):
        """Emit blockchain event (simulated)"""
        event_data = {
            "event": event_name,
            "contract": self.contract_address,
            "chain_id": self.chain_id,
            "data": data,
            "timestamp": int(time.time()),
        }

        # In a real blockchain, this would emit an actual event
        # For now, just log it
        logger.debug(f"Event emitted: {json.dumps(event_data)}")

    def get_contract_state(self) -> dict[str, Any]:
        """Get current contract state"""
        return {
            "contract_address": self.contract_address,
            "chain_id": self.chain_id,
            "owner": self.owner,
            "paused": self.paused,
            "total_attestations": len(self.attestations),
            "pending_attestations": len(self.pending_attestations),
            "verified_attestations": sum(
                1 for a in self.attestations.values() if a.status == AttestationStatus.VERIFIED
            ),
            "disputed_attestations": sum(
                1 for a in self.attestations.values() if a.status == AttestationStatus.DISPUTED
            ),
            "authorized_verifiers": len(self.authorized_verifiers),
            "total_verifications": sum(len(v) for v in self.verifications.values()),
        }


def create_attestation_hash(
    model_hash: str,
    dataset_hash: str,
    snapshot_merkle_root: str,
    metadata: dict[str, Any],
) -> str:
    """
    Create a deterministic hash for an attestation.

    Args:
        model_hash: Hash of the model
        dataset_hash: Hash of the dataset
        snapshot_merkle_root: Merkle root of snapshots
        metadata: Additional metadata

    Returns:
        Attestation hash
    """
    # Sort metadata for deterministic hashing
    sorted_metadata = json.dumps(metadata, sort_keys=True)

    attestation_data = f"{model_hash}{dataset_hash}{snapshot_merkle_root}{sorted_metadata}"
    return hashlib.sha256(attestation_data.encode()).hexdigest()
