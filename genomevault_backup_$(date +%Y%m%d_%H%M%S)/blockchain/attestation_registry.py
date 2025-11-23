"""
Attestation Registry for GenomeVault Blockchain Integration

Lightweight blockchain integration for data provenance and audit trails.
Uses existing smart contracts (VerificationContract.sol) for immutable attestations.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import numpy as np

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class AttestationType(Enum):
    """Types of attestations recorded on-chain"""

    DIFFERENTIAL_ENCODING = auto()
    ZK_PROOF_GENERATION = auto()
    PIR_QUERY = auto()
    HDC_ENCODING = auto()
    KAN_ENCODING = auto()  # For future KAN-HD integration


@dataclass
class AttestationMetadata:
    """Metadata associated with an attestation"""

    compression_ratio: Optional[float] = None
    k_anonymity: Optional[int] = None
    dimension: Optional[int] = None
    processing_time_ms: Optional[float] = None
    security_level: Optional[str] = None
    circuit_type: Optional[str] = None
    verification_status: Optional[bool] = None
    privacy_tier: Optional[str] = None
    additional_data: Optional[dict[str, Any]] = None


@dataclass
class AttestationRecord:
    """Complete attestation record"""

    attestation_id: str
    attestation_type: AttestationType
    input_hash: str
    output_hash: str
    timestamp: int
    metadata: AttestationMetadata
    blockchain_tx: Optional[str] = None
    blockchain_confirmed: bool = False
    gas_used: Optional[int] = None


class AttestationRegistry:
    """
    Lightweight blockchain integration for data provenance.
    Uses existing smart contracts, minimal new infrastructure.
    """

    def __init__(
        self,
        contract_interface: Optional[Any] = None,
        blockchain_enabled: bool = True,
        batch_mode: bool = True,
        batch_size: int = 10,
    ):
        """
        Initialize the attestation registry.

        Args:
            contract_interface: Web3 contract interface (optional for offline mode)
            blockchain_enabled: Whether to actually submit to blockchain
            batch_mode: Batch multiple attestations to reduce gas
            batch_size: Number of attestations per batch
        """
        self.contract_interface = contract_interface
        self.blockchain_enabled = blockchain_enabled
        self.batch_mode = batch_mode
        self.batch_size = batch_size

        # Local attestation storage (before blockchain submission)
        self.pending_attestations: list[AttestationRecord] = []
        self.confirmed_attestations: dict[str, AttestationRecord] = {}
        self.attestation_cache: dict[str, AttestationRecord] = {}

        # Statistics
        self.total_attestations = 0
        self.total_gas_used = 0
        self.total_transactions = 0

        logger.info(
            f"Attestation registry initialized (blockchain={'enabled' if blockchain_enabled else 'disabled'}, "
            f"batch_mode={batch_mode}, batch_size={batch_size})"
        )

    def record_encoding(
        self,
        encoding_id: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Record differential encoding on-chain.

        Args:
            encoding_id: Unique identifier for this encoding
            input_data: Input genomic data (will be hashed)
            output_data: Encoded output (will be hashed)
            metadata: Optional metadata (compression ratio, k-anonymity, etc.)

        Returns:
            Transaction hash for audit trail (or local attestation ID if offline)
        """
        # Hash input and output data
        input_hash = self._compute_hash(input_data)
        output_hash = self._compute_hash(output_data)

        # Create metadata
        attestation_metadata = AttestationMetadata(
            compression_ratio=metadata.get("compression_ratio") if metadata else None,
            k_anonymity=metadata.get("k_anonymity") if metadata else None,
            processing_time_ms=metadata.get("processing_time_ms") if metadata else None,
            additional_data=metadata or {},
        )

        # Create attestation record
        attestation = AttestationRecord(
            attestation_id=encoding_id,
            attestation_type=AttestationType.DIFFERENTIAL_ENCODING,
            input_hash=input_hash,
            output_hash=output_hash,
            timestamp=int(time.time()),
            metadata=attestation_metadata,
        )

        # Submit to blockchain
        return self._submit_attestation(attestation)

    def record_zk_proof(
        self,
        proof_id: str,
        circuit_type: str,
        verification_status: bool,
        proof_data: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Record ZK proof verification result.
        Already matches VerificationContract.sol interface.

        Args:
            proof_id: Unique identifier for the proof
            circuit_type: Type of circuit (variant_presence, etc.)
            verification_status: Whether proof verification succeeded
            proof_data: Optional proof data (will be hashed)

        Returns:
            Transaction hash for audit trail
        """
        # Hash proof data
        proof_hash = self._compute_hash(proof_data) if proof_data else "0x0"

        # Create metadata
        attestation_metadata = AttestationMetadata(
            circuit_type=circuit_type,
            verification_status=verification_status,
            security_level="groth16",
            additional_data=proof_data or {},
        )

        # Create attestation record
        attestation = AttestationRecord(
            attestation_id=proof_id,
            attestation_type=AttestationType.ZK_PROOF_GENERATION,
            input_hash=proof_hash,
            output_hash=proof_hash,  # Same for ZK proofs
            timestamp=int(time.time()),
            metadata=attestation_metadata,
        )

        # Submit to blockchain
        return self._submit_attestation(attestation)

    def record_pir_query(
        self,
        query_id: str,
        query_data: Any,
        privacy_preserved: bool = True,
    ) -> str:
        """
        Record PIR query without revealing query content.

        Args:
            query_id: Unique query identifier
            query_data: Query data (will be hashed, not stored)
            privacy_preserved: Whether privacy was preserved

        Returns:
            Transaction hash for audit trail
        """
        # Hash query (no query content on-chain)
        query_hash = self._compute_hash(query_data)

        # Create metadata (minimal for privacy)
        attestation_metadata = AttestationMetadata(
            security_level="information_theoretic",
            privacy_tier="HIGHLY_SENSITIVE" if privacy_preserved else "SENSITIVE",
            additional_data={"privacy_preserved": privacy_preserved},
        )

        # Create attestation record
        attestation = AttestationRecord(
            attestation_id=query_id,
            attestation_type=AttestationType.PIR_QUERY,
            input_hash=query_hash,
            output_hash=query_hash,  # Same for PIR queries
            timestamp=int(time.time()),
            metadata=attestation_metadata,
        )

        # Submit to blockchain
        return self._submit_attestation(attestation)

    def record_hdc_encoding(
        self,
        encoding_id: str,
        input_data: Any,
        hd_vector: Any,
        dimension: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Record HDC encoding attestation.

        Args:
            encoding_id: Unique identifier
            input_data: Input data (will be hashed)
            hd_vector: Output HD vector (will be hashed)
            dimension: HD vector dimension
            metadata: Optional metadata

        Returns:
            Transaction hash
        """
        input_hash = self._compute_hash(input_data)
        output_hash = self._compute_hash(hd_vector)

        attestation_metadata = AttestationMetadata(
            dimension=dimension,
            processing_time_ms=metadata.get("processing_time_ms") if metadata else None,
            additional_data=metadata or {},
        )

        attestation = AttestationRecord(
            attestation_id=encoding_id,
            attestation_type=AttestationType.HDC_ENCODING,
            input_hash=input_hash,
            output_hash=output_hash,
            timestamp=int(time.time()),
            metadata=attestation_metadata,
        )

        return self._submit_attestation(attestation)

    def _submit_attestation(self, attestation: AttestationRecord) -> str:
        """
        Submit attestation to blockchain (or batch queue).

        Args:
            attestation: Attestation record to submit

        Returns:
            Transaction hash or local attestation ID
        """
        self.total_attestations += 1

        if not self.blockchain_enabled:
            # Offline mode with batch support
            if self.batch_mode:
                self.pending_attestations.append(attestation)
                logger.debug(
                    f"Attestation {attestation.attestation_id} added to offline batch queue "
                    f"({len(self.pending_attestations)}/{self.batch_size})"
                )

                # Submit batch if threshold reached
                if len(self.pending_attestations) >= self.batch_size:
                    return self._submit_batch()
                else:
                    return f"pending:{attestation.attestation_id}"
            else:
                # Immediate local storage
                self.attestation_cache[attestation.attestation_id] = attestation
                logger.debug(
                    f"Attestation {attestation.attestation_id} stored locally (blockchain disabled)"
                )
                return f"local:{attestation.attestation_id}"

        if self.batch_mode:
            # Batch mode: add to queue
            self.pending_attestations.append(attestation)
            logger.debug(
                f"Attestation {attestation.attestation_id} added to batch queue "
                f"({len(self.pending_attestations)}/{self.batch_size})"
            )

            # Submit batch if threshold reached
            if len(self.pending_attestations) >= self.batch_size:
                return self._submit_batch()
            else:
                # Return pending status
                return f"pending:{attestation.attestation_id}"
        else:
            # Immediate mode: submit to blockchain
            return self._submit_single(attestation)

    def _submit_single(self, attestation: AttestationRecord) -> str:
        """
        Submit single attestation to blockchain.

        Args:
            attestation: Attestation to submit

        Returns:
            Transaction hash
        """
        if not self.contract_interface:
            logger.warning("No contract interface configured, storing locally")
            self.attestation_cache[attestation.attestation_id] = attestation
            return f"local:{attestation.attestation_id}"

        try:
            # Call smart contract based on attestation type
            if attestation.attestation_type == AttestationType.ZK_PROOF_GENERATION:
                tx_hash = self.contract_interface.record_proof(
                    proof_id=attestation.attestation_id.encode(),
                    circuit_type=attestation.metadata.circuit_type or "unknown",
                    verification_result=attestation.metadata.verification_status or False,
                    metadata_hash=attestation.input_hash.encode(),
                )
            else:
                # Generic attestation (use recordProof for now)
                tx_hash = self.contract_interface.record_proof(
                    proof_id=attestation.attestation_id.encode(),
                    circuit_type=attestation.attestation_type.name.lower(),
                    verification_result=True,
                    metadata_hash=attestation.output_hash.encode(),
                )

            # Update attestation record
            attestation.blockchain_tx = tx_hash
            attestation.blockchain_confirmed = True
            self.confirmed_attestations[attestation.attestation_id] = attestation
            self.total_transactions += 1

            logger.info(
                f"Attestation {attestation.attestation_id} submitted to blockchain: {tx_hash}"
            )
            return tx_hash

        except Exception as e:
            logger.error(f"Failed to submit attestation to blockchain: {e}")
            # Fallback to local storage
            self.attestation_cache[attestation.attestation_id] = attestation
            return f"local:{attestation.attestation_id}"

    def _submit_batch(self) -> str:
        """
        Submit batch of attestations to blockchain.

        Returns:
            Transaction hash of batch submission
        """
        if not self.pending_attestations:
            return ""

        if not self.contract_interface:
            logger.warning("No contract interface, storing batch locally")
            for attestation in self.pending_attestations:
                self.attestation_cache[attestation.attestation_id] = attestation
            batch_size = len(self.pending_attestations)
            self.pending_attestations.clear()
            self.total_transactions += 1  # Count batch as one transaction
            logger.debug(f"Offline batch of {batch_size} attestations stored locally")
            return "local:batch"

        try:
            # Prepare batch data
            proof_ids = [a.attestation_id.encode() for a in self.pending_attestations]
            circuit_types = [
                a.attestation_type.name.lower() for a in self.pending_attestations
            ]
            metadata_hashes = [a.output_hash.encode() for a in self.pending_attestations]

            # Submit batch
            tx_hash = self.contract_interface.batch_record_proofs(
                proof_ids=proof_ids,
                circuit_types=circuit_types,
                metadata_hashes=metadata_hashes,
            )

            # Update all attestations
            for attestation in self.pending_attestations:
                attestation.blockchain_tx = tx_hash
                attestation.blockchain_confirmed = True
                self.confirmed_attestations[attestation.attestation_id] = attestation

            batch_size = len(self.pending_attestations)
            self.total_transactions += 1
            self.pending_attestations.clear()

            logger.info(f"Batch of {batch_size} attestations submitted: {tx_hash}")
            return tx_hash

        except Exception as e:
            logger.error(f"Failed to submit batch to blockchain: {e}")
            # Fallback to local storage
            for attestation in self.pending_attestations:
                self.attestation_cache[attestation.attestation_id] = attestation
            self.pending_attestations.clear()
            return "local:batch"

    def flush_pending(self) -> Optional[str]:
        """
        Flush any pending attestations to blockchain.

        Returns:
            Transaction hash if any were submitted
        """
        if self.pending_attestations:
            logger.info(f"Flushing {len(self.pending_attestations)} pending attestations")
            return self._submit_batch()
        return None

    def get_attestation(self, attestation_id: str) -> Optional[AttestationRecord]:
        """
        Retrieve attestation record.

        Args:
            attestation_id: Attestation identifier

        Returns:
            Attestation record if found
        """
        # Check confirmed first
        if attestation_id in self.confirmed_attestations:
            return self.confirmed_attestations[attestation_id]

        # Check cache
        if attestation_id in self.attestation_cache:
            return self.attestation_cache[attestation_id]

        # Check pending
        for attestation in self.pending_attestations:
            if attestation.attestation_id == attestation_id:
                return attestation

        return None

    def get_statistics(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_attestations": self.total_attestations,
            "confirmed_attestations": len(self.confirmed_attestations),
            "pending_attestations": len(self.pending_attestations),
            "cached_attestations": len(self.attestation_cache),
            "total_transactions": self.total_transactions,
            "total_gas_used": self.total_gas_used,
            "blockchain_enabled": self.blockchain_enabled,
            "batch_mode": self.batch_mode,
        }

    def _compute_hash(self, data: Any) -> str:
        """
        Compute SHA-256 hash of data.

        Args:
            data: Data to hash (numpy array, dict, bytes, or string)

        Returns:
            Hex-encoded hash
        """
        if data is None:
            return "0x" + "0" * 64

        if isinstance(data, np.ndarray):
            # Hash numpy array
            return "0x" + hashlib.sha256(data.tobytes()).hexdigest()
        elif isinstance(data, dict):
            # Hash dictionary (sorted keys for determinism)
            import json

            json_str = json.dumps(data, sort_keys=True)
            return "0x" + hashlib.sha256(json_str.encode()).hexdigest()
        elif isinstance(data, bytes):
            # Hash bytes directly
            return "0x" + hashlib.sha256(data).hexdigest()
        elif isinstance(data, str):
            # Hash string
            return "0x" + hashlib.sha256(data.encode()).hexdigest()
        else:
            # Convert to string and hash
            return "0x" + hashlib.sha256(str(data).encode()).hexdigest()


def create_attestation_registry(
    blockchain_config: Optional[dict[str, Any]] = None,
) -> AttestationRegistry:
    """
    Factory function to create attestation registry from configuration.

    Args:
        blockchain_config: Configuration dictionary

    Returns:
        Configured attestation registry
    """
    if not blockchain_config:
        # Default: blockchain disabled
        return AttestationRegistry(blockchain_enabled=False)

    enabled = blockchain_config.get("enabled", False)
    batch_mode = blockchain_config.get("attestation", {}).get("batch_mode", True)
    batch_size = blockchain_config.get("attestation", {}).get("batch_size", 10)

    if not enabled:
        return AttestationRegistry(blockchain_enabled=False)

    # Try to create contract interface
    try:
        from genomevault.blockchain.contract_interface import ContractInterface

        network = blockchain_config.get("network", "polygon-mumbai")
        contract_address = blockchain_config.get("contract_address")

        if contract_address:
            contract_interface = ContractInterface(network, contract_address)
        else:
            logger.warning("No contract address configured, blockchain disabled")
            contract_interface = None
    except Exception as e:
        logger.warning(f"Failed to create contract interface: {e}")
        contract_interface = None

    return AttestationRegistry(
        contract_interface=contract_interface,
        blockchain_enabled=enabled and contract_interface is not None,
        batch_mode=batch_mode,
        batch_size=batch_size,
    )
