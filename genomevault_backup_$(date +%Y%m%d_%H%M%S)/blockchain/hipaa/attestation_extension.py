"""
Attestation Registry Extension for HIPAA Phase 2

Extends the base AttestationRegistry with multi-signature institutional attestations.
Integrates with TrustedSignatoryRegistry for institutional verification.

Features:
- Multi-signature attestation support
- Institutional verification requirements
- Enhanced metadata for HIPAA compliance
- Integration with Phase 1 attestation system
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

from genomevault.blockchain.attestation_registry import (
    AttestationMetadata,
    AttestationRecord,
    AttestationRegistry,
    AttestationType,
)
from genomevault.utils.logging import get_logger

from .trusted_signatory_registry import (
    MultiSigAttestation,
    TrustedSignatory,
    TrustedSignatoryRegistry,
)

logger = get_logger(__name__)


class InstitutionalAttestationType(Enum):
    """Types of institutional attestations requiring multi-sig"""

    BULK_DATA_CONTRIBUTION = auto()  # Large dataset contributions
    PATIENT_CONSENT_FRAMEWORK = auto()  # Consent framework changes
    CLINICAL_PROTOCOL_UPDATE = auto()  # Clinical protocol changes
    PHI_ACCESS_GRANT = auto()  # Protected Health Information access
    DATA_SHARING_AGREEMENT = auto()  # Inter-institutional data sharing


@dataclass
class InstitutionalAttestationMetadata(AttestationMetadata):
    """Extended metadata for institutional attestations"""

    # Institutional info
    npi: Optional[str] = None
    institution_name: Optional[str] = None

    # Multi-signature info
    required_signatures: Optional[int] = None
    collected_signatures: Optional[int] = None
    total_signatory_weight: Optional[int] = None

    # HIPAA compliance
    baa_compliant: bool = True
    phi_involved: bool = False
    patient_count: Optional[int] = None

    # Data governance
    data_classification: Optional[str] = None  # "PUBLIC", "SENSITIVE", "HIGHLY_SENSITIVE"
    retention_period_days: Optional[int] = None


class HIPAAAttestationRegistry(AttestationRegistry):
    """
    Extended attestation registry with HIPAA Phase 2 features.

    Adds multi-signature institutional attestations on top of
    Phase 1 single-signature attestations.
    """

    def __init__(
        self,
        contract_interface: Optional[Any] = None,
        blockchain_enabled: bool = True,
        batch_mode: bool = True,
        batch_size: int = 10,
        signatory_registry: Optional[TrustedSignatoryRegistry] = None,
    ):
        """
        Initialize HIPAA attestation registry.

        Args:
            contract_interface: Web3 contract interface
            blockchain_enabled: Whether to use blockchain
            batch_mode: Batch attestations
            batch_size: Attestations per batch
            signatory_registry: Trusted signatory registry for multi-sig
        """
        # Initialize base registry
        super().__init__(
            contract_interface=contract_interface,
            blockchain_enabled=blockchain_enabled,
            batch_mode=batch_mode,
            batch_size=batch_size,
        )

        # Phase 2 components
        self.signatory_registry = signatory_registry

        # Institutional attestation tracking
        self.institutional_attestations: dict[str, MultiSigAttestation] = {}
        self.total_institutional_attestations = 0
        self.total_multi_sig_attestations = 0

        logger.info(
            f"HIPAA attestation registry initialized "
            f"(multi-sig={'enabled' if signatory_registry else 'disabled'})"
        )

    def record_institutional_encoding(
        self,
        encoding_id: str,
        npi: str,
        institution_name: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[dict[str, Any]] = None,
        require_multi_sig: bool = False,
        required_signatures: int = 3,
        required_weight: int = 10,
    ) -> str:
        """
        Record institutional differential encoding with optional multi-sig.

        Args:
            encoding_id: Unique identifier
            npi: National Provider Identifier
            institution_name: Institution name
            input_data: Input genomic data
            output_data: Encoded output
            metadata: Optional metadata
            require_multi_sig: Whether to require multiple signatures
            required_signatures: Number of signatures required
            required_weight: Total signatory weight required

        Returns:
            Transaction hash or attestation ID
        """
        # Compute hashes
        input_hash = self._compute_hash(input_data)
        output_hash = self._compute_hash(output_data)

        # Create institutional metadata
        inst_metadata = InstitutionalAttestationMetadata(
            npi=npi,
            institution_name=institution_name,
            compression_ratio=metadata.get("compression_ratio") if metadata else None,
            k_anonymity=metadata.get("k_anonymity") if metadata else None,
            dimension=metadata.get("dimension") if metadata else None,
            phi_involved=metadata.get("phi_involved", False) if metadata else False,
            patient_count=metadata.get("patient_count") if metadata else None,
            data_classification=metadata.get("data_classification", "SENSITIVE") if metadata else "SENSITIVE",
            additional_data=metadata or {},
        )

        # If multi-sig not required, use standard attestation
        if not require_multi_sig or not self.signatory_registry:
            self.total_institutional_attestations += 1
            return self.record_encoding(
                encoding_id=encoding_id,
                input_data=input_data,
                output_data=output_data,
                metadata=metadata or {},
            )

        # Create multi-sig attestation
        attestation = self.signatory_registry.create_multi_sig_attestation(
            attestation_id=encoding_id,
            data_hash=output_hash,
            required_signatures=required_signatures,
            required_weight=required_weight,
        )

        # Store institutional attestation
        self.institutional_attestations[encoding_id] = attestation
        self.total_institutional_attestations += 1

        # Update metadata with multi-sig info
        inst_metadata.required_signatures = required_signatures
        inst_metadata.collected_signatures = 0
        inst_metadata.total_signatory_weight = 0

        # Create attestation record (pending until multi-sig complete)
        attestation_record = AttestationRecord(
            attestation_id=encoding_id,
            attestation_type=AttestationType.DIFFERENTIAL_ENCODING,
            input_hash=input_hash,
            output_hash=output_hash,
            timestamp=int(time.time()),
            metadata=inst_metadata,
        )

        # Cache locally (not submitted to blockchain until multi-sig complete)
        self.attestation_cache[encoding_id] = attestation_record

        logger.info(
            f"Created multi-sig institutional attestation {encoding_id} "
            f"(requires {required_signatures} sigs, weight {required_weight})"
        )

        return f"multisig_pending:{encoding_id}"

    def add_institutional_signature(
        self,
        attestation_id: str,
        signer_npi: str,
        signature: str,
    ) -> dict[str, Any]:
        """
        Add institutional signature to multi-sig attestation.

        Args:
            attestation_id: Attestation ID
            signer_npi: NPI of signing institution
            signature: Cryptographic signature

        Returns:
            Status dictionary
        """
        if not self.signatory_registry:
            raise ValueError("Multi-sig not enabled (no signatory registry)")

        # Get multi-sig attestation
        multi_sig = self.institutional_attestations.get(attestation_id)
        if not multi_sig:
            raise ValueError(f"Institutional attestation not found: {attestation_id}")

        # Add signature via signatory registry
        success = self.signatory_registry.add_attestation_signature(
            attestation_id=attestation_id,
            npi=signer_npi,
            signature=signature,
        )

        if not success:
            return {
                "success": False,
                "message": "Failed to add signature",
                "is_complete": False,
            }

        # Update attestation metadata
        attestation = self.attestation_cache.get(attestation_id)
        if attestation and isinstance(attestation.metadata, InstitutionalAttestationMetadata):
            attestation.metadata.collected_signatures = len(multi_sig.signatures)
            attestation.metadata.total_signatory_weight = multi_sig.get_current_weight()

        # Check if attestation is now complete
        if multi_sig.is_complete:
            # Submit to blockchain
            if self.blockchain_enabled:
                tx_hash = self._submit_multi_sig_attestation(attestation_id, multi_sig)
                attestation.blockchain_tx = tx_hash
                attestation.blockchain_confirmed = True

                # Move to confirmed
                self.confirmed_attestations[attestation_id] = attestation
                del self.attestation_cache[attestation_id]

                self.total_multi_sig_attestations += 1

                logger.info(f"Multi-sig attestation {attestation_id} completed and submitted: {tx_hash}")

                return {
                    "success": True,
                    "message": "Attestation complete and submitted to blockchain",
                    "is_complete": True,
                    "blockchain_tx": tx_hash,
                    "total_signatures": len(multi_sig.signatures),
                    "total_weight": multi_sig.get_current_weight(),
                }

        # Not yet complete
        return {
            "success": True,
            "message": "Signature added, awaiting additional signatures",
            "is_complete": False,
            "signatures_collected": len(multi_sig.signatures),
            "signatures_required": multi_sig.required_signatures,
            "weight_collected": multi_sig.get_current_weight(),
            "weight_required": multi_sig.required_weight,
        }

    def get_institutional_attestation_status(self, attestation_id: str) -> dict[str, Any]:
        """
        Get status of institutional multi-sig attestation.

        Args:
            attestation_id: Attestation ID

        Returns:
            Status dictionary
        """
        multi_sig = self.institutional_attestations.get(attestation_id)
        if not multi_sig:
            return {"found": False}

        return {
            "found": True,
            "attestation_id": attestation_id,
            "is_complete": multi_sig.is_complete,
            "signatures_collected": len(multi_sig.signatures),
            "signatures_required": multi_sig.required_signatures,
            "weight_collected": multi_sig.get_current_weight(),
            "weight_required": multi_sig.required_weight,
            "signatories": [
                {
                    "npi": npi,
                    "institution": sig.institution_name,
                    "weight": sig.get_signatory_weight(),
                }
                for npi, sig in multi_sig.signatories.items()
            ],
            "blockchain_tx": multi_sig.blockchain_tx,
        }

    def _submit_multi_sig_attestation(
        self,
        attestation_id: str,
        multi_sig: MultiSigAttestation,
    ) -> str:
        """
        Submit completed multi-sig attestation to blockchain.

        Args:
            attestation_id: Attestation ID
            multi_sig: Completed multi-sig attestation

        Returns:
            Transaction hash
        """
        if not self.contract_interface:
            # Simulate blockchain transaction
            tx_data = {
                "attestation_id": attestation_id,
                "data_hash": multi_sig.data_hash,
                "signatures": len(multi_sig.signatures),
                "total_weight": multi_sig.get_current_weight(),
            }
            tx_hash = f"0x{hashlib.sha256(str(tx_data).encode()).hexdigest()}"
            return tx_hash

        # In production, call smart contract for multi-sig attestation
        # For now, simulate
        tx_hash = f"0xmultisig_{hashlib.sha256(attestation_id.encode()).hexdigest()}"
        return tx_hash

    def get_statistics(self) -> dict[str, Any]:
        """Get extended statistics including Phase 2"""
        base_stats = super().get_statistics()

        phase2_stats = {
            "institutional_attestations": self.total_institutional_attestations,
            "multi_sig_attestations_completed": self.total_multi_sig_attestations,
            "multi_sig_attestations_pending": len(
                [a for a in self.institutional_attestations.values() if not a.is_complete]
            ),
            "signatory_registry_enabled": self.signatory_registry is not None,
        }

        # Merge stats
        return {**base_stats, **phase2_stats}


def create_hipaa_attestation_registry(
    blockchain_config: Optional[dict[str, Any]] = None,
    signatory_registry: Optional[TrustedSignatoryRegistry] = None,
) -> HIPAAAttestationRegistry:
    """
    Factory function to create HIPAA attestation registry.

    Args:
        blockchain_config: Blockchain configuration
        signatory_registry: Optional signatory registry for multi-sig

    Returns:
        Configured HIPAA attestation registry
    """
    if not blockchain_config:
        # Default: blockchain disabled
        return HIPAAAttestationRegistry(
            blockchain_enabled=False,
            batch_mode=True,
            batch_size=10,
            signatory_registry=signatory_registry,
        )

    enabled = blockchain_config.get("enabled", False)
    batch_mode = blockchain_config.get("attestation", {}).get("batch_mode", True)
    batch_size = blockchain_config.get("attestation", {}).get("batch_size", 10)

    if not enabled:
        return HIPAAAttestationRegistry(
            blockchain_enabled=False,
            batch_mode=batch_mode,
            batch_size=batch_size,
            signatory_registry=signatory_registry,
        )

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

    return HIPAAAttestationRegistry(
        contract_interface=contract_interface,
        blockchain_enabled=enabled and contract_interface is not None,
        batch_mode=batch_mode,
        batch_size=batch_size,
        signatory_registry=signatory_registry,
    )
