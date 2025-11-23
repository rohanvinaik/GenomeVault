"""
Trusted Signatory Registry for GenomeVault Phase 2

On-chain registry of verified HIPAA institutions and their signing authority.
Implements multi-signature requirements for sensitive genomic operations.

Features:
- On-chain verification records
- Multi-signature attestation support
- Institutional credential management
- Revocation and expiration tracking
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Optional

from genomevault.utils.logging import get_logger

from .models import HIPAACredentials, VerificationRecord, VerificationStatus
from .npi_verification import HIPAACredentialVerifier, NPIVerificationResult

logger = get_logger(__name__)


class SignatoryTier(Enum):
    """Tier of trusted signatory (determines voting weight)"""

    BASIC = auto()  # Newly verified, weight=1
    VERIFIED = auto()  # Multiple verifications, weight=5
    TRUSTED = auto()  # Long-term good standing, weight=10
    FOUNDER = auto()  # Founding institutions, weight=20


@dataclass
class TrustedSignatory:
    """Trusted signatory record"""

    npi: str
    institution_name: str
    tier: SignatoryTier
    verification_status: VerificationStatus

    # Verification details
    credentials_hash: str  # SHA-256 of HIPAA credentials
    blockchain_tx: Optional[str] = None  # On-chain registration transaction
    verified_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Signatory power
    weight: int = 1  # Voting/attestation weight
    honesty_probability: float = 0.98  # Bayesian honesty estimate

    # Multi-sig participation
    total_attestations: int = 0
    verified_attestations: int = 0
    failed_attestations: int = 0

    # Revocation
    is_active: bool = True
    revoked_at: Optional[datetime] = None
    revocation_reason: Optional[str] = None

    def get_signatory_weight(self) -> int:
        """Get signatory weight based on tier"""
        tier_weights = {
            SignatoryTier.BASIC: 1,
            SignatoryTier.VERIFIED: 5,
            SignatoryTier.TRUSTED: 10,
            SignatoryTier.FOUNDER: 20,
        }
        return tier_weights.get(self.tier, 1)

    def is_valid(self) -> bool:
        """Check if signatory is currently valid"""
        if not self.is_active:
            return False

        if self.verification_status != VerificationStatus.VERIFIED:
            return False

        if self.expires_at and datetime.now() > self.expires_at:
            return False

        if self.revoked_at:
            return False

        return True

    def to_chain_data(self) -> dict[str, Any]:
        """Convert to data for blockchain storage"""
        return {
            "npi": self.npi,
            "institution_name": self.institution_name,
            "tier": self.tier.name,
            "credentials_hash": self.credentials_hash,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "weight": self.get_signatory_weight(),
            "honesty_probability": self.honesty_probability,
        }


@dataclass
class MultiSigAttestation:
    """Multi-signature attestation record"""

    attestation_id: str
    data_hash: str  # Hash of data being attested
    required_signatures: int  # Minimum signatures required
    required_weight: int  # Minimum total weight required

    # Signatures collected
    signatures: dict[str, str] = None  # NPI -> signature
    signatories: dict[str, TrustedSignatory] = None  # NPI -> signatory record

    # Status
    is_complete: bool = False
    completed_at: Optional[datetime] = None
    blockchain_tx: Optional[str] = None

    def __post_init__(self):
        if self.signatures is None:
            self.signatures = {}
        if self.signatories is None:
            self.signatories = {}

    def add_signature(self, npi: str, signature: str, signatory: TrustedSignatory) -> bool:
        """
        Add signature to attestation.

        Args:
            npi: National Provider Identifier
            signature: Cryptographic signature
            signatory: Signatory record

        Returns:
            True if signature added successfully
        """
        # Verify signatory is valid
        if not signatory.is_valid():
            logger.warning(f"Cannot add signature from invalid signatory: {npi}")
            return False

        # Add signature
        self.signatures[npi] = signature
        self.signatories[npi] = signatory

        # Check if attestation is now complete
        if self.is_attestation_complete():
            self.is_complete = True
            self.completed_at = datetime.now()
            logger.info(f"Multi-sig attestation {self.attestation_id} completed")

        return True

    def is_attestation_complete(self) -> bool:
        """Check if attestation has sufficient signatures"""
        # Check signature count
        if len(self.signatures) < self.required_signatures:
            return False

        # Check total weight
        total_weight = sum(s.get_signatory_weight() for s in self.signatories.values())
        if total_weight < self.required_weight:
            return False

        return True

    def get_current_weight(self) -> int:
        """Get current total weight of signatures"""
        return sum(s.get_signatory_weight() for s in self.signatories.values())


class TrustedSignatoryRegistry:
    """
    Registry of trusted HIPAA signatories.

    Manages institutional verification, multi-signature attestations,
    and on-chain signatory records.
    """

    def __init__(
        self,
        verifier: HIPAACredentialVerifier,
        contract_interface: Optional[Any] = None,
        blockchain_enabled: bool = False,
    ):
        """
        Initialize trusted signatory registry.

        Args:
            verifier: HIPAA credential verifier
            contract_interface: Blockchain contract interface (optional)
            blockchain_enabled: Whether to record on blockchain
        """
        self.verifier = verifier
        self.contract_interface = contract_interface
        self.blockchain_enabled = blockchain_enabled

        # Local registry
        self.signatories: dict[str, TrustedSignatory] = {}
        self.pending_verifications: dict[str, HIPAACredentials] = {}
        self.multi_sig_attestations: dict[str, MultiSigAttestation] = {}

        # Statistics
        self.total_signatories = 0
        self.total_attestations = 0
        self.total_blockchain_transactions = 0

        logger.info(
            f"Trusted signatory registry initialized "
            f"(blockchain={'enabled' if blockchain_enabled else 'disabled'})"
        )

    def register_signatory(
        self,
        credentials: HIPAACredentials,
        tier: SignatoryTier = SignatoryTier.BASIC,
        validity_days: int = 365,
    ) -> TrustedSignatory:
        """
        Register a new trusted signatory.

        Args:
            credentials: HIPAA credentials
            tier: Initial signatory tier
            validity_days: Validity period in days

        Returns:
            Trusted signatory record
        """
        logger.info(f"Registering signatory for NPI {credentials.npi}")

        # Verify credentials
        verification = self.verifier.verify_credentials(credentials)

        if not verification.is_valid:
            raise ValueError(f"Credential verification failed: {verification.error_message}")

        # Compute credentials hash
        credentials_str = f"{credentials.npi}:{credentials.baa_hash}:{credentials.risk_analysis_hash}:{credentials.hsm_serial}"
        credentials_hash = hashlib.sha256(credentials_str.encode()).hexdigest()

        # Create signatory record
        signatory = TrustedSignatory(
            npi=credentials.npi,
            institution_name=verification.npi_record.name,
            tier=tier,
            verification_status=VerificationStatus.VERIFIED,
            credentials_hash=credentials_hash,
            verified_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=validity_days),
            weight=tier.value,
            honesty_probability=0.98,
        )

        # Record on blockchain (if enabled)
        if self.blockchain_enabled and self.contract_interface:
            try:
                tx_hash = self._record_on_chain(signatory)
                signatory.blockchain_tx = tx_hash
                self.total_blockchain_transactions += 1
                logger.info(f"Signatory recorded on-chain: {tx_hash}")
            except Exception as e:
                logger.warning(f"Failed to record signatory on-chain: {e}")

        # Add to registry
        self.signatories[credentials.npi] = signatory
        self.total_signatories += 1

        logger.info(
            f"Registered signatory: {signatory.institution_name} "
            f"(NPI: {credentials.npi}, tier: {tier.name}, weight: {signatory.get_signatory_weight()})"
        )

        return signatory

    def create_multi_sig_attestation(
        self,
        attestation_id: str,
        data_hash: str,
        required_signatures: int = 3,
        required_weight: int = 10,
    ) -> MultiSigAttestation:
        """
        Create multi-signature attestation.

        Args:
            attestation_id: Unique attestation identifier
            data_hash: Hash of data to attest
            required_signatures: Minimum number of signatures
            required_weight: Minimum total signatory weight

        Returns:
            Multi-sig attestation record
        """
        attestation = MultiSigAttestation(
            attestation_id=attestation_id,
            data_hash=data_hash,
            required_signatures=required_signatures,
            required_weight=required_weight,
        )

        self.multi_sig_attestations[attestation_id] = attestation
        self.total_attestations += 1

        logger.info(
            f"Created multi-sig attestation {attestation_id} "
            f"(required: {required_signatures} sigs, {required_weight} weight)"
        )

        return attestation

    def add_attestation_signature(
        self,
        attestation_id: str,
        npi: str,
        signature: str,
    ) -> bool:
        """
        Add signature to multi-sig attestation.

        Args:
            attestation_id: Attestation ID
            npi: National Provider Identifier of signatory
            signature: Cryptographic signature

        Returns:
            True if signature added successfully
        """
        # Get attestation
        attestation = self.multi_sig_attestations.get(attestation_id)
        if not attestation:
            logger.error(f"Attestation not found: {attestation_id}")
            return False

        # Get signatory
        signatory = self.signatories.get(npi)
        if not signatory:
            logger.error(f"Signatory not found: {npi}")
            return False

        # Add signature
        success = attestation.add_signature(npi, signature, signatory)

        if success:
            # Update signatory statistics
            signatory.total_attestations += 1

            # If attestation is complete, record on-chain
            if attestation.is_complete and self.blockchain_enabled:
                try:
                    tx_hash = self._record_attestation_on_chain(attestation)
                    attestation.blockchain_tx = tx_hash
                    self.total_blockchain_transactions += 1

                    # Update all signatories
                    for sig_npi in attestation.signatories.keys():
                        self.signatories[sig_npi].verified_attestations += 1

                    logger.info(f"Multi-sig attestation recorded on-chain: {tx_hash}")
                except Exception as e:
                    logger.warning(f"Failed to record attestation on-chain: {e}")

        return success

    def get_signatory(self, npi: str) -> Optional[TrustedSignatory]:
        """Get signatory by NPI"""
        return self.signatories.get(npi)

    def get_active_signatories(self) -> list[TrustedSignatory]:
        """Get all active signatories"""
        return [s for s in self.signatories.values() if s.is_valid()]

    def revoke_signatory(self, npi: str, reason: str) -> bool:
        """
        Revoke signatory status.

        Args:
            npi: National Provider Identifier
            reason: Revocation reason

        Returns:
            True if revoked successfully
        """
        signatory = self.signatories.get(npi)
        if not signatory:
            return False

        signatory.is_active = False
        signatory.revoked_at = datetime.now()
        signatory.revocation_reason = reason
        signatory.verification_status = VerificationStatus.REVOKED

        logger.info(f"Revoked signatory {npi}: {reason}")
        return True

    def upgrade_signatory_tier(self, npi: str, new_tier: SignatoryTier) -> bool:
        """
        Upgrade signatory to higher tier.

        Args:
            npi: National Provider Identifier
            new_tier: New signatory tier

        Returns:
            True if upgraded successfully
        """
        signatory = self.signatories.get(npi)
        if not signatory:
            return False

        old_tier = signatory.tier
        signatory.tier = new_tier
        signatory.weight = new_tier.value

        logger.info(f"Upgraded signatory {npi}: {old_tier.name} -> {new_tier.name}")
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics"""
        active_signatories = self.get_active_signatories()

        return {
            "total_signatories": self.total_signatories,
            "active_signatories": len(active_signatories),
            "total_attestations": self.total_attestations,
            "completed_attestations": sum(
                1 for a in self.multi_sig_attestations.values() if a.is_complete
            ),
            "total_blockchain_transactions": self.total_blockchain_transactions,
            "tier_distribution": {
                tier.name: sum(1 for s in active_signatories if s.tier == tier)
                for tier in SignatoryTier
            },
        }

    def _record_on_chain(self, signatory: TrustedSignatory) -> str:
        """Record signatory on blockchain"""
        if not self.contract_interface:
            return f"local:signatory_{signatory.npi}"

        # In production, this would call smart contract
        # For now, simulate transaction
        chain_data = signatory.to_chain_data()
        tx_hash = f"0x{hashlib.sha256(str(chain_data).encode()).hexdigest()}"

        return tx_hash

    def _record_attestation_on_chain(self, attestation: MultiSigAttestation) -> str:
        """Record multi-sig attestation on blockchain"""
        if not self.contract_interface:
            return f"local:attestation_{attestation.attestation_id}"

        # In production, this would call smart contract
        # For now, simulate transaction
        attestation_data = {
            "attestation_id": attestation.attestation_id,
            "data_hash": attestation.data_hash,
            "signatures": len(attestation.signatures),
            "total_weight": attestation.get_current_weight(),
        }
        tx_hash = f"0x{hashlib.sha256(str(attestation_data).encode()).hexdigest()}"

        return tx_hash


def create_signatory_registry(
    verifier: HIPAACredentialVerifier,
    blockchain_enabled: bool = False,
    contract_interface: Optional[Any] = None,
) -> TrustedSignatoryRegistry:
    """
    Create configured signatory registry.

    Args:
        verifier: HIPAA credential verifier
        blockchain_enabled: Whether to use blockchain
        contract_interface: Optional blockchain interface

    Returns:
        Configured TrustedSignatoryRegistry
    """
    registry = TrustedSignatoryRegistry(
        verifier=verifier,
        contract_interface=contract_interface,
        blockchain_enabled=blockchain_enabled,
    )

    return registry
