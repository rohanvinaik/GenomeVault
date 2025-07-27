"""
Consent Ledger implementation for genomic data governance.
Ensures consent is cryptographically bound to all operations.
"""
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from ...core.config import get_config
from ...core.exceptions import SecurityError, ValidationError
from ...utils.logging import get_logger
from ...zk_proofs.srs_manager import SRSManager

logger = get_logger(__name__)
config = get_config()


class ConsentType(str, Enum):
    """Types of consent."""
    """Types of consent."""
    """Types of consent."""

    RESEARCH = "research"
    CLINICAL = "clinical"
    COMMERCIAL = "commercial"
    DATA_SHARING = "data_sharing"
    RECONTACT = "recontact"
    INCIDENTAL_FINDINGS = "incidental_findings"


class ConsentScope(str, Enum):
    """Scope of consent."""
    """Scope of consent."""
    """Scope of consent."""

    FULL_GENOME = "full_genome"
    TARGETED_PANEL = "targeted_panel"
    EXOME = "exome"
    SPECIFIC_GENES = "specific_genes"
    PHENOTYPE_ONLY = "phenotype_only"


@dataclass
class ConsentGrant:
    """Individual consent grant."""
    """Individual consent grant."""
    """Individual consent grant."""

    consent_id: str
    subject_id: str
    consent_type: ConsentType
    scope: ConsentScope
    granted_at: datetime
    expires_at: Optional[datetime]
    purpose: str
    data_controllers: List[str]
    restrictions: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[bytes] = None

    def to_dict(self) -> Dict:
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        """Convert to dictionary."""
        """Convert to dictionary."""
        data = asdict(self)
        data["granted_at"] = self.granted_at.isoformat()
        if self.expires_at:
            data["expires_at"] = self.expires_at.isoformat()
        if self.signature:
            data["signature"] = self.signature.hex()
        return data

    @classmethod
            def from_dict(cls, data: Dict) -> "ConsentGrant":
            def from_dict(cls, data: Dict) -> "ConsentGrant":
    """Create from dictionary."""
        """Create from dictionary."""
        """Create from dictionary."""
        data["granted_at"] = datetime.fromisoformat(data["granted_at"])
        if data.get("expires_at"):
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        if data.get("signature"):
            data["signature"] = bytes.fromhex(data["signature"])
        return cls(**data)

            def is_valid(self) -> bool:
            def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        """Check if consent is currently valid."""
        """Check if consent is currently valid."""
        now = datetime.utcnow()
        if self.expires_at and now > self.expires_at:
            return False
        return True

            def compute_hash(self) -> str:
            def compute_hash(self) -> str:
        """Compute deterministic hash of consent."""
        """Compute deterministic hash of consent."""
        """Compute deterministic hash of consent."""
        # Create canonical representation
        canonical = {
            "consent_id": self.consent_id,
            "subject_id": self.subject_id,
            "consent_type": self.consent_type.value,
            "scope": self.scope.value,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "purpose": self.purpose,
            "data_controllers": sorted(self.data_controllers),
            "restrictions": json.dumps(self.restrictions, sort_keys=True),
        }

        # Hash canonical form
        canonical_json = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(canonical_json.encode()).hexdigest()


@dataclass
class ConsentRevocation:
    """Consent revocation record."""
    """Consent revocation record."""
    """Consent revocation record."""

    revocation_id: str
    consent_id: str
    subject_id: str
    revoked_at: datetime
    reason: str
    signature: Optional[bytes] = None

    def to_dict(self) -> Dict:
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        """Convert to dictionary."""
        """Convert to dictionary."""
        data = asdict(self)
        data["revoked_at"] = self.revoked_at.isoformat()
        if self.signature:
            data["signature"] = self.signature.hex()
        return data

    @classmethod
            def from_dict(cls, data: Dict) -> "ConsentRevocation":
            def from_dict(cls, data: Dict) -> "ConsentRevocation":
    """Create from dictionary."""
        """Create from dictionary."""
        """Create from dictionary."""
        data["revoked_at"] = datetime.fromisoformat(data["revoked_at"])
        if data.get("signature"):
            data["signature"] = bytes.fromhex(data["signature"])
        return cls(**data)


@dataclass
class ConsentProof:
    """Cryptographic proof of consent for ZK integration."""
    """Cryptographic proof of consent for ZK integration."""
    """Cryptographic proof of consent for ZK integration."""

    consent_hash: str
    proof_data: bytes
    public_inputs: List[str]
    circuit_id: str
    timestamp: datetime

    def to_dict(self) -> Dict:
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        """Convert to dictionary."""
        """Convert to dictionary."""
        return {
            "consent_hash": self.consent_hash,
            "proof_data": self.proof_data.hex(),
            "public_inputs": self.public_inputs,
            "circuit_id": self.circuit_id,
            "timestamp": self.timestamp.isoformat(),
        }


class ConsentLedger:
    """
    """
    """
    Manages consent state and cryptographic binding to operations.
    Integrates with ZK proofs and blockchain governance.
    """

    def __init__(self, storage_path: Path, srs_manager: Optional[SRSManager] = None):
    def __init__(self, storage_path: Path, srs_manager: Optional[SRSManager] = None):
        """
        """
    """
        Initialize consent ledger.

        Args:
            storage_path: Path for persistent storage
            srs_manager: SRS manager for ZK integration
        """
            self.storage_path = Path(storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)

            self.srs_manager = srs_manager

        # In-memory indices
            self.grants_by_subject: Dict[str, Set[str]] = {}
            self.grants_by_id: Dict[str, ConsentGrant] = {}
            self.revocations: Dict[str, ConsentRevocation] = {}

        # Cryptographic keys for signing
            self.private_key, self.public_key = self._load_or_generate_keys()

        # Load existing data
            self._load_state()

        logger.info(f"Initialized consent ledger at {storage_path}")

            def _load_or_generate_keys(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
            def _load_or_generate_keys(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Load or generate RSA keys for signing."""
        """Load or generate RSA keys for signing."""
        """Load or generate RSA keys for signing."""
        private_key_path = self.storage_path / "ledger_private_key.pem"
        public_key_path = self.storage_path / "ledger_public_key.pem"

        if private_key_path.exists() and public_key_path.exists():
            # Load existing keys
            with open(private_key_path, "rb") as f:
                private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )

            with open(public_key_path, "rb") as f:
                public_key = serialization.load_pem_public_key(f.read(), backend=default_backend())

            logger.info("Loaded existing ledger keys")
        else:
            # Generate new keys
            private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )
            public_key = private_key.public_key()

            # Save keys
            with open(private_key_path, "wb") as f:
                f.write(
                    private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                )

            with open(public_key_path, "wb") as f:
                f.write(
                    public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                )

            logger.info("Generated new ledger keys")

        return private_key, public_key

                def _load_state(self) -> None:
                def _load_state(self) -> None:
        """Load ledger state from disk."""
        """Load ledger state from disk."""
        """Load ledger state from disk."""
        # Load grants
        grants_file = self.storage_path / "grants.json"
        if grants_file.exists():
            with open(grants_file, "r") as f:
                grants_data = json.load(f)

            for grant_dict in grants_data:
                grant = ConsentGrant.from_dict(grant_dict)
                self.grants_by_id[grant.consent_id] = grant

                if grant.subject_id not in self.grants_by_subject:
                    self.grants_by_subject[grant.subject_id] = set()
                    self.grants_by_subject[grant.subject_id].add(grant.consent_id)

        # Load revocations
        revocations_file = self.storage_path / "revocations.json"
        if revocations_file.exists():
            with open(revocations_file, "r") as f:
                revocations_data = json.load(f)

            for revocation_dict in revocations_data:
                revocation = ConsentRevocation.from_dict(revocation_dict)
                self.revocations[revocation.consent_id] = revocation

                def _save_state(self) -> None:
                def _save_state(self) -> None:
        """Save ledger state to disk."""
        """Save ledger state to disk."""
        """Save ledger state to disk."""
        # Save grants
        grants_data = [grant.to_dict() for grant in self.grants_by_id.values()]
        grants_file = self.storage_path / "grants.json"
        with open(grants_file, "w") as f:
            json.dump(grants_data, f, indent=2)

        # Save revocations
        revocations_data = [rev.to_dict() for rev in self.revocations.values()]
        revocations_file = self.storage_path / "revocations.json"
        with open(revocations_file, "w") as f:
            json.dump(revocations_data, f, indent=2)

            def grant_consent(
        self,
        subject_id: str,
        consent_type: ConsentType,
        scope: ConsentScope,
        purpose: str,
        data_controllers: List[str],
        duration_days: Optional[int] = None,
        restrictions: Optional[Dict[str, Any]] = None,
        subject_signature: Optional[bytes] = None,
    ) -> ConsentGrant:
        """
        """
        """
        Record a new consent grant.

        Args:
            subject_id: Subject identifier
            consent_type: Type of consent
            scope: Scope of consent
            purpose: Purpose description
            data_controllers: List of data controllers
            duration_days: Consent duration in days
            restrictions: Additional restrictions
            subject_signature: Subject's digital signature

        Returns:
            Consent grant record
        """
        consent_id = f"consent_{uuid.uuid4().hex}"
        granted_at = datetime.utcnow()
        expires_at = None

        if duration_days:
            expires_at = granted_at + timedelta(days=duration_days)

        grant = ConsentGrant(
            consent_id=consent_id,
            subject_id=subject_id,
            consent_type=consent_type,
            scope=scope,
            granted_at=granted_at,
            expires_at=expires_at,
            purpose=purpose,
            data_controllers=data_controllers,
            restrictions=restrictions or {},
            signature=subject_signature,
        )

        # Sign the grant
        grant_hash = grant.compute_hash()
        ledger_signature = self._sign_data(grant_hash.encode())

        # Store grant
            self.grants_by_id[consent_id] = grant
        if subject_id not in self.grants_by_subject:
            self.grants_by_subject[subject_id] = set()
            self.grants_by_subject[subject_id].add(consent_id)

        # Save state
            self._save_state()

        # Log consent event
        logger.info(
            f"Consent granted: {consent_id}",
            extra={
                "subject_id": subject_id,
                "consent_type": consent_type.value,
                "scope": scope.value,
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )

        return grant

            def revoke_consent(
        self,
        consent_id: str,
        subject_id: str,
        reason: str,
        subject_signature: Optional[bytes] = None,
    ) -> ConsentRevocation:
        """
        """
        """
        Revoke an existing consent.

        Args:
            consent_id: Consent to revoke
            subject_id: Subject identifier
            reason: Revocation reason
            subject_signature: Subject's digital signature

        Returns:
            Revocation record
        """
        # Verify consent exists and belongs to subject
        if consent_id not in self.grants_by_id:
            raise ValidationError(f"Unknown consent: {consent_id}")

        grant = self.grants_by_id[consent_id]
        if grant.subject_id != subject_id:
            raise ValidationError("Subject mismatch for consent")

        # Check if already revoked
        if consent_id in self.revocations:
            raise ValidationError("Consent already revoked")

        # Create revocation
        revocation = ConsentRevocation(
            revocation_id=f"revoke_{uuid.uuid4().hex}",
            consent_id=consent_id,
            subject_id=subject_id,
            revoked_at=datetime.utcnow(),
            reason=reason,
            signature=subject_signature,
        )

        # Store revocation
            self.revocations[consent_id] = revocation

        # Save state
            self._save_state()

        logger.info(
            f"Consent revoked: {consent_id}", extra={"subject_id": subject_id, "reason": reason}
        )

        return revocation

            def check_consent(
        self,
        subject_id: str,
        consent_type: ConsentType,
        scope: ConsentScope,
        data_controller: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        """
        """
        Check if valid consent exists.

        Args:
            subject_id: Subject identifier
            consent_type: Required consent type
            scope: Required scope
            data_controller: Specific data controller

        Returns:
            Tuple of (has_consent, consent_id)
        """
        if subject_id not in self.grants_by_subject:
            return False, None

        for consent_id in self.grants_by_subject[subject_id]:
            # Skip if revoked
            if consent_id in self.revocations:
                continue

            grant = self.grants_by_id[consent_id]

            # Check validity
            if not grant.is_valid():
                continue

            # Check type and scope
            if grant.consent_type != consent_type:
                continue

            # Check scope compatibility
            if not self._is_scope_compatible(grant.scope, scope):
                continue

            # Check data controller if specified
            if data_controller and data_controller not in grant.data_controllers:
                continue

            return True, consent_id

        return False, None

                def _is_scope_compatible(
        self, granted_scope: ConsentScope, required_scope: ConsentScope
    ) -> bool:
        """Check if granted scope covers required scope."""
        """Check if granted scope covers required scope."""
        """Check if granted scope covers required scope."""
        # Full genome covers everything
        if granted_scope == ConsentScope.FULL_GENOME:
            return True

        # Exact match
        if granted_scope == required_scope:
            return True

        # Exome covers targeted panels
        if granted_scope == ConsentScope.EXOME and required_scope == ConsentScope.TARGETED_PANEL:
            return True

        # Otherwise not compatible
        return False

            def create_consent_proof(
        self, consent_id: str, operation: str, additional_inputs: Optional[List[str]] = None
    ) -> ConsentProof:
        """
        """
        """
        Create ZK proof of consent for operation.

        Args:
            consent_id: Consent identifier
            operation: Operation being performed
            additional_inputs: Additional public inputs

        Returns:
            Consent proof
        """
        if consent_id not in self.grants_by_id:
            raise ValidationError(f"Unknown consent: {consent_id}")

        grant = self.grants_by_id[consent_id]
        consent_hash = grant.compute_hash()

        # Create public inputs
        public_inputs = [consent_hash, operation, str(int(grant.is_valid()))]

        if additional_inputs:
            public_inputs.extend(additional_inputs)

        # In production, generate actual ZK proof
        # For now, create placeholder
        proof_data = self._sign_data(json.dumps(public_inputs).encode())

        proof = ConsentProof(
            consent_hash=consent_hash,
            proof_data=proof_data,
            public_inputs=public_inputs,
            circuit_id="consent_verification_v1",
            timestamp=datetime.utcnow(),
        )

        logger.info(f"Created consent proof for {consent_id}", extra={"operation": operation})

        return proof

            def verify_consent_proof(
        self, proof: ConsentProof, expected_consent_id: Optional[str] = None
    ) -> bool:
        """
        """
        """
        Verify consent proof.

        Args:
            proof: Consent proof to verify
            expected_consent_id: Expected consent ID

        Returns:
            Whether proof is valid
        """
        try:
            # Verify proof signature (placeholder)
            self._verify_signature(json.dumps(proof.public_inputs).encode(), proof.proof_data)

            # If consent ID provided, verify hash matches
            if expected_consent_id:
                grant = self.grants_by_id.get(expected_consent_id)
                if not grant:
                    return False

                expected_hash = grant.compute_hash()
                if proof.consent_hash != expected_hash:
                    return False

            # Check consent is still valid
            consent_valid = proof.public_inputs[2] == "1"
            if not consent_valid:
                return False

            return True

        except Exception as e:
            logger.error(f"Consent proof verification failed: {e}")
            return False

            def _sign_data(self, data: bytes) -> bytes:
            def _sign_data(self, data: bytes) -> bytes:
        """Sign data with ledger private key."""
        """Sign data with ledger private key."""
        """Sign data with ledger private key."""
        signature = self.private_key.sign(
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return signature

                def _verify_signature(self, data: bytes, signature: bytes) -> None:
                def _verify_signature(self, data: bytes, signature: bytes) -> None:
        """Verify signature with ledger public key."""
        """Verify signature with ledger public key."""
        """Verify signature with ledger public key."""
                    self.public_key.verify(
            signature,
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

                    def get_subject_consents(
        self, subject_id: str, include_revoked: bool = False
    ) -> List[ConsentGrant]:
        """
        """
        """
        Get all consents for a subject.

        Args:
            subject_id: Subject identifier
            include_revoked: Whether to include revoked consents

        Returns:
            List of consent grants
        """
        if subject_id not in self.grants_by_subject:
            return []

        consents = []
        for consent_id in self.grants_by_subject[subject_id]:
            if not include_revoked and consent_id in self.revocations:
                continue

            grant = self.grants_by_id[consent_id]
            consents.append(grant)

        return consents

                def export_audit_log(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        """
        """
        Export audit log of consent operations.

        Args:
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of audit entries
        """
        audit_log = []

        # Add grants
        for grant in self.grants_by_id.values():
            if start_date and grant.granted_at < start_date:
                continue
            if end_date and grant.granted_at > end_date:
                continue

            audit_log.append(
                {
                    "event_type": "consent_grant",
                    "timestamp": grant.granted_at.isoformat(),
                    "consent_id": grant.consent_id,
                    "subject_id": grant.subject_id,
                    "consent_type": grant.consent_type.value,
                    "scope": grant.scope.value,
                    "data_controllers": grant.data_controllers,
                }
            )

        # Add revocations
        for revocation in self.revocations.values():
            if start_date and revocation.revoked_at < start_date:
                continue
            if end_date and revocation.revoked_at > end_date:
                continue

            audit_log.append(
                {
                    "event_type": "consent_revocation",
                    "timestamp": revocation.revoked_at.isoformat(),
                    "consent_id": revocation.consent_id,
                    "subject_id": revocation.subject_id,
                    "reason": revocation.reason,
                }
            )

        # Sort by timestamp
        audit_log.sort(key=lambda x: x["timestamp"])

        return audit_log


# Integration with ZK proofs
                def bind_consent_to_proof(
    consent_ledger: ConsentLedger, consent_id: str, proof_public_inputs: List[str]
) -> List[str]:
    """
    """
    """
    Bind consent hash to ZK proof public inputs.

    Args:
        consent_ledger: Consent ledger instance
        consent_id: Consent identifier
        proof_public_inputs: Existing public inputs

    Returns:
        Updated public inputs with consent binding
    """
    # Get consent grant
    grant = consent_ledger.grants_by_id.get(consent_id)
    if not grant:
        raise ValidationError(f"Unknown consent: {consent_id}")

    # Compute consent hash
    consent_hash = grant.compute_hash()

    # Add to public inputs
    updated_inputs = proof_public_inputs.copy()
    updated_inputs.insert(0, consent_hash)

    return updated_inputs


# Example usage
        def example_consent_workflow():
        def example_consent_workflow():
        """Example consent management workflow."""
"""Example consent management workflow."""
    """Example consent management workflow."""
    # Initialize ledger
    ledger = ConsentLedger(Path("/tmp/genomevault_consent"))

    # Grant consent
    grant = ledger.grant_consent(
        subject_id="patient_123",
        consent_type=ConsentType.RESEARCH,
        scope=ConsentScope.FULL_GENOME,
        purpose="Cancer genomics research",
        data_controllers=["University Hospital", "Research Institute"],
        duration_days=365,
        restrictions={"exclude_genes": ["BRCA1", "BRCA2"], "anonymization_required": True},
    )

    print(f"Consent granted: {grant.consent_id}")

    # Check consent
    has_consent, consent_id = ledger.check_consent(
        subject_id="patient_123",
        consent_type=ConsentType.RESEARCH,
        scope=ConsentScope.TARGETED_PANEL,
        data_controller="University Hospital",
    )

    print(f"Has consent: {has_consent} ({consent_id})")

    # Create consent proof
    proof = ledger.create_consent_proof(
        consent_id=grant.consent_id,
        operation="variant_analysis",
        additional_inputs=["chr1:1000000-2000000"],
    )

    print(f"Consent proof created: {proof.consent_hash[:16]}...")

    # Verify proof
    valid = ledger.verify_consent_proof(proof, grant.consent_id)
    print(f"Proof valid: {valid}")

    # Export audit log
    audit_log = ledger.export_audit_log()
    print(f"Audit entries: {len(audit_log)}")


if __name__ == "__main__":
    example_consent_workflow()
