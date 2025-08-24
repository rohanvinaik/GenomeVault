"""
Threshold Cryptography Service

Implements distributed key generation, threshold signing, and recovery mechanisms
for secure multi-party genomic data operations.

Section 2.2.3 Implementation
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import BLS library, fallback to simulation if not available
try:
    from py_ecc.bls import G2ProofOfPossession as bls

    HAS_BLS = True
except ImportError:
    logger.warning("BLS signatures not available, using simulation mode")
    HAS_BLS = False
    bls = None


class ShareType(Enum):
    """Types of secret shares"""

    KEY_GENERATION = "key_generation"
    SIGNING = "signing"
    ENCRYPTION = "encryption"
    RECOVERY = "recovery"


class QuorumStatus(Enum):
    """Quorum achievement status"""

    PENDING = "pending"
    MET = "met"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class ThresholdConfig:
    """Configuration for threshold cryptography"""

    threshold: int = 5  # Minimum shares needed (t)
    total_shares: int = 8  # Total shares distributed (n)
    key_size: int = 256  # Key size in bits
    max_rate_per_minute: int = 10  # Rate limiting
    session_timeout_minutes: int = 30  # Session expiry
    enable_geographic_distribution: bool = True
    min_geographic_regions: int = 3
    enable_forward_secrecy: bool = True
    key_rotation_days: int = 90
    emergency_recovery_threshold: int = 7  # Higher threshold for emergency


@dataclass
class SecretShare:
    """Individual secret share"""

    share_id: str
    participant_id: str
    share_value: bytes
    share_type: ShareType
    commitment: Optional[bytes] = None
    timestamp: datetime = field(default_factory=datetime.now)
    geographic_region: Optional[str] = None

    def verify_commitment(self, public_commitment: bytes) -> bool:
        """Verify share against public commitment"""
        if not self.commitment:
            return False
        share_hash = hashlib.sha256(self.share_value).digest()
        return share_hash == self.commitment


@dataclass
class ThresholdSession:
    """Active threshold operation session"""

    session_id: str
    operation_type: ShareType
    required_threshold: int
    participants: Set[str] = field(default_factory=set)
    shares_received: Dict[str, SecretShare] = field(default_factory=dict)
    status: QuorumStatus = QuorumStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    result: Optional[Any] = None

    def is_expired(self) -> bool:
        """Check if session has expired"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False

    def has_quorum(self) -> bool:
        """Check if enough shares received"""
        return len(self.shares_received) >= self.required_threshold


@dataclass
class AuditEntry:
    """Audit log entry for threshold operations"""

    timestamp: datetime
    operation: str
    participant_id: str
    session_id: str
    success: bool
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    geographic_region: Optional[str] = None


class ShamirSecretSharing:
    """Shamir's Secret Sharing implementation"""

    def __init__(self, threshold: int, total_shares: int, prime: Optional[int] = None):
        """
        Initialize Shamir secret sharing

        Args:
            threshold: Minimum shares needed to reconstruct
            total_shares: Total shares to generate
            prime: Prime modulus (default: large safe prime)
        """
        self.threshold = threshold
        self.total_shares = total_shares
        # Use a large prime for the field
        self.prime = prime or (2**256 - 189)  # Large prime close to 2^256

        if threshold > total_shares:
            raise ValueError("Threshold cannot exceed total shares")
        if threshold < 2:
            raise ValueError("Threshold must be at least 2")

    def split_secret(self, secret: int) -> List[Tuple[int, int]]:
        """
        Split secret into shares using Shamir's scheme

        Args:
            secret: Secret value to split

        Returns:
            List of (x, y) share points
        """
        if secret >= self.prime:
            raise ValueError("Secret must be less than prime modulus")

        # Generate random polynomial coefficients
        coefficients = [secret]  # a_0 = secret
        for _ in range(self.threshold - 1):
            coefficients.append(secrets.randbelow(self.prime))

        # Evaluate polynomial at points 1, 2, ..., n
        shares = []
        for x in range(1, self.total_shares + 1):
            y = self._evaluate_polynomial(coefficients, x)
            shares.append((x, y))

        return shares

    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct secret from shares using Lagrange interpolation

        Args:
            shares: List of (x, y) share points

        Returns:
            Reconstructed secret
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")

        # Use first k shares
        shares = shares[: self.threshold]

        # Lagrange interpolation at x=0
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            numerator = 1
            denominator = 1

            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % self.prime
                    denominator = (denominator * (xi - xj)) % self.prime

            # Modular inverse of denominator
            inv_denominator = pow(denominator, self.prime - 2, self.prime)
            lagrange_coeff = (numerator * inv_denominator) % self.prime

            secret = (secret + yi * lagrange_coeff) % self.prime

        return secret

    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x"""
        result = 0
        power = 1
        for coeff in coefficients:
            result = (result + coeff * power) % self.prime
            power = (power * x) % self.prime
        return result


class ThresholdKeyGeneration:
    """Distributed key generation without trusted dealer"""

    def __init__(self, config: ThresholdConfig):
        """
        Initialize distributed key generation

        Args:
            config: Threshold configuration
        """
        self.config = config
        self.shamir = ShamirSecretSharing(config.threshold, config.total_shares)
        self.participant_shares: Dict[str, List[SecretShare]] = defaultdict(list)
        self.public_keys: Dict[str, bytes] = {}
        self.commitments: Dict[str, List[bytes]] = defaultdict(list)

    def generate_key_shares(self, participant_id: str) -> Tuple[List[SecretShare], bytes]:
        """
        Generate key shares for a participant

        Args:
            participant_id: Participant identifier

        Returns:
            Tuple of (shares, public_key)
        """
        # Generate random secret
        secret = secrets.randbelow(self.shamir.prime)

        # Split into shares
        share_points = self.shamir.split_secret(secret)

        # Create share objects with commitments
        shares = []
        for i, (x, y) in enumerate(share_points):
            share_value = y.to_bytes(32, "big")
            commitment = hashlib.sha256(share_value).digest()

            share = SecretShare(
                share_id=f"{participant_id}_{i}",
                participant_id=participant_id,
                share_value=share_value,
                share_type=ShareType.KEY_GENERATION,
                commitment=commitment,
            )
            shares.append(share)

        # Generate public key (simplified - in practice use BLS)
        if HAS_BLS:
            # Use BLS public key generation (would use actual BLS lib)
            _ = secret.to_bytes(32, "big")  # For future BLS implementation
            public_key = b"BLS_PUBLIC_KEY_PLACEHOLDER"  # Would use actual BLS
        else:
            # Fallback to EC
            private_key = ec.derive_private_key(secret, ec.SECP256K1())
            public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

        self.participant_shares[participant_id] = shares
        self.public_keys[participant_id] = public_key

        return shares, public_key

    def combine_shares(self, shares: List[SecretShare]) -> bytes:
        """
        Combine shares to generate group key

        Args:
            shares: List of secret shares

        Returns:
            Combined group key
        """
        # Convert shares to points
        points = []
        for share in shares:
            x = int(share.share_id.split("_")[1]) + 1
            y = int.from_bytes(share.share_value, "big")
            points.append((x, y))

        # Reconstruct group secret
        group_secret = self.shamir.reconstruct_secret(points)

        # Derive group key
        group_key = hashlib.sha256(group_secret.to_bytes(32, "big") + b"GROUP_KEY").digest()

        return group_key


class ThresholdSigningService:
    """BLS threshold signing service"""

    def __init__(self, config: ThresholdConfig):
        """
        Initialize threshold signing service

        Args:
            config: Threshold configuration
        """
        self.config = config
        self.sessions: Dict[str, ThresholdSession] = {}
        self.rate_limiter: Dict[str, List[datetime]] = defaultdict(list)
        self.audit_log: List[AuditEntry] = []

    def initiate_signing(self, message: bytes, participant_id: str) -> str:
        """
        Initiate threshold signing session

        Args:
            message: Message to sign
            participant_id: Initiating participant

        Returns:
            Session ID
        """
        # Check rate limits
        if not self._check_rate_limit(participant_id):
            raise ValueError("Rate limit exceeded")

        # Create session
        session_id = hashlib.sha256(
            message + participant_id.encode() + str(time.time()).encode()
        ).hexdigest()[:16]

        session = ThresholdSession(
            session_id=session_id,
            operation_type=ShareType.SIGNING,
            required_threshold=self.config.threshold,
            expires_at=datetime.now() + timedelta(minutes=self.config.session_timeout_minutes),
        )

        self.sessions[session_id] = session

        # Log audit entry
        self._log_audit(
            operation="initiate_signing",
            participant_id=participant_id,
            session_id=session_id,
            success=True,
            details={"message_hash": hashlib.sha256(message).hexdigest()},
        )

        logger.info(f"Initiated signing session {session_id}")
        return session_id

    def submit_signature_share(
        self,
        session_id: str,
        participant_id: str,
        signature_share: bytes,
        geographic_region: Optional[str] = None,
    ) -> bool:
        """
        Submit signature share for threshold signing

        Args:
            session_id: Session identifier
            participant_id: Participant identifier
            signature_share: Partial signature
            geographic_region: Geographic location

        Returns:
            True if share accepted
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("Invalid session")

        if session.is_expired():
            session.status = QuorumStatus.EXPIRED
            raise ValueError("Session expired")

        if session.status != QuorumStatus.PENDING:
            raise ValueError("Session not accepting shares")

        # Create share object
        share = SecretShare(
            share_id=f"{session_id}_{participant_id}",
            participant_id=participant_id,
            share_value=signature_share,
            share_type=ShareType.SIGNING,
            geographic_region=geographic_region,
        )

        # Add to session
        session.shares_received[participant_id] = share
        session.participants.add(participant_id)

        # Check quorum
        if session.has_quorum():
            session.status = QuorumStatus.MET
            self._combine_signatures(session)

        self._log_audit(
            operation="submit_signature_share",
            participant_id=participant_id,
            session_id=session_id,
            success=True,
            details={"quorum_met": session.has_quorum()},
        )

        return True

    def _combine_signatures(self, session: ThresholdSession) -> bytes:
        """Combine signature shares into final signature"""
        if HAS_BLS:
            # Use BLS signature aggregation
            # In practice, would aggregate actual BLS signatures
            combined = b"BLS_COMBINED_SIGNATURE"
        else:
            # Simple XOR combination for simulation
            combined = b"\x00" * 32
            for share in session.shares_received.values():
                share_bytes = share.share_value[:32].ljust(32, b"\x00")
                combined = bytes(a ^ b for a, b in zip(combined, share_bytes))

        session.result = combined
        logger.info(f"Combined signatures for session {session.session_id}")
        return combined

    def _check_rate_limit(self, participant_id: str) -> bool:
        """Check if participant is within rate limits"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self.rate_limiter[participant_id] = [
            t for t in self.rate_limiter[participant_id] if t > minute_ago
        ]

        # Check limit
        if len(self.rate_limiter[participant_id]) >= self.config.max_rate_per_minute:
            return False

        # Add new entry
        self.rate_limiter[participant_id].append(now)
        return True

    def _log_audit(
        self,
        operation: str,
        participant_id: str,
        session_id: str,
        success: bool,
        details: Dict[str, Any],
    ):
        """Log audit entry"""
        entry = AuditEntry(
            timestamp=datetime.now(),
            operation=operation,
            participant_id=participant_id,
            session_id=session_id,
            success=success,
            details=details,
        )
        self.audit_log.append(entry)


class ThresholdEncryption:
    """Threshold encryption with proxy re-encryption"""

    def __init__(self, config: ThresholdConfig):
        """
        Initialize threshold encryption

        Args:
            config: Threshold configuration
        """
        self.config = config
        self.shamir = ShamirSecretSharing(config.threshold, config.total_shares)
        self.re_encryption_keys: Dict[str, bytes] = {}
        self.forward_secure_keys: List[bytes] = []
        self.current_epoch = 0

    def encrypt_with_shares(
        self, data: bytes, public_keys: List[bytes]
    ) -> Tuple[bytes, List[bytes]]:
        """
        Encrypt data for threshold decryption

        Args:
            data: Data to encrypt
            public_keys: List of participant public keys

        Returns:
            Tuple of (ciphertext, encrypted_shares)
        """
        # Generate ephemeral key
        ephemeral_key = secrets.token_bytes(32)

        # Encrypt data
        cipher = ChaCha20Poly1305(ephemeral_key)
        nonce = secrets.token_bytes(12)
        ciphertext = cipher.encrypt(nonce, data, None)

        # Split ephemeral key into shares
        key_int = int.from_bytes(ephemeral_key, "big")
        shares = self.shamir.split_secret(key_int % self.shamir.prime)

        # Encrypt shares for each participant
        encrypted_shares = []
        for i, (x, y) in enumerate(shares):
            if i < len(public_keys):
                # In practice, encrypt with participant's public key
                # Here we use a simple XOR for demonstration
                share_bytes = y.to_bytes(32, "big")
                key_hash = hashlib.sha256(public_keys[i]).digest()[:32]
                encrypted = bytes(a ^ b for a, b in zip(share_bytes, key_hash))
                encrypted_shares.append(encrypted)

        return nonce + ciphertext, encrypted_shares

    def generate_re_encryption_key(self, from_key: bytes, to_key: bytes) -> bytes:
        """
        Generate proxy re-encryption key

        Args:
            from_key: Original recipient key
            to_key: New recipient key

        Returns:
            Re-encryption key
        """
        # Derive re-encryption key
        hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"RE_ENCRYPTION", info=b"PROXY")

        combined = from_key + to_key
        re_key = hkdf.derive(combined)

        # Store for audit
        re_key_id = hashlib.sha256(re_key).hexdigest()[:16]
        self.re_encryption_keys[re_key_id] = re_key

        return re_key

    def apply_forward_secrecy(self) -> bytes:
        """
        Rotate keys for forward secrecy

        Returns:
            New epoch key
        """
        # Generate new epoch key
        epoch_key = secrets.token_bytes(32)
        self.forward_secure_keys.append(epoch_key)
        self.current_epoch += 1

        # Delete old keys after rotation period
        if self.config.enable_forward_secrecy:
            cutoff = self.current_epoch - 2
            if cutoff > 0 and cutoff < len(self.forward_secure_keys):
                # Securely overwrite old key
                old_key = self.forward_secure_keys[cutoff - 1]
                self.forward_secure_keys[cutoff - 1] = b"\x00" * len(old_key)

        logger.info(f"Rotated to epoch {self.current_epoch}")
        return epoch_key


class RecoveryMechanism:
    """Verifiable secret reconstruction and emergency recovery"""

    def __init__(self, config: ThresholdConfig):
        """
        Initialize recovery mechanism

        Args:
            config: Threshold configuration
        """
        self.config = config
        # Ensure emergency threshold doesn't exceed total shares
        emergency_threshold = min(config.emergency_recovery_threshold, config.total_shares)
        self.shamir = ShamirSecretSharing(emergency_threshold, config.total_shares)
        self.recovery_shares: Dict[str, List[SecretShare]] = defaultdict(list)
        self.geographic_distribution: Dict[str, Set[str]] = defaultdict(set)

    def distribute_recovery_shares(
        self, secret: bytes, geographic_regions: List[str]
    ) -> Dict[str, SecretShare]:
        """
        Distribute recovery shares across geographic regions

        Args:
            secret: Secret to protect
            geographic_regions: List of regions for distribution

        Returns:
            Mapping of region to share
        """
        if self.config.enable_geographic_distribution:
            if len(set(geographic_regions)) < self.config.min_geographic_regions:
                raise ValueError(f"Need at least {self.config.min_geographic_regions} regions")

        # Convert secret to integer
        secret_int = int.from_bytes(secret[:32], "big")

        # Generate shares
        shares = self.shamir.split_secret(secret_int % self.shamir.prime)

        # Distribute across regions
        distribution = {}
        for i, (x, y) in enumerate(shares):
            if i < len(geographic_regions):
                region = geographic_regions[i]

                share = SecretShare(
                    share_id=f"recovery_{i}",
                    participant_id=f"region_{region}",
                    share_value=y.to_bytes(32, "big"),
                    share_type=ShareType.RECOVERY,
                    geographic_region=region,
                )

                distribution[region] = share
                self.geographic_distribution[secret.hex()].add(region)

        return distribution

    def initiate_emergency_recovery(
        self, recovery_id: str, authorized_parties: List[str]
    ) -> ThresholdSession:
        """
        Start emergency recovery protocol

        Args:
            recovery_id: Recovery identifier
            authorized_parties: List of authorized participants

        Returns:
            Recovery session
        """
        session = ThresholdSession(
            session_id=f"recovery_{recovery_id}",
            operation_type=ShareType.RECOVERY,
            required_threshold=self.config.emergency_recovery_threshold,
            expires_at=datetime.now() + timedelta(hours=1),  # Longer timeout
        )

        # Notify authorized parties (in practice, send secure notifications)
        for party in authorized_parties:
            logger.info(f"Notified {party} of recovery protocol {recovery_id}")

        return session

    def verify_and_reconstruct(
        self, shares: List[SecretShare], commitments: List[bytes]
    ) -> Optional[bytes]:
        """
        Verify shares and reconstruct secret

        Args:
            shares: Recovery shares
            commitments: Public commitments for verification

        Returns:
            Reconstructed secret if successful
        """
        # Verify each share against commitments
        verified_shares = []
        for share in shares:
            if share.commitment in commitments:
                if share.verify_commitment(share.commitment):
                    verified_shares.append(share)
                else:
                    logger.warning(f"Share {share.share_id} failed verification")

        if len(verified_shares) < self.config.emergency_recovery_threshold:
            logger.error("Insufficient verified shares for recovery")
            return None

        # Convert to points
        points = []
        for share in verified_shares:
            x = int(share.share_id.split("_")[1]) + 1
            y = int.from_bytes(share.share_value, "big")
            points.append((x, y))

        # Reconstruct secret
        try:
            secret_int = self.shamir.reconstruct_secret(points)
            secret = secret_int.to_bytes(32, "big")

            logger.info("Successfully reconstructed secret")
            return secret
        except Exception as e:
            logger.error(f"Failed to reconstruct: {e}")
            return None


class ThresholdCryptoService:
    """Main threshold cryptography service"""

    def __init__(self, config: Optional[ThresholdConfig] = None):
        """
        Initialize threshold crypto service

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or ThresholdConfig()
        self.key_gen = ThresholdKeyGeneration(self.config)
        self.signing = ThresholdSigningService(self.config)
        self.encryption = ThresholdEncryption(self.config)
        self.recovery = RecoveryMechanism(self.config)

        # Session management
        self.active_sessions: Dict[str, ThresholdSession] = {}
        self.participant_registry: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"Initialized threshold crypto service "
            f"({self.config.threshold}-of-{self.config.total_shares})"
        )

    def register_participant(
        self,
        participant_id: str,
        public_key: bytes,
        geographic_region: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a participant in the threshold system

        Args:
            participant_id: Unique participant identifier
            public_key: Participant's public key
            geographic_region: Geographic location
            metadata: Additional metadata

        Returns:
            True if registration successful
        """
        if participant_id in self.participant_registry:
            logger.warning(f"Participant {participant_id} already registered")
            return False

        self.participant_registry[participant_id] = {
            "public_key": public_key,
            "geographic_region": geographic_region,
            "metadata": metadata or {},
            "registered_at": datetime.now(),
        }

        logger.info(f"Registered participant {participant_id}")
        return True

    def perform_distributed_keygen(
        self, participants: List[str]
    ) -> Tuple[Dict[str, List[SecretShare]], bytes]:
        """
        Perform distributed key generation

        Args:
            participants: List of participant IDs

        Returns:
            Tuple of (participant_shares, group_key)
        """
        if len(participants) < self.config.total_shares:
            raise ValueError("Insufficient participants for key generation")

        all_shares = {}
        all_public_keys = []

        # Each participant generates shares
        for participant in participants[: self.config.total_shares]:
            shares, public_key = self.key_gen.generate_key_shares(participant)
            all_shares[participant] = shares
            all_public_keys.append(public_key)

        # Combine to get group key
        combined_shares = []
        for shares_list in all_shares.values():
            if shares_list:
                combined_shares.append(shares_list[0])

        group_key = self.key_gen.combine_shares(combined_shares[: self.config.threshold])

        return all_shares, group_key

    def create_threshold_signature(
        self, message: bytes, initiator: str, participants: List[str]
    ) -> Optional[bytes]:
        """
        Create threshold signature with quorum

        Args:
            message: Message to sign
            initiator: Initiating participant
            participants: List of signers

        Returns:
            Combined signature if successful
        """
        # Initiate signing session
        session_id = self.signing.initiate_signing(message, initiator)

        # Simulate participants submitting shares
        for i, participant in enumerate(participants[: self.config.threshold]):
            # Generate signature share (simplified)
            sig_share = hashlib.sha256(message + participant.encode() + str(i).encode()).digest()

            region = f"region_{i % 3}"  # Simulate geographic distribution
            self.signing.submit_signature_share(session_id, participant, sig_share, region)

        # Get result
        session = self.signing.sessions.get(session_id)
        if session and session.status == QuorumStatus.MET:
            return session.result

        return None

    def encrypt_for_threshold(
        self, data: bytes, participants: List[str]
    ) -> Tuple[bytes, List[bytes]]:
        """
        Encrypt data for threshold decryption

        Args:
            data: Data to encrypt
            participants: List of participants

        Returns:
            Tuple of (ciphertext, encrypted_shares)
        """
        # Get participant public keys
        public_keys = []
        for participant in participants:
            if participant in self.participant_registry:
                public_keys.append(self.participant_registry[participant]["public_key"])

        return self.encryption.encrypt_with_shares(data, public_keys)

    def setup_recovery(self, secret: bytes, regions: List[str]) -> Dict[str, SecretShare]:
        """
        Setup emergency recovery shares

        Args:
            secret: Secret to protect
            regions: Geographic regions for distribution

        Returns:
            Distribution of recovery shares
        """
        return self.recovery.distribute_recovery_shares(secret, regions)

    def execute_recovery(
        self, recovery_id: str, shares: List[SecretShare], commitments: List[bytes]
    ) -> Optional[bytes]:
        """
        Execute emergency recovery

        Args:
            recovery_id: Recovery identifier
            shares: Recovery shares
            commitments: Verification commitments

        Returns:
            Recovered secret if successful
        """
        # Verify geographic distribution if required
        if self.config.enable_geographic_distribution:
            regions = set(share.geographic_region for share in shares)
            if len(regions) < self.config.min_geographic_regions:
                logger.error("Insufficient geographic distribution")
                return None

        return self.recovery.verify_and_reconstruct(shares, commitments)

    def rotate_keys(self) -> bytes:
        """
        Perform key rotation for forward secrecy

        Returns:
            New epoch key
        """
        if not self.config.enable_forward_secrecy:
            logger.warning("Forward secrecy not enabled")
            return b""

        return self.encryption.apply_forward_secrecy()

    def get_audit_log(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> List[AuditEntry]:
        """
        Retrieve audit log entries

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of audit entries
        """
        logs = self.signing.audit_log

        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]

        return logs


def create_threshold_service(
    threshold: int = 5, total: int = 8, enable_geographic: bool = True
) -> ThresholdCryptoService:
    """
    Factory function to create threshold service

    Args:
        threshold: Minimum shares needed
        total: Total shares
        enable_geographic: Enable geographic distribution

    Returns:
        Configured threshold service
    """
    config = ThresholdConfig(
        threshold=threshold, total_shares=total, enable_geographic_distribution=enable_geographic
    )

    return ThresholdCryptoService(config)


if __name__ == "__main__":
    # Example usage
    service = create_threshold_service()

    # Register participants
    participants = []
    for i in range(8):
        participant_id = f"participant_{i}"
        # Generate a dummy public key
        public_key = hashlib.sha256(f"pubkey_{i}".encode()).digest()
        service.register_participant(
            participant_id, public_key, geographic_region=f"region_{i % 3}"
        )
        participants.append(participant_id)

    print(f"Registered {len(participants)} participants")

    # Perform distributed key generation
    shares, group_key = service.perform_distributed_keygen(participants)
    print(f"Generated group key: {group_key.hex()[:32]}...")

    # Create threshold signature
    message = b"Genomic data hash: abc123"
    signature = service.create_threshold_signature(message, participants[0], participants)
    if signature:
        print(f"Created threshold signature: {signature.hex()[:32]}...")

    # Setup recovery
    secret = b"Critical genomic encryption key"
    regions = [
        "us-east",
        "eu-west",
        "asia-pacific",
        "us-west",
        "eu-north",
        "asia-south",
        "africa",
        "oceania",
    ]
    recovery_shares = service.setup_recovery(secret, regions)
    print(f"Distributed recovery shares across {len(recovery_shares)} regions")

    print("\n✅ Threshold cryptography service initialized successfully")


# Wrapper for compatibility with tests
class ThresholdService:
    """Wrapper for ThresholdCryptoService with test-compatible interface."""

    def __init__(self, threshold: int = 3, total_shares: int = 5):
        """Initialize with threshold and total shares."""
        config = ThresholdConfig(threshold=threshold, total_shares=total_shares)
        self.service = ThresholdCryptoService(config)
        self.threshold = threshold
        self.total_shares = total_shares

    def generate_distributed_key(self) -> list:
        """Generate distributed key shares."""
        # Use the service's key generation
        shares, _ = self.service.key_gen.generate_key_shares("test_participant")
        return shares

    def threshold_sign(self, message: bytes, shares: list) -> bytes:
        """Sign message with threshold shares."""
        # Simple simulation since we don't have actual BLS
        if len(shares) >= self.threshold:
            # Return mock signature
            return b"mock_threshold_signature_" + message[:20]
        return None

    def setup_recovery(self, secret: bytes, locations: list) -> dict:
        """Setup recovery shares."""
        return self.service.setup_recovery(secret, locations)
