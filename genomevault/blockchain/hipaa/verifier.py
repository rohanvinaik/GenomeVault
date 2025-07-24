"""
HIPAA Verifier Implementation

Core verification logic for HIPAA fast-track system.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from ...core.exceptions import VerificationError
from ...utils import audit_logger, get_logger
from .models import (
    HIPAACredentials,
    NPIRecord,
    NPIType,
    VerificationRecord,
    VerificationStatus,
)

logger = get_logger(__name__)


class NPIRegistry(ABC):
    """Abstract base class for NPI registry access"""

    @abstractmethod
    async def lookup_npi(self, npi: str) -> Optional[NPIRecord]:
        """Look up an NPI in the registry"""
        pass

    @abstractmethod
    async def validate_npi(self, npi: str) -> bool:
        """Validate if an NPI exists and is active"""
        pass


class CMSNPIRegistry(NPIRegistry):
    """
    CMS NPPES (National Plan and Provider Enumeration System) registry client.

    In production, this would connect to the actual CMS API.
    For development, it simulates the registry.
    """

    def __init__(self, api_endpoint: Optional[str] = None):
        """Initialize CMS registry client"""
        self.api_endpoint = api_endpoint or "https://npiregistry.cms.hhs.gov/api"
        self._cache: Dict[str, NPIRecord] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()

    async def lookup_npi(self, npi: str) -> Optional[NPIRecord]:
        """
        Look up an NPI in the CMS registry.

        Args:
            npi: National Provider Identifier

        Returns:
            NPIRecord if found, None otherwise
        """
        # Check cache first
        if npi in self._cache:
            return self._cache[npi]

        # Validate format
        if not self._validate_npi_format(npi):
            return None

        try:
            # In production, make actual API call
            # For now, simulate based on NPI patterns
            record = await self._simulate_npi_lookup(npi)

            if record:
                self._cache[npi] = record

            return record

        except Exception as e:
            logger.error(f"Error looking up NPI {npi}: {e}")
            return None

    async def validate_npi(self, npi: str) -> bool:
        """
        Validate if an NPI exists and is active.

        Args:
            npi: National Provider Identifier

        Returns:
            True if NPI is valid and active
        """
        record = await self.lookup_npi(npi)
        return record is not None and record.is_active

    def _validate_npi_format(self, npi: str) -> bool:
        """
        Validate NPI format using Luhn algorithm.

        NPIs are 10-digit numbers where the last digit is a check digit.
        """
        if not npi or len(npi) != 10 or not npi.isdigit():
            return False

        # Luhn algorithm
        total = 0
        for i, digit in enumerate(npi[:-1]):
            d = int(digit)
            if i % 2 == 0:  # Double every other digit
                d *= 2
                if d > 9:
                    d = d - 9
            total += d

        check_digit = (10 - (total % 10)) % 10
        return int(npi[-1]) == check_digit

    async def _simulate_npi_lookup(self, npi: str) -> Optional[NPIRecord]:
        """
        Simulate NPI lookup for development.

        In production, this would make actual API calls to CMS.
        """
        # Simulate different provider types based on NPI patterns
        if npi.startswith("1"):
            # Individual provider
            return NPIRecord(
                npi=npi,
                npi_type=NPIType.INDIVIDUAL,
                name=f"Dr. Test Provider {npi[-4:]}",
                first_name="Test",
                last_name=f"Provider {npi[-4:]}",
                credential="MD",
                primary_taxonomy="207Q00000X",  # Family Medicine
                is_active=True,
            )
        elif npi.startswith("2"):
            # Organization
            return NPIRecord(
                npi=npi,
                npi_type=NPIType.ORGANIZATION,
                name=f"Test Medical Center {npi[-4:]}",
                organization_name=f"Test Medical Center {npi[-4:]}",
                ein=f"12-34567{npi[-2:]}",
                primary_taxonomy="282N00000X",  # General Acute Care Hospital
                is_active=True,
            )
        else:
            # Simulate some NPIs being inactive
            if npi.startswith("9"):
                return NPIRecord(
                    npi=npi,
                    npi_type=NPIType.INDIVIDUAL,
                    name=f"Inactive Provider {npi[-4:]}",
                    is_active=False,
                    deactivation_date=datetime.now() - timedelta(days=30),
                )
            return None


class HIPAAVerifier:
    """
    Main HIPAA verification service.

    Handles the fast-track verification process for healthcare providers
    to become Trusted Signatories.
    """

    def __init__(self, npi_registry: Optional[NPIRegistry] = None):
        """
        Initialize HIPAA verifier.

        Args:
            npi_registry: NPI registry client (defaults to CMS)
        """
        self.npi_registry = npi_registry or CMSNPIRegistry()
        self.verification_records: Dict[str, VerificationRecord] = {}
        self.pending_verifications: Dict[str, HIPAACredentials] = {}

        # Verification parameters
        self.signatory_weight = 10
        self.honesty_probability = 0.98
        self.verification_expiry_days = 365

        logger.info("HIPAA Verifier initialized")

    async def submit_verification(self, credentials: HIPAACredentials) -> str:
        """
        Submit credentials for HIPAA fast-track verification.

        Args:
            credentials: HIPAA provider credentials

        Returns:
            Verification ID for tracking
        """
        # Generate verification ID
        verification_id = self._generate_verification_id(credentials)

        # Check if already verified
        if credentials.npi in self.verification_records:
            record = self.verification_records[credentials.npi]
            if record.is_active():
                raise VerificationError(f"NPI {credentials.npi} already verified")

        # Store pending verification
        self.pending_verifications[verification_id] = credentials

        # Log submission
        audit_logger.log_event(
            event_type="hipaa_verification_submitted",
            actor=credentials.npi,
            action="submit_verification",
            resource=verification_id,
            metadata={
                "hsm_serial": credentials.hsm_serial,
                "npi_type": (credentials.npi_type.value if credentials.npi_type else None),
            },
        )

        logger.info(f"HIPAA verification submitted: {verification_id}")

        return verification_id

    async def process_verification(self, verification_id: str) -> VerificationRecord:
        """
        Process a pending HIPAA verification.

        Args:
            verification_id: ID of verification to process

        Returns:
            Verification record with results
        """
        if verification_id not in self.pending_verifications:
            raise VerificationError(f"Verification {verification_id} not found")

        credentials = self.pending_verifications[verification_id]

        try:
            # Step 1: Validate NPI format
            if not self.npi_registry._validate_npi_format(credentials.npi):
                raise VerificationError("Invalid NPI format")

            # Step 2: Check CMS registry
            npi_record = await self.npi_registry.lookup_npi(credentials.npi)
            if not npi_record:
                raise VerificationError("NPI not found in CMS registry")

            if not npi_record.is_active:
                raise VerificationError("NPI is not active")

            # Step 3: Validate credentials
            self._validate_credentials(credentials)

            # Step 4: Create verification record
            record = VerificationRecord(
                credentials=credentials,
                status=VerificationStatus.VERIFIED,
                verified_at=datetime.now(),
                verifier_signature=self._generate_signature(credentials),
                signatory_weight=self.signatory_weight,
                honesty_probability=self.honesty_probability,
                cms_data={
                    "name": npi_record.name,
                    "type": npi_record.npi_type.value,
                    "taxonomy": npi_record.primary_taxonomy,
                },
                expires_at=datetime.now() + timedelta(days=self.verification_expiry_days),
            )

            # Store verification
            self.verification_records[credentials.npi] = record
            del self.pending_verifications[verification_id]

            # Audit log success
            audit_logger.log_event(
                event_type="hipaa_verification_completed",
                actor="hipaa_verifier",
                action="verify_provider",
                resource=credentials.npi,
                metadata={
                    "verification_id": verification_id,
                    "provider_name": npi_record.name,
                    "expires_at": record.expires_at.isoformat(),
                },
            )

            logger.info(f"HIPAA verification successful for NPI {credentials.npi}")

            return record

        except VerificationError as e:
            # Create failed record
            record = VerificationRecord(
                credentials=credentials,
                status=VerificationStatus.FAILED,
                verified_at=datetime.now(),
                verifier_signature="",
                cms_data={"error": str(e)},
            )

            # Log failure
            audit_logger.log_event(
                event_type="hipaa_verification_failed",
                actor="hipaa_verifier",
                action="verify_provider",
                resource=credentials.npi,
                metadata={"verification_id": verification_id, "error": str(e)},
            )

            logger.warning(f"HIPAA verification failed for NPI {credentials.npi}: {e}")

            # Clean up pending
            del self.pending_verifications[verification_id]

            raise

    def get_verification_status(self, npi: str) -> Optional[VerificationRecord]:
        """
        Get verification status for an NPI.

        Args:
            npi: National Provider Identifier

        Returns:
            Verification record if exists
        """
        return self.verification_records.get(npi)

    def revoke_verification(self, npi: str, reason: str) -> bool:
        """
        Revoke verification for a provider.

        Args:
            npi: National Provider Identifier
            reason: Reason for revocation

        Returns:
            True if revoked successfully
        """
        if npi not in self.verification_records:
            return False

        record = self.verification_records[npi]
        if record.status != VerificationStatus.VERIFIED:
            return False

        # Update record
        record.status = VerificationStatus.REVOKED
        record.revoked_at = datetime.now()
        record.revocation_reason = reason

        # Audit log
        audit_logger.log_event(
            event_type="hipaa_verification_revoked",
            actor="hipaa_verifier",
            action="revoke_verification",
            resource=npi,
            metadata={"reason": reason},
        )

        logger.info(f"HIPAA verification revoked for NPI {npi}: {reason}")

        return True

    def _generate_verification_id(self, credentials: HIPAACredentials) -> str:
        """Generate unique verification ID"""
        data = f"{credentials.npi}:{credentials.baa_hash}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _validate_credentials(self, credentials: HIPAACredentials):
        """Validate credential components"""
        # In production, would verify:
        # - BAA hash matches expected format/content
        # - Risk analysis hash is recent and valid
        # - HSM serial is registered and valid

        # For now, basic validation
        if len(credentials.baa_hash) != 64:
            raise VerificationError("Invalid BAA hash")

        if len(credentials.risk_analysis_hash) != 64:
            raise VerificationError("Invalid risk analysis hash")

        if not credentials.hsm_serial:
            raise VerificationError("Missing HSM serial")

    def _generate_signature(self, credentials: HIPAACredentials) -> str:
        """Generate verification signature"""
        data = json.dumps(
            {
                "npi": credentials.npi,
                "baa_hash": credentials.baa_hash,
                "risk_analysis_hash": credentials.risk_analysis_hash,
                "hsm_serial": credentials.hsm_serial,
                "timestamp": datetime.now().isoformat(),
            },
            sort_keys=True,
        )

        return hashlib.sha256(data.encode()).hexdigest()

    def get_active_verifications(self) -> List[Tuple[str, VerificationRecord]]:
        """Get all active verifications"""
        active = []
        for npi, record in self.verification_records.items():
            if record.is_active():
                active.append((npi, record))
        return active

    def cleanup_expired(self) -> int:
        """Clean up expired verifications"""
        expired_count = 0
        for npi, record in list(self.verification_records.items()):
            if record.expires_at and datetime.now() > record.expires_at:
                record.status = VerificationStatus.EXPIRED
                expired_count += 1

                logger.info(f"Expired HIPAA verification for NPI {npi}")

        return expired_count


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_hipaa_verification():
        """Test HIPAA verification flow"""

        # Initialize verifier with CMS registry
        async with CMSNPIRegistry() as registry:
            verifier = HIPAAVerifier(npi_registry=registry)

            # Test credentials
            credentials = HIPAACredentials(
                npi="1234567893",  # Valid NPI with correct check digit
                baa_hash="a" * 64,  # SHA256 hash
                risk_analysis_hash="b" * 64,  # SHA256 hash
                hsm_serial="HSM-12345-ABCDE",
                provider_name="Dr. John Smith",
            )

            print("Submitting HIPAA verification...")
            verification_id = await verifier.submit_verification(credentials)
            print(f"Verification ID: {verification_id}")

            print("\nProcessing verification...")
            try:
                record = await verifier.process_verification(verification_id)
                print("Verification successful!")
                print(f"  Status: {record.status.value}")
                print(f"  Signatory weight: {record.signatory_weight}")
                print(f"  Honesty probability: {record.honesty_probability}")
                print(f"  Expires: {record.expires_at}")

                # Check status
                status = verifier.get_verification_status(credentials.npi)
                print(f"\nVerification active: {status.is_active()}")

            except VerificationError as e:
                print(f"Verification failed: {e}")

            # Test invalid NPI
            print("\n\nTesting invalid NPI...")
            bad_credentials = HIPAACredentials(
                npi="1234567890",  # Invalid check digit
                baa_hash="c" * 64,
                risk_analysis_hash="d" * 64,
                hsm_serial="HSM-99999",
            )

            try:
                bad_id = await verifier.submit_verification(bad_credentials)
                await verifier.process_verification(bad_id)
            except VerificationError as e:
                print(f"Correctly rejected: {e}")

    # Run test
    asyncio.run(test_hipaa_verification())
