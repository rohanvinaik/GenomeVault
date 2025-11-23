"""
NPI Verification System for GenomeVault Phase 2

Provides comprehensive National Provider Identifier (NPI) verification
using the CMS NPPES NPI Registry API.

Features:
- Real-time NPI validation against CMS registry
- Local caching for performance
- Batch verification support
- HIPAA credential validation
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import requests

from genomevault.utils.logging import get_logger

from .models import HIPAACredentials, NPIRecord, NPIType, VerificationStatus

logger = get_logger(__name__)


@dataclass
class NPIVerificationResult:
    """Result of NPI verification"""

    npi: str
    is_valid: bool
    status: VerificationStatus
    npi_record: Optional[NPIRecord] = None
    error_message: Optional[str] = None
    verified_at: Optional[datetime] = None
    cms_data: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "npi": self.npi,
            "is_valid": self.is_valid,
            "status": self.status.value,
            "npi_record": asdict(self.npi_record) if self.npi_record else None,
            "error_message": self.error_message,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "cms_data": self.cms_data,
        }


class CMSNPIRegistry:
    """
    Interface to CMS NPPES NPI Registry.

    Uses the public NPPES NPI Registry API for real-time validation.
    Implements caching and rate limiting for production use.
    """

    # CMS NPPES API endpoint
    NPPES_API_URL = "https://npiregistry.cms.hhs.gov/api/"
    NPPES_VERSION = "2.1"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 24,
        enable_cache: bool = True,
        rate_limit_per_second: int = 10,
    ):
        """
        Initialize NPI registry client.

        Args:
            cache_dir: Directory for caching NPI lookups
            cache_ttl_hours: Cache time-to-live in hours
            enable_cache: Whether to enable local caching
            rate_limit_per_second: Max API requests per second
        """
        self.cache_dir = cache_dir or Path.home() / ".genomevault" / "npi_cache"
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.enable_cache = enable_cache
        self.rate_limit = rate_limit_per_second
        self.last_request_time = 0.0

        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"NPI Registry initialized (cache={'enabled' if enable_cache else 'disabled'}, "
            f"rate_limit={rate_limit_per_second}/s)"
        )

    def lookup_npi(self, npi: str) -> Optional[NPIRecord]:
        """
        Look up NPI in CMS NPPES registry.

        Args:
            npi: 10-digit National Provider Identifier

        Returns:
            NPIRecord if found, None otherwise
        """
        # Validate NPI format
        if not npi or len(npi) != 10 or not npi.isdigit():
            logger.error(f"Invalid NPI format: {npi}")
            return None

        # Check cache first
        if self.enable_cache:
            cached = self._get_cached_npi(npi)
            if cached:
                logger.debug(f"NPI {npi} found in cache")
                return cached

        # Rate limiting
        self._rate_limit()

        # Query NPPES API
        try:
            url = f"{self.NPPES_API_URL}?version={self.NPPES_VERSION}&number={npi}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Parse response
            if data.get("result_count", 0) == 0:
                logger.warning(f"NPI {npi} not found in NPPES registry")
                return None

            # Extract NPI data
            result = data["results"][0]
            npi_record = self._parse_nppes_response(result)

            # Cache result
            if self.enable_cache and npi_record:
                self._cache_npi(npi, npi_record, data)

            logger.info(f"Successfully validated NPI {npi}")
            return npi_record

        except requests.RequestException as e:
            logger.error(f"NPPES API request failed for NPI {npi}: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse NPPES response for NPI {npi}: {e}")
            return None

    def batch_lookup(self, npis: list[str]) -> dict[str, Optional[NPIRecord]]:
        """
        Batch lookup multiple NPIs.

        Args:
            npis: List of NPIs to look up

        Returns:
            Dictionary mapping NPI to NPIRecord (or None if not found)
        """
        results = {}
        for npi in npis:
            results[npi] = self.lookup_npi(npi)
            # Rate limiting between requests
            time.sleep(1.0 / self.rate_limit)

        logger.info(f"Batch lookup completed: {len(results)} NPIs")
        return results

    def verify_npi_active(self, npi: str) -> bool:
        """
        Verify NPI is currently active.

        Args:
            npi: National Provider Identifier

        Returns:
            True if NPI is active
        """
        record = self.lookup_npi(npi)
        return record is not None and record.is_active

    def _parse_nppes_response(self, data: dict[str, Any]) -> NPIRecord:
        """Parse NPPES API response into NPIRecord"""
        # Determine NPI type
        enumeration_type = data.get("enumeration_type")
        if enumeration_type == "NPI-1":
            npi_type = NPIType.INDIVIDUAL
        elif enumeration_type == "NPI-2":
            npi_type = NPIType.ORGANIZATION
        else:
            npi_type = NPIType.INDIVIDUAL  # Default

        # Extract common fields
        npi = data.get("number")
        basic = data.get("basic", {})

        # Organization fields
        organization_name = None
        ein = None
        if npi_type == NPIType.ORGANIZATION:
            organization_name = basic.get("organization_name")
            ein = basic.get("ein")

        # Individual fields
        first_name = None
        last_name = None
        credential = None
        if npi_type == NPIType.INDIVIDUAL:
            first_name = basic.get("first_name")
            last_name = basic.get("last_name")
            credential = basic.get("credential")

        # Common fields
        name = organization_name if npi_type == NPIType.ORGANIZATION else f"{first_name} {last_name}"

        # Taxonomy (specialty)
        taxonomies = data.get("taxonomies", [])
        primary_taxonomy = None
        if taxonomies:
            primary_taxonomy = taxonomies[0].get("desc")

        # Address
        addresses = data.get("addresses", [])
        address = None
        if addresses:
            addr = addresses[0]
            address = {
                "address_1": addr.get("address_1"),
                "address_2": addr.get("address_2"),
                "city": addr.get("city"),
                "state": addr.get("state"),
                "postal_code": addr.get("postal_code"),
                "country_code": addr.get("country_code"),
            }

        # Phone
        phone = addresses[0].get("telephone_number") if addresses else None

        # Status
        status = basic.get("status")
        is_active = status == "A"  # A = Active

        # Create record
        record = NPIRecord(
            npi=npi,
            npi_type=npi_type,
            name=name,
            organization_name=organization_name,
            ein=ein,
            first_name=first_name,
            last_name=last_name,
            credential=credential,
            primary_taxonomy=primary_taxonomy,
            address=address,
            phone=phone,
            is_active=is_active,
        )

        return record

    def _get_cached_npi(self, npi: str) -> Optional[NPIRecord]:
        """Get NPI record from cache"""
        cache_file = self.cache_dir / f"{npi}.json"

        if not cache_file.exists():
            return None

        try:
            # Check if cache is expired
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age > self.cache_ttl:
                logger.debug(f"Cache expired for NPI {npi}")
                cache_file.unlink()
                return None

            # Load from cache
            with open(cache_file) as f:
                data = json.load(f)

            # Reconstruct NPIRecord
            npi_type = NPIType.ORGANIZATION if data["npi_type"] == "ORGANIZATION" else NPIType.INDIVIDUAL

            record = NPIRecord(
                npi=data["npi"],
                npi_type=npi_type,
                name=data["name"],
                organization_name=data.get("organization_name"),
                ein=data.get("ein"),
                first_name=data.get("first_name"),
                last_name=data.get("last_name"),
                credential=data.get("credential"),
                primary_taxonomy=data.get("primary_taxonomy"),
                address=data.get("address"),
                phone=data.get("phone"),
                is_active=data.get("is_active", True),
            )

            return record

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cached NPI {npi}: {e}")
            return None

    def _cache_npi(self, npi: str, record: NPIRecord, raw_data: dict):
        """Cache NPI record to disk"""
        cache_file = self.cache_dir / f"{npi}.json"

        try:
            cache_data = {
                "npi": record.npi,
                "npi_type": record.npi_type.name,
                "name": record.name,
                "organization_name": record.organization_name,
                "ein": record.ein,
                "first_name": record.first_name,
                "last_name": record.last_name,
                "credential": record.credential,
                "primary_taxonomy": record.primary_taxonomy,
                "address": record.address,
                "phone": record.phone,
                "is_active": record.is_active,
                "cached_at": datetime.now().isoformat(),
                "raw_cms_data": raw_data,
            }

            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            logger.debug(f"Cached NPI {npi}")

        except IOError as e:
            logger.warning(f"Failed to cache NPI {npi}: {e}")

    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.rate_limit

        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def clear_cache(self):
        """Clear all cached NPI records"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("NPI cache cleared")


class HIPAACredentialVerifier:
    """
    Verifies HIPAA credentials for institutional onboarding.

    Validates:
    - NPI exists and is active
    - Business Associate Agreement (BAA) hash
    - HIPAA Risk Analysis hash
    - Hardware Security Module (HSM) serial
    """

    def __init__(self, npi_registry: CMSNPIRegistry):
        """
        Initialize credential verifier.

        Args:
            npi_registry: CMS NPI registry client
        """
        self.npi_registry = npi_registry
        self.verification_history: dict[str, list[NPIVerificationResult]] = {}

        logger.info("HIPAA credential verifier initialized")

    def verify_credentials(self, credentials: HIPAACredentials) -> NPIVerificationResult:
        """
        Verify HIPAA credentials.

        Args:
            credentials: HIPAA credentials to verify

        Returns:
            Verification result
        """
        logger.info(f"Verifying credentials for NPI {credentials.npi}")

        # Step 1: Validate NPI format
        try:
            credentials.__post_init__()  # Trigger validation
        except ValueError as e:
            return NPIVerificationResult(
                npi=credentials.npi,
                is_valid=False,
                status=VerificationStatus.FAILED,
                error_message=f"Invalid credential format: {e}",
                verified_at=datetime.now(),
            )

        # Step 2: Look up NPI in CMS registry
        npi_record = self.npi_registry.lookup_npi(credentials.npi)

        if not npi_record:
            return NPIVerificationResult(
                npi=credentials.npi,
                is_valid=False,
                status=VerificationStatus.FAILED,
                error_message="NPI not found in CMS NPPES registry",
                verified_at=datetime.now(),
            )

        # Step 3: Verify NPI is active
        if not npi_record.is_active:
            return NPIVerificationResult(
                npi=credentials.npi,
                is_valid=False,
                status=VerificationStatus.FAILED,
                npi_record=npi_record,
                error_message="NPI is not active",
                verified_at=datetime.now(),
            )

        # Step 4: Validate BAA hash (in production, verify signature)
        if not self._validate_baa_hash(credentials.baa_hash):
            return NPIVerificationResult(
                npi=credentials.npi,
                is_valid=False,
                status=VerificationStatus.FAILED,
                npi_record=npi_record,
                error_message="Invalid BAA hash",
                verified_at=datetime.now(),
            )

        # Step 5: Validate Risk Analysis hash
        if not self._validate_risk_analysis(credentials.risk_analysis_hash):
            return NPIVerificationResult(
                npi=credentials.npi,
                is_valid=False,
                status=VerificationStatus.FAILED,
                npi_record=npi_record,
                error_message="Invalid risk analysis hash",
                verified_at=datetime.now(),
            )

        # Step 6: Validate HSM serial (in production, verify HSM is authorized)
        if not self._validate_hsm(credentials.hsm_serial):
            return NPIVerificationResult(
                npi=credentials.npi,
                is_valid=False,
                status=VerificationStatus.FAILED,
                npi_record=npi_record,
                error_message="Invalid or unauthorized HSM",
                verified_at=datetime.now(),
            )

        # All checks passed
        result = NPIVerificationResult(
            npi=credentials.npi,
            is_valid=True,
            status=VerificationStatus.VERIFIED,
            npi_record=npi_record,
            verified_at=datetime.now(),
            cms_data={
                "name": npi_record.name,
                "taxonomy": npi_record.primary_taxonomy,
                "npi_type": npi_record.npi_type.name,
            },
        )

        # Record verification
        if credentials.npi not in self.verification_history:
            self.verification_history[credentials.npi] = []
        self.verification_history[credentials.npi].append(result)

        logger.info(f"Credentials verified for NPI {credentials.npi}: {npi_record.name}")
        return result

    def batch_verify(
        self, credentials_list: list[HIPAACredentials]
    ) -> dict[str, NPIVerificationResult]:
        """
        Batch verify multiple credentials.

        Args:
            credentials_list: List of credentials to verify

        Returns:
            Dictionary mapping NPI to verification result
        """
        results = {}
        for creds in credentials_list:
            results[creds.npi] = self.verify_credentials(creds)

        logger.info(f"Batch verification completed: {len(results)} credentials")
        return results

    def get_verification_history(self, npi: str) -> list[NPIVerificationResult]:
        """Get verification history for an NPI"""
        return self.verification_history.get(npi, [])

    def _validate_baa_hash(self, baa_hash: str) -> bool:
        """Validate Business Associate Agreement hash"""
        # In production, this would:
        # 1. Verify hash signature
        # 2. Check against registry of approved BAA templates
        # 3. Validate expiration date encoded in hash

        # For now, just check format
        return len(baa_hash) == 64 and all(c in "0123456789abcdef" for c in baa_hash.lower())

    def _validate_risk_analysis(self, risk_hash: str) -> bool:
        """Validate HIPAA Risk Analysis hash"""
        # In production, this would:
        # 1. Verify hash signature
        # 2. Check completeness of risk analysis
        # 3. Validate required sections are present

        # For now, just check format
        return len(risk_hash) == 64 and all(c in "0123456789abcdef" for c in risk_hash.lower())

    def _validate_hsm(self, hsm_serial: str) -> bool:
        """Validate Hardware Security Module"""
        # In production, this would:
        # 1. Check HSM is from approved vendor (Thales, AWS CloudHSM, etc.)
        # 2. Verify HSM is FIPS 140-2 Level 3 certified
        # 3. Check HSM is registered in institutional registry

        # For now, just check format
        return bool(hsm_serial) and len(hsm_serial) > 0


# Factory function for easy initialization
def create_npi_verifier(
    cache_dir: Optional[Path] = None,
    enable_cache: bool = True,
) -> HIPAACredentialVerifier:
    """
    Create configured NPI verifier.

    Args:
        cache_dir: Optional cache directory
        enable_cache: Whether to enable caching

    Returns:
        Configured HIPAACredentialVerifier
    """
    registry = CMSNPIRegistry(
        cache_dir=cache_dir,
        enable_cache=enable_cache,
    )

    verifier = HIPAACredentialVerifier(npi_registry=registry)

    return verifier
