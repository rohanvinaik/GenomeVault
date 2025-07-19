"""
HIPAA Verification Models

Data models for HIPAA fast-track verification system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class VerificationStatus(Enum):
    """Status of HIPAA verification"""

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    REVOKED = "revoked"
    EXPIRED = "expired"


class NPIType(Enum):
    """Type of National Provider Identifier"""

    INDIVIDUAL = 1  # Healthcare providers
    ORGANIZATION = 2  # Healthcare organizations


@dataclass
class HIPAACredentials:
    """HIPAA provider credentials for fast-track verification"""

    npi: str  # National Provider Identifier (10 digits)
    baa_hash: str  # SHA256 hash of Business Associate Agreement
    risk_analysis_hash: str  # SHA256 hash of HIPAA risk analysis
    hsm_serial: str  # Hardware Security Module serial number

    # Optional fields
    organization_name: Optional[str] = None
    provider_name: Optional[str] = None
    npi_type: Optional[NPIType] = None

    def __post_init__(self):
        """Validate credentials format"""
        if not self.npi or len(self.npi) != 10 or not self.npi.isdigit():
            raise ValueError("NPI must be 10 digits")

        if not self.baa_hash or len(self.baa_hash) != 64:
            raise ValueError("BAA hash must be 64 characters (SHA256)")

        if not self.risk_analysis_hash or len(self.risk_analysis_hash) != 64:
            raise ValueError("Risk analysis hash must be 64 characters (SHA256)")

        if not self.hsm_serial:
            raise ValueError("HSM serial number is required")


@dataclass
class VerificationRecord:
    """Record of HIPAA verification"""

    credentials: HIPAACredentials
    status: VerificationStatus
    verified_at: datetime
    verifier_signature: str

    # Verification details
    signatory_weight: int = 10  # Trusted signatory weight
    honesty_probability: float = 0.98  # Higher for HIPAA-compliant

    # CMS registry data
    cms_data: Optional[Dict[str, Any]] = None

    # Revocation info if applicable
    revoked_at: Optional[datetime] = None
    revocation_reason: Optional[str] = None

    # Expiration
    expires_at: Optional[datetime] = None

    def is_active(self) -> bool:
        """Check if verification is currently active"""
        if self.status != VerificationStatus.VERIFIED:
            return False

        if self.revoked_at is not None:
            return False

        if self.expires_at and datetime.now() > self.expires_at:
            return False

        return True

    def to_chain_data(self) -> Dict[str, Any]:
        """Convert to data for blockchain storage"""
        return {
            "npi": self.credentials.npi,
            "baa_hash": self.credentials.baa_hash,
            "risk_analysis_hash": self.credentials.risk_analysis_hash,
            "hsm_serial": self.credentials.hsm_serial,
            "status": self.status.value,
            "verified_at": self.verified_at.isoformat(),
            "signatory_weight": self.signatory_weight,
            "honesty_probability": self.honesty_probability,
            "verifier_signature": self.verifier_signature,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class NPIRecord:
    """National Provider Identifier registry record"""

    npi: str
    npi_type: NPIType
    name: str

    # Organization fields
    organization_name: Optional[str] = None
    ein: Optional[str] = None  # Employer Identification Number

    # Individual fields
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    credential: Optional[str] = None  # MD, DO, etc.

    # Common fields
    primary_taxonomy: Optional[str] = None  # Provider specialty
    address: Optional[Dict[str, str]] = None
    phone: Optional[str] = None

    # Status
    is_active: bool = True
    deactivation_date: Optional[datetime] = None
    reactivation_date: Optional[datetime] = None

    def __str__(self) -> str:
        if self.npi_type == NPIType.ORGANIZATION:
            return f"{self.organization_name} (NPI: {self.npi})"
        else:
            return f"{self.first_name} {self.last_name}, {self.credential} (NPI: {self.npi})"
