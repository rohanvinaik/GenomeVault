"""
Consent management system for GenomeVault.

Implements granular consent management with:
- Separate consent for research vs clinical use
- Consent withdrawal mechanisms
- Audit trail for all consent changes
- GDPR Article 7 compliance (consent conditions)
- Dynamic consent for emerging use cases
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)

# Metrics
consent_operations = Counter(
    "genomevault_consent_operations_total",
    "Total consent operations",
    ["operation_type", "consent_type"],
)

active_consents = Gauge(
    "genomevault_active_consents", "Current number of active consents", ["consent_type", "purpose"]
)

consent_withdrawals = Counter(
    "genomevault_consent_withdrawals_total",
    "Total consent withdrawals",
    ["consent_type", "withdrawal_reason"],
)


class ConsentType(Enum):
    """Types of consent."""

    CLINICAL = "clinical"  # Clinical care and diagnosis
    RESEARCH = "research"  # Research participation
    COMMERCIAL = "commercial"  # Commercial use of data
    SHARING = "sharing"  # Data sharing with partners
    MARKETING = "marketing"  # Marketing communications
    ANALYTICS = "analytics"  # Analytics and insights


class ConsentStatus(Enum):
    """Consent status."""

    ACTIVE = "active"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"
    REVOKED = "revoked"


class ConsentPurpose(Enum):
    """Specific purposes for data use."""

    # Clinical purposes
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    CARE_COORDINATION = "care_coordination"

    # Research purposes
    DRUG_DISCOVERY = "drug_discovery"
    POPULATION_STUDIES = "population_studies"
    RARE_DISEASE = "rare_disease"
    PHARMACOGENOMICS = "pharmacogenomics"

    # Commercial purposes
    PRODUCT_DEVELOPMENT = "product_development"
    LICENSING = "licensing"

    # Administrative
    QUALITY_IMPROVEMENT = "quality_improvement"
    BILLING = "billing"
    COMPLIANCE = "compliance"


@dataclass
class ConsentRecord:
    """Individual consent record."""

    consent_id: str
    user_id: str
    consent_type: ConsentType
    purposes: List[ConsentPurpose]
    status: ConsentStatus
    granted_at: datetime
    expires_at: Optional[datetime]
    withdrawn_at: Optional[datetime]
    withdrawal_reason: Optional[str] = None

    # Consent details
    specific_permissions: Dict[str, bool] = field(default_factory=dict)
    data_sharing_partners: List[str] = field(default_factory=list)
    geographic_restrictions: List[str] = field(default_factory=list)

    # Metadata
    consent_version: str = "1.0"
    language: str = "en"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_method: str = "web_form"  # web_form, paper, verbal, etc.

    # Audit trail
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    def is_active(self) -> bool:
        """Check if consent is currently active."""
        if self.status != ConsentStatus.ACTIVE:
            return False

        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False

        return True

    def is_expired(self) -> bool:
        """Check if consent has expired."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return True
        return False

    def add_audit_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Add audit trail entry."""
        self.audit_trail.append(
            {"timestamp": datetime.utcnow().isoformat(), "action": action, "details": details}
        )

    def has_permission(self, purpose: ConsentPurpose) -> bool:
        """Check if consent covers specific purpose."""
        return self.is_active() and purpose in self.purposes

    def allows_data_sharing(self, partner: str) -> bool:
        """Check if data sharing is allowed with partner."""
        if not self.is_active():
            return False

        if (
            ConsentType.SHARING not in [self.consent_type]
            and ConsentPurpose.CARE_COORDINATION not in self.purposes
        ):
            return False

        # Check if partner is explicitly allowed
        if self.data_sharing_partners:
            return partner in self.data_sharing_partners

        return False


@dataclass
class ConsentTemplate:
    """Template for consent forms."""

    template_id: str
    name: str
    description: str
    consent_type: ConsentType
    purposes: List[ConsentPurpose]
    required_permissions: List[str]
    optional_permissions: List[str] = field(default_factory=list)
    default_expiry_days: Optional[int] = None
    renewal_required: bool = True

    # Form configuration
    form_sections: List[Dict[str, Any]] = field(default_factory=list)
    legal_text: Dict[str, str] = field(default_factory=dict)  # Keyed by language

    def generate_form(self, user_id: str, language: str = "en") -> Dict[str, Any]:
        """Generate consent form for user."""
        return {
            "template_id": self.template_id,
            "user_id": user_id,
            "language": language,
            "title": self.name,
            "description": self.description,
            "sections": self.form_sections,
            "legal_text": self.legal_text.get(language, self.legal_text.get("en", "")),
            "required_permissions": self.required_permissions,
            "optional_permissions": self.optional_permissions,
            "expiry_info": (
                f"This consent expires in {self.default_expiry_days} days"
                if self.default_expiry_days
                else None
            ),
        }


class ConsentManager:
    """Main consent management system."""

    def __init__(self, storage_path: str = "/var/genomevault/consent"):
        self.storage_path = storage_path

        # In-memory storage (would be replaced with database)
        self.consents: Dict[str, ConsentRecord] = {}
        self.user_consents: Dict[str, List[str]] = {}  # user_id -> [consent_ids]
        self.templates: Dict[str, ConsentTemplate] = {}

        # Initialize default templates
        self._init_default_templates()

        # Consent validation rules
        self.validation_rules: Dict[ConsentType, Dict[str, Any]] = {
            ConsentType.CLINICAL: {
                "max_expiry_days": None,  # No expiry for clinical
                "renewable": True,
                "withdrawal_grace_period": 30,  # days
            },
            ConsentType.RESEARCH: {
                "max_expiry_days": 1095,  # 3 years
                "renewable": True,
                "withdrawal_grace_period": 0,
            },
            ConsentType.COMMERCIAL: {
                "max_expiry_days": 365,  # 1 year
                "renewable": False,
                "withdrawal_grace_period": 7,
            },
        }

    def _init_default_templates(self) -> None:
        """Initialize default consent templates."""

        # Clinical consent template
        clinical_template = ConsentTemplate(
            template_id="clinical_v1",
            name="Clinical Care Consent",
            description="Consent for clinical care and treatment",
            consent_type=ConsentType.CLINICAL,
            purposes=[
                ConsentPurpose.DIAGNOSIS,
                ConsentPurpose.TREATMENT,
                ConsentPurpose.CARE_COORDINATION,
            ],
            required_permissions=[
                "genomic_analysis",
                "clinical_reporting",
                "provider_communication",
            ],
            optional_permissions=["quality_improvement", "anonymous_research"],
            form_sections=[
                {
                    "section": "primary_use",
                    "title": "Primary Clinical Use",
                    "description": "We will analyze your genomic data to provide clinical insights",
                },
                {
                    "section": "data_sharing",
                    "title": "Clinical Data Sharing",
                    "description": "Sharing with your healthcare providers for coordinated care",
                },
            ],
        )

        # Research consent template
        research_template = ConsentTemplate(
            template_id="research_v1",
            name="Research Participation Consent",
            description="Consent for research participation",
            consent_type=ConsentType.RESEARCH,
            purposes=[ConsentPurpose.DRUG_DISCOVERY, ConsentPurpose.POPULATION_STUDIES],
            required_permissions=["anonymized_analysis", "aggregate_reporting"],
            optional_permissions=[
                "longitudinal_follow_up",
                "recontact_for_studies",
                "international_collaboration",
            ],
            default_expiry_days=1095,  # 3 years
            form_sections=[
                {
                    "section": "research_purposes",
                    "title": "Research Purposes",
                    "description": "Your data will be used for advancing genomic medicine research",
                },
                {
                    "section": "data_retention",
                    "title": "Data Retention",
                    "description": "Research data may be retained for long-term studies",
                },
            ],
        )

        self.templates["clinical_v1"] = clinical_template
        self.templates["research_v1"] = research_template

    def grant_consent(
        self,
        user_id: str,
        template_id: str,
        granted_permissions: List[str],
        data_sharing_partners: Optional[List[str]] = None,
        expiry_days: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> ConsentRecord:
        """Grant consent using template."""

        if template_id not in self.templates:
            raise ValueError(f"Unknown consent template: {template_id}")

        template = self.templates[template_id]

        # Validate permissions
        required_perms = set(template.required_permissions)
        granted_perms = set(granted_permissions)

        if not required_perms.issubset(granted_perms):
            missing = required_perms - granted_perms
            raise ValueError(f"Missing required permissions: {missing}")

        # Calculate expiry
        expires_at = None
        if expiry_days:
            expires_at = datetime.utcnow() + timedelta(days=expiry_days)
        elif template.default_expiry_days:
            expires_at = datetime.utcnow() + timedelta(days=template.default_expiry_days)

        # Create consent record
        consent = ConsentRecord(
            consent_id=str(uuid4()),
            user_id=user_id,
            consent_type=template.consent_type,
            purposes=template.purposes.copy(),
            status=ConsentStatus.ACTIVE,
            granted_at=datetime.utcnow(),
            expires_at=expires_at,
            specific_permissions={
                perm: perm in granted_permissions
                for perm in (template.required_permissions + template.optional_permissions)
            },
            data_sharing_partners=data_sharing_partners or [],
            ip_address=ip_address,
            user_agent=user_agent,
            consent_method="web_form",
        )

        consent.add_audit_entry(
            "consent_granted",
            {
                "template_id": template_id,
                "permissions": granted_permissions,
                "expiry": expires_at.isoformat() if expires_at else None,
            },
        )

        # Store consent
        self.consents[consent.consent_id] = consent

        if user_id not in self.user_consents:
            self.user_consents[user_id] = []
        self.user_consents[user_id].append(consent.consent_id)

        # Update metrics
        consent_operations.labels(
            operation_type="grant", consent_type=template.consent_type.value
        ).inc()

        active_consents.labels(
            consent_type=template.consent_type.value,
            purpose=",".join([p.value for p in template.purposes]),
        ).inc()

        logger.info(f"Granted {template.consent_type.value} consent for user {user_id}")

        return consent

    def withdraw_consent(
        self,
        user_id: str,
        consent_id: str,
        withdrawal_reason: str,
        effective_date: Optional[datetime] = None,
    ) -> bool:
        """Withdraw specific consent."""

        if consent_id not in self.consents:
            return False

        consent = self.consents[consent_id]

        if consent.user_id != user_id:
            return False

        if consent.status != ConsentStatus.ACTIVE:
            return False

        # Apply grace period if applicable
        rules = self.validation_rules.get(consent.consent_type, {})
        grace_period = rules.get("withdrawal_grace_period", 0)

        if effective_date is None:
            effective_date = datetime.utcnow() + timedelta(days=grace_period)

        consent.status = ConsentStatus.WITHDRAWN
        consent.withdrawn_at = effective_date
        consent.withdrawal_reason = withdrawal_reason

        consent.add_audit_entry(
            "consent_withdrawn",
            {
                "reason": withdrawal_reason,
                "effective_date": effective_date.isoformat(),
                "grace_period_days": grace_period,
            },
        )

        # Update metrics
        consent_withdrawals.labels(
            consent_type=consent.consent_type.value, withdrawal_reason=withdrawal_reason
        ).inc()

        active_consents.labels(
            consent_type=consent.consent_type.value,
            purpose=",".join([p.value for p in consent.purposes]),
        ).dec()

        logger.info(f"Withdrew consent {consent_id} for user {user_id}")

        return True

    def withdraw_all_consents(
        self, user_id: str, withdrawal_reason: str = "user_request"
    ) -> List[str]:
        """Withdraw all active consents for user."""

        if user_id not in self.user_consents:
            return []

        withdrawn_consents = []

        for consent_id in self.user_consents[user_id]:
            if self.withdraw_consent(user_id, consent_id, withdrawal_reason):
                withdrawn_consents.append(consent_id)

        return withdrawn_consents

    def check_consent(
        self, user_id: str, purpose: ConsentPurpose, partner: Optional[str] = None
    ) -> bool:
        """Check if user has active consent for purpose."""

        if user_id not in self.user_consents:
            return False

        for consent_id in self.user_consents[user_id]:
            consent = self.consents.get(consent_id)
            if not consent:
                continue

            if not consent.has_permission(purpose):
                continue

            # Check data sharing permission if partner specified
            if partner and not consent.allows_data_sharing(partner):
                continue

            return True

        return False

    def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get all consents for user."""
        if user_id not in self.user_consents:
            return []

        consents = []
        for consent_id in self.user_consents[user_id]:
            consent = self.consents.get(consent_id)
            if consent:
                consents.append(consent)

        return consents

    def get_active_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get active consents for user."""
        return [c for c in self.get_user_consents(user_id) if c.is_active()]

    def renew_consent(self, user_id: str, consent_id: str, new_expiry_days: int) -> bool:
        """Renew existing consent."""

        if consent_id not in self.consents:
            return False

        consent = self.consents[consent_id]

        if consent.user_id != user_id:
            return False

        # Check if renewal is allowed
        rules = self.validation_rules.get(consent.consent_type, {})
        if not rules.get("renewable", False):
            return False

        # Check max expiry
        max_days = rules.get("max_expiry_days")
        if max_days and new_expiry_days > max_days:
            return False

        old_expiry = consent.expires_at
        consent.expires_at = datetime.utcnow() + timedelta(days=new_expiry_days)
        consent.status = ConsentStatus.ACTIVE

        consent.add_audit_entry(
            "consent_renewed",
            {
                "old_expiry": old_expiry.isoformat() if old_expiry else None,
                "new_expiry": consent.expires_at.isoformat(),
                "extension_days": new_expiry_days,
            },
        )

        return True

    def get_consent_summary(self, user_id: str) -> Dict[str, Any]:
        """Get consent summary for user."""
        consents = self.get_user_consents(user_id)
        active_consents_list = [c for c in consents if c.is_active()]

        # Group by type
        by_type = {}
        for consent in active_consents_list:
            consent_type = consent.consent_type.value
            if consent_type not in by_type:
                by_type[consent_type] = []
            by_type[consent_type].append(
                {
                    "consent_id": consent.consent_id,
                    "purposes": [p.value for p in consent.purposes],
                    "granted_at": consent.granted_at.isoformat(),
                    "expires_at": consent.expires_at.isoformat() if consent.expires_at else None,
                    "permissions": consent.specific_permissions,
                }
            )

        return {
            "user_id": user_id,
            "total_consents": len(consents),
            "active_consents": len(active_consents_list),
            "consents_by_type": by_type,
            "last_updated": datetime.utcnow().isoformat(),
        }

    def expire_consents(self) -> List[str]:
        """Check and expire old consents. Returns list of expired consent IDs."""
        now = datetime.utcnow()
        expired_ids = []

        for consent_id, consent in self.consents.items():
            if consent.status == ConsentStatus.ACTIVE and consent.is_expired():
                consent.status = ConsentStatus.EXPIRED
                consent.add_audit_entry("consent_expired", {"expired_at": now.isoformat()})
                expired_ids.append(consent_id)

                # Update metrics
                active_consents.labels(
                    consent_type=consent.consent_type.value,
                    purpose=",".join([p.value for p in consent.purposes]),
                ).dec()

        if expired_ids:
            logger.info(f"Expired {len(expired_ids)} consents")

        return expired_ids

    def create_template(self, template: ConsentTemplate) -> None:
        """Create new consent template."""
        self.templates[template.template_id] = template
        logger.info(f"Created consent template: {template.template_id}")

    def get_template(self, template_id: str) -> Optional[ConsentTemplate]:
        """Get consent template."""
        return self.templates.get(template_id)

    def list_templates(self) -> List[ConsentTemplate]:
        """List all consent templates."""
        return list(self.templates.values())

    def validate_data_use(
        self,
        user_id: str,
        purpose: ConsentPurpose,
        data_elements: List[str],
        partner: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate if data use is permitted under current consents."""

        # Check if user has consent for purpose
        has_consent = self.check_consent(user_id, purpose, partner)

        if not has_consent:
            return {
                "permitted": False,
                "reason": "no_active_consent",
                "required_consent_type": self._get_required_consent_type(purpose),
                "data_elements": data_elements,
            }

        # Get relevant consents
        consents = self.get_active_consents(user_id)
        applicable_consents = [c for c in consents if purpose in c.purposes]

        # Check specific permissions for data elements
        permitted_elements = []
        denied_elements = []

        for element in data_elements:
            element_permitted = False
            for consent in applicable_consents:
                if consent.specific_permissions.get(element, False):
                    element_permitted = True
                    break

            if element_permitted:
                permitted_elements.append(element)
            else:
                denied_elements.append(element)

        return {
            "permitted": len(denied_elements) == 0,
            "permitted_elements": permitted_elements,
            "denied_elements": denied_elements,
            "applicable_consents": [c.consent_id for c in applicable_consents],
            "partner_approved": partner is None
            or any(c.allows_data_sharing(partner) for c in applicable_consents),
        }

    def _get_required_consent_type(self, purpose: ConsentPurpose) -> ConsentType:
        """Get required consent type for purpose."""
        clinical_purposes = {
            ConsentPurpose.DIAGNOSIS,
            ConsentPurpose.TREATMENT,
            ConsentPurpose.CARE_COORDINATION,
        }

        research_purposes = {
            ConsentPurpose.DRUG_DISCOVERY,
            ConsentPurpose.POPULATION_STUDIES,
            ConsentPurpose.RARE_DISEASE,
            ConsentPurpose.PHARMACOGENOMICS,
        }

        if purpose in clinical_purposes:
            return ConsentType.CLINICAL
        elif purpose in research_purposes:
            return ConsentType.RESEARCH
        else:
            return ConsentType.COMMERCIAL
