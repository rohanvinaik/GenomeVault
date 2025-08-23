"""
Breach detection and notification system for GenomeVault.

Implements automated detection of potential privacy breaches and
manages notification workflows for HIPAA and GDPR compliance:

- HIPAA: 60 days to individuals, 60 days to HHS, media if >500 individuals
- GDPR: 72 hours to supervisory authority, without undue delay to individuals
"""

import json
import logging
import smtplib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import requests
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Metrics
breach_detections = Counter(
    'genomevault_breach_detections_total',
    'Total breach detections',
    ['breach_type', 'severity', 'source']
)

breach_notifications = Counter(
    'genomevault_breach_notifications_total',
    'Total breach notifications sent',
    ['notification_type', 'channel']
)

breach_response_time = Histogram(
    'genomevault_breach_response_seconds',
    'Time from detection to initial response',
    ['breach_type'],
    buckets=(60, 300, 900, 1800, 3600, 7200, 14400)  # 1min to 4h
)

active_breaches = Gauge(
    'genomevault_active_breaches',
    'Number of active breaches under investigation',
    ['severity']
)


class BreachType(Enum):
    """Types of potential breaches."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    INSIDER_THREAT = "insider_threat"
    MALWARE_INFECTION = "malware_infection"
    SYSTEM_INTRUSION = "system_intrusion"
    PHI_EXPOSURE = "phi_exposure"
    RANSOMWARE = "ransomware"
    PHYSICAL_THEFT = "physical_theft"
    CONFIGURATION_ERROR = "configuration_error"
    VENDOR_BREACH = "vendor_breach"


class BreachSeverity(Enum):
    """Breach severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BreachStatus(Enum):
    """Breach investigation status."""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    CONTAINED = "contained"
    RESOLVED = "resolved"


class NotificationType(Enum):
    """Types of breach notifications."""
    HIPAA_INDIVIDUAL = "hipaa_individual"
    HIPAA_HHS = "hipaa_hhs"
    HIPAA_MEDIA = "hipaa_media"
    GDPR_AUTHORITY = "gdpr_authority"
    GDPR_INDIVIDUAL = "gdpr_individual"
    INTERNAL_ALERT = "internal_alert"
    VENDOR_NOTIFICATION = "vendor_notification"


@dataclass
class BreachIndicator:
    """Single breach indicator/IOC."""
    indicator_type: str  # ip, domain, hash, pattern, etc.
    value: str
    confidence: float  # 0.0 - 1.0
    source: str
    first_seen: datetime
    description: str
    
    def matches(self, event_data: Dict[str, Any]) -> bool:
        """Check if this indicator matches event data."""
        # Simplified matching logic
        if self.indicator_type == "ip":
            return event_data.get("ip_address") == self.value
        elif self.indicator_type == "user":
            return event_data.get("user_id") == self.value
        elif self.indicator_type == "pattern":
            # Pattern matching logic would go here
            return False
        return False


@dataclass
class BreachEvent:
    """Detected breach event."""
    event_id: str
    breach_type: BreachType
    severity: BreachSeverity
    status: BreachStatus
    detected_at: datetime
    source: str  # system that detected the breach
    confidence: float  # 0.0 - 1.0
    
    # Affected data
    phi_involved: bool = False
    affected_individuals: Set[str] = field(default_factory=set)
    data_elements: List[str] = field(default_factory=list)
    estimated_records: int = 0
    
    # Technical details
    attack_vector: Optional[str] = None
    indicators: List[BreachIndicator] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    
    # Response tracking
    response_started_at: Optional[datetime] = None
    contained_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Notifications
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    
    # Investigation details
    investigation_notes: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    
    def add_investigation_note(self, note: str) -> None:
        """Add investigation note."""
        timestamp = datetime.utcnow().isoformat()
        self.investigation_notes.append(f"[{timestamp}] {note}")
    
    def add_evidence(self, evidence_path: str) -> None:
        """Add evidence reference."""
        self.evidence.append(evidence_path)
    
    def is_major_breach(self) -> bool:
        """Check if this qualifies as a major breach."""
        # HIPAA: 500+ individuals
        # GDPR: High risk to rights and freedoms
        return (
            self.estimated_records >= 500 or
            self.severity in [BreachSeverity.HIGH, BreachSeverity.CRITICAL] or
            self.breach_type in [BreachType.DATA_EXFILTRATION, BreachType.RANSOMWARE]
        )
    
    def requires_immediate_notification(self) -> bool:
        """Check if immediate notification is required."""
        return (
            self.severity == BreachSeverity.CRITICAL or
            self.breach_type in [BreachType.RANSOMWARE, BreachType.DATA_EXFILTRATION] or
            self.estimated_records >= 1000
        )


class BreachDetector:
    """Automated breach detection system."""
    
    def __init__(self):
        self.indicators: List[BreachIndicator] = []
        self.detection_rules: List[Dict[str, Any]] = []
        self.baseline_metrics: Dict[str, float] = {}
        
        # Load default indicators and rules
        self._load_default_indicators()
        self._load_detection_rules()
    
    def _load_default_indicators(self) -> None:
        """Load default threat indicators."""
        default_indicators = [
            BreachIndicator(
                indicator_type="pattern",
                value="multiple_failed_logins",
                confidence=0.7,
                source="security_team",
                first_seen=datetime.utcnow(),
                description="Multiple failed login attempts from single IP"
            ),
            BreachIndicator(
                indicator_type="pattern", 
                value="bulk_phi_access",
                confidence=0.9,
                source="audit_system",
                first_seen=datetime.utcnow(),
                description="Unusual bulk access to PHI records"
            ),
            BreachIndicator(
                indicator_type="pattern",
                value="off_hours_admin_access",
                confidence=0.6,
                source="access_monitor",
                first_seen=datetime.utcnow(),
                description="Administrative access during off hours"
            )
        ]
        
        self.indicators.extend(default_indicators)
    
    def _load_detection_rules(self) -> None:
        """Load breach detection rules."""
        self.detection_rules = [
            {
                "name": "failed_login_threshold",
                "condition": "failed_logins > 10 in 5 minutes",
                "breach_type": BreachType.UNAUTHORIZED_ACCESS,
                "severity": BreachSeverity.MEDIUM,
                "confidence": 0.8
            },
            {
                "name": "bulk_data_access",
                "condition": "phi_records_accessed > 100 in 1 hour by single user",
                "breach_type": BreachType.INSIDER_THREAT,
                "severity": BreachSeverity.HIGH,
                "confidence": 0.9
            },
            {
                "name": "unusual_data_export",
                "condition": "data_export_size > 1GB AND time = off_hours",
                "breach_type": BreachType.DATA_EXFILTRATION,
                "severity": BreachSeverity.CRITICAL,
                "confidence": 0.95
            },
            {
                "name": "encryption_failure",
                "condition": "unencrypted_phi_detected = true",
                "breach_type": BreachType.PHI_EXPOSURE,
                "severity": BreachSeverity.CRITICAL,
                "confidence": 1.0
            }
        ]
    
    def analyze_event(self, event_data: Dict[str, Any]) -> Optional[BreachEvent]:
        """Analyze event for potential breach indicators."""
        # Check against known indicators
        matched_indicators = []
        for indicator in self.indicators:
            if indicator.matches(event_data):
                matched_indicators.append(indicator)
        
        # Check against detection rules
        for rule in self.detection_rules:
            if self._rule_matches(rule, event_data):
                # Create breach event
                breach = BreachEvent(
                    event_id=str(uuid4()),
                    breach_type=rule["breach_type"],
                    severity=rule["severity"],
                    status=BreachStatus.DETECTED,
                    detected_at=datetime.utcnow(),
                    source="automated_detection",
                    confidence=rule["confidence"],
                    indicators=matched_indicators,
                    phi_involved=event_data.get("phi_involved", False),
                    affected_systems=[event_data.get("system", "unknown")]
                )
                
                # Estimate affected records
                breach.estimated_records = self._estimate_affected_records(event_data, rule)
                
                return breach
        
        return None
    
    def _rule_matches(self, rule: Dict[str, Any], event_data: Dict[str, Any]) -> bool:
        """Check if detection rule matches event data."""
        # Simplified rule matching - would implement proper rule engine
        condition = rule["condition"]
        
        if "failed_logins > 10" in condition:
            return event_data.get("failed_login_count", 0) > 10
        
        if "phi_records_accessed > 100" in condition:
            return event_data.get("records_accessed", 0) > 100
        
        if "data_export_size > 1GB" in condition:
            return event_data.get("export_size_bytes", 0) > 1024*1024*1024
        
        if "unencrypted_phi_detected = true" in condition:
            return event_data.get("unencrypted_phi_detected", False)
        
        return False
    
    def _estimate_affected_records(self, event_data: Dict[str, Any], rule: Dict[str, Any]) -> int:
        """Estimate number of affected records."""
        # Simple estimation logic
        if "bulk_data_access" in rule["name"]:
            return event_data.get("records_accessed", 0)
        elif "data_export" in rule["name"]:
            # Estimate based on file size (rough approximation)
            size_bytes = event_data.get("export_size_bytes", 0)
            return max(1, size_bytes // (10 * 1024))  # Assume ~10KB per record
        else:
            return event_data.get("estimated_records", 1)
    
    def add_indicator(self, indicator: BreachIndicator) -> None:
        """Add new threat indicator."""
        self.indicators.append(indicator)
        logger.info(f"Added breach indicator: {indicator.value}")


class RiskAssessment:
    """HIPAA breach risk assessment calculator."""
    
    RISK_FACTORS = {
        # Nature and extent of PHI
        "phi_sensitivity": {
            "genetic_data": 0.9,
            "clinical_diagnoses": 0.8,
            "mental_health": 0.8,
            "substance_abuse": 0.8,
            "demographic": 0.3
        },
        
        # Who accessed/received PHI
        "accessor_risk": {
            "unauthorized_external": 1.0,
            "unauthorized_internal": 0.8,
            "business_associate": 0.4,
            "covered_entity": 0.2
        },
        
        # Was PHI actually viewed/acquired
        "acquisition_certainty": {
            "confirmed_access": 1.0,
            "likely_access": 0.7,
            "possible_access": 0.4,
            "no_evidence": 0.1
        },
        
        # Extent of mitigation
        "mitigation_effectiveness": {
            "no_mitigation": 1.0,
            "partial_mitigation": 0.6,
            "strong_mitigation": 0.3,
            "complete_mitigation": 0.1
        }
    }
    
    def assess_breach_risk(self, breach: BreachEvent) -> Dict[str, Any]:
        """Perform HIPAA breach risk assessment."""
        
        # Calculate risk scores for each factor
        sensitivity_score = self._calculate_sensitivity_score(breach.data_elements)
        accessor_score = self._calculate_accessor_score(breach.breach_type)
        acquisition_score = self._calculate_acquisition_score(breach.confidence)
        mitigation_score = self._calculate_mitigation_score(breach.status)
        
        # Overall risk score (weighted average)
        overall_risk = (
            sensitivity_score * 0.3 +
            accessor_score * 0.3 +
            acquisition_score * 0.2 +
            mitigation_score * 0.2
        )
        
        # Determine if notification is required
        notification_required = overall_risk >= 0.5 or breach.estimated_records >= 500
        
        # Risk level classification
        if overall_risk >= 0.8:
            risk_level = "HIGH"
        elif overall_risk >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "overall_risk_score": round(overall_risk, 2),
            "risk_level": risk_level,
            "notification_required": notification_required,
            "factor_scores": {
                "phi_sensitivity": round(sensitivity_score, 2),
                "accessor_risk": round(accessor_score, 2),
                "acquisition_certainty": round(acquisition_score, 2),
                "mitigation_effectiveness": round(mitigation_score, 2)
            },
            "affected_individuals": len(breach.affected_individuals),
            "estimated_records": breach.estimated_records,
            "assessment_date": datetime.utcnow().isoformat()
        }
    
    def _calculate_sensitivity_score(self, data_elements: List[str]) -> float:
        """Calculate PHI sensitivity score."""
        if not data_elements:
            return 0.5  # Default moderate sensitivity
        
        max_sensitivity = 0.0
        for element in data_elements:
            element_lower = element.lower()
            for category, score in self.RISK_FACTORS["phi_sensitivity"].items():
                if category in element_lower:
                    max_sensitivity = max(max_sensitivity, score)
        
        return max_sensitivity or 0.5
    
    def _calculate_accessor_score(self, breach_type: BreachType) -> float:
        """Calculate accessor risk score."""
        external_threats = {
            BreachType.SYSTEM_INTRUSION,
            BreachType.MALWARE_INFECTION,
            BreachType.RANSOMWARE
        }
        
        if breach_type in external_threats:
            return self.RISK_FACTORS["accessor_risk"]["unauthorized_external"]
        elif breach_type == BreachType.INSIDER_THREAT:
            return self.RISK_FACTORS["accessor_risk"]["unauthorized_internal"]
        else:
            return self.RISK_FACTORS["accessor_risk"]["business_associate"]
    
    def _calculate_acquisition_score(self, confidence: float) -> float:
        """Calculate acquisition certainty score."""
        if confidence >= 0.9:
            return self.RISK_FACTORS["acquisition_certainty"]["confirmed_access"]
        elif confidence >= 0.7:
            return self.RISK_FACTORS["acquisition_certainty"]["likely_access"]
        elif confidence >= 0.4:
            return self.RISK_FACTORS["acquisition_certainty"]["possible_access"]
        else:
            return self.RISK_FACTORS["acquisition_certainty"]["no_evidence"]
    
    def _calculate_mitigation_score(self, status: BreachStatus) -> float:
        """Calculate mitigation effectiveness score."""
        if status == BreachStatus.RESOLVED:
            return self.RISK_FACTORS["mitigation_effectiveness"]["complete_mitigation"]
        elif status == BreachStatus.CONTAINED:
            return self.RISK_FACTORS["mitigation_effectiveness"]["strong_mitigation"]
        elif status == BreachStatus.INVESTIGATING:
            return self.RISK_FACTORS["mitigation_effectiveness"]["partial_mitigation"]
        else:
            return self.RISK_FACTORS["mitigation_effectiveness"]["no_mitigation"]


class NotificationManager:
    """Manages breach notifications."""
    
    def __init__(self, 
                 smtp_server: str = "smtp.genomevault.io",
                 smtp_port: int = 587,
                 from_email: str = "privacy@genomevault.io"):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.from_email = from_email
        
        # Notification templates
        self.templates = self._load_notification_templates()
        
        # Regulatory contacts
        self.regulatory_contacts = {
            "hipaa_hhs": {
                "email": "OCRComplaint@hhs.gov",
                "portal": "https://ocrportal.hhs.gov/ocr/breach/wizard_breach.jsf"
            },
            "gdpr_authorities": {
                "US": {"email": "privacy@ftc.gov"},
                "EU": {"email": "info@edpb.europa.eu"},
                "UK": {"email": "casework@ico.org.uk"}
            }
        }
    
    def _load_notification_templates(self) -> Dict[str, str]:
        """Load notification email templates."""
        return {
            "hipaa_individual": """
Subject: Important Notice: Privacy Incident Affecting Your Information

Dear Patient,

We are writing to inform you of an incident that may have involved some of your protected health information (PHI). We take the privacy and security of your information very seriously and want to provide you with information about the incident and the steps we are taking.

What Happened:
{incident_description}

What Information Was Involved:
{affected_data}

What We Are Doing:
{remediation_steps}

What You Can Do:
{patient_actions}

For More Information:
If you have questions, please contact our Privacy Officer at privacy@genomevault.io or call {contact_phone}.

Sincerely,
GenomeVault Privacy Office
            """,
            
            "gdpr_authority": """
Subject: GDPR Data Breach Notification - {breach_id}

Dear Data Protection Authority,

We are reporting a personal data breach in accordance with Article 33 of the GDPR.

Breach Details:
- Breach Reference: {breach_id}
- Date/Time of Breach: {breach_datetime}
- Discovery Date/Time: {discovery_datetime}
- Categories of Data Subjects: {data_subject_categories}
- Approximate Number Affected: {affected_count}
- Categories of Data: {data_categories}

Description of Breach:
{breach_description}

Consequences and Risk Assessment:
{risk_assessment}

Measures Taken:
{measures_taken}

Contact Information:
Data Protection Officer: {dpo_contact}
Organization: GenomeVault, Inc.

Regards,
GenomeVault Data Protection Team
            """,
            
            "internal_alert": """
Subject: SECURITY ALERT - Potential Breach Detected [{severity}]

Security Team,

A potential security breach has been detected:

Breach ID: {breach_id}
Type: {breach_type}
Severity: {severity}
Confidence: {confidence}%
Detection Time: {detection_time}

Affected Systems: {affected_systems}
PHI Involved: {phi_involved}
Estimated Records: {estimated_records}

Immediate Actions Required:
1. Investigate and confirm the incident
2. Contain any ongoing threat
3. Preserve evidence
4. Assess impact and risk
5. Prepare notifications if confirmed

Incident Response Plan: https://wiki.genomevault.io/incident-response

This is an automated alert from GenomeVault Breach Detection System.
            """
        }
    
    def send_notification(self,
                         notification_type: NotificationType,
                         breach: BreachEvent,
                         recipients: List[str],
                         **kwargs) -> bool:
        """Send breach notification."""
        
        try:
            # Get template
            template = self._get_template(notification_type, breach, **kwargs)
            
            # Send via appropriate channel
            if notification_type in [NotificationType.HIPAA_INDIVIDUAL, 
                                   NotificationType.GDPR_INDIVIDUAL,
                                   NotificationType.INTERNAL_ALERT]:
                success = self._send_email(template["subject"], template["body"], recipients)
            elif notification_type in [NotificationType.HIPAA_HHS,
                                     NotificationType.GDPR_AUTHORITY]:
                success = self._send_regulatory_notification(notification_type, template, **kwargs)
            elif notification_type == NotificationType.HIPAA_MEDIA:
                success = self._send_media_notification(template, **kwargs)
            else:
                success = self._send_email(template["subject"], template["body"], recipients)
            
            if success:
                # Record notification
                notification_record = {
                    "type": notification_type.value,
                    "sent_at": datetime.utcnow().isoformat(),
                    "recipients": recipients,
                    "success": True
                }
                breach.notifications_sent.append(notification_record)
                
                breach_notifications.labels(
                    notification_type=notification_type.value,
                    channel="email"
                ).inc()
                
                logger.info(f"Sent {notification_type.value} notification for breach {breach.event_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send {notification_type.value} notification: {str(e)}")
            return False
    
    def _get_template(self, 
                     notification_type: NotificationType,
                     breach: BreachEvent,
                     **kwargs) -> Dict[str, str]:
        """Get formatted notification template."""
        
        template_key = {
            NotificationType.HIPAA_INDIVIDUAL: "hipaa_individual",
            NotificationType.GDPR_INDIVIDUAL: "hipaa_individual",  # Same template
            NotificationType.GDPR_AUTHORITY: "gdpr_authority",
            NotificationType.INTERNAL_ALERT: "internal_alert"
        }.get(notification_type, "internal_alert")
        
        template = self.templates[template_key]
        
        # Format template variables
        formatted = template.format(
            breach_id=breach.event_id,
            breach_type=breach.breach_type.value,
            severity=breach.severity.value,
            confidence=int(breach.confidence * 100),
            detection_time=breach.detected_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            breach_datetime=breach.detected_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            discovery_datetime=breach.detected_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            affected_systems=", ".join(breach.affected_systems),
            phi_involved="Yes" if breach.phi_involved else "No",
            estimated_records=breach.estimated_records,
            affected_count=len(breach.affected_individuals),
            data_subject_categories="Healthcare patients",
            data_categories=", ".join(breach.data_elements),
            breach_description=kwargs.get("breach_description", "Automated breach detection"),
            risk_assessment=kwargs.get("risk_assessment", "Assessment pending"),
            measures_taken=kwargs.get("measures_taken", "Investigation ongoing"),
            incident_description=kwargs.get("incident_description", "Security incident detected"),
            affected_data=", ".join(breach.data_elements),
            remediation_steps=kwargs.get("remediation_steps", "Investigation and remediation in progress"),
            patient_actions=kwargs.get("patient_actions", "Monitor accounts for unusual activity"),
            contact_phone=kwargs.get("contact_phone", "1-800-PRIVACY"),
            dpo_contact="privacy@genomevault.io"
        )
        
        # Extract subject line
        lines = formatted.strip().split('\n')
        subject = lines[0].replace("Subject: ", "") if lines[0].startswith("Subject: ") else "Breach Notification"
        body = '\n'.join(lines[1:])
        
        return {"subject": subject, "body": body}
    
    def _send_email(self, subject: str, body: str, recipients: List[str]) -> bool:
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Would implement actual SMTP sending
            logger.info(f"Email notification sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def _send_regulatory_notification(self,
                                    notification_type: NotificationType,
                                    template: Dict[str, str],
                                    **kwargs) -> bool:
        """Send notification to regulatory authorities."""
        
        if notification_type == NotificationType.HIPAA_HHS:
            # Submit to HHS OCR portal
            return self._submit_to_hhs_portal(template, **kwargs)
        elif notification_type == NotificationType.GDPR_AUTHORITY:
            # Submit to relevant GDPR authority
            return self._submit_to_gdpr_authority(template, **kwargs)
        
        return False
    
    def _submit_to_hhs_portal(self, template: Dict[str, str], **kwargs) -> bool:
        """Submit breach notification to HHS OCR portal."""
        # Would implement actual OCR portal submission
        logger.info("Submitted breach notification to HHS OCR")
        return True
    
    def _submit_to_gdpr_authority(self, template: Dict[str, str], **kwargs) -> bool:
        """Submit breach notification to GDPR supervisory authority."""
        # Would implement actual authority submission
        logger.info("Submitted breach notification to GDPR authority")
        return True
    
    def _send_media_notification(self, template: Dict[str, str], **kwargs) -> bool:
        """Send media notification for large breaches."""
        # Would implement media notification process
        logger.info("Media notification prepared")
        return True


class BreachNotificationSystem:
    """Main breach notification coordination system."""
    
    def __init__(self):
        self.detector = BreachDetector()
        self.risk_assessor = RiskAssessment()
        self.notification_manager = NotificationManager()
        
        # Active breaches
        self.active_breaches: Dict[str, BreachEvent] = {}
        self.resolved_breaches: Dict[str, BreachEvent] = {}
        
        # Notification deadlines
        self.notification_deadlines = {
            NotificationType.GDPR_AUTHORITY: timedelta(hours=72),
            NotificationType.GDPR_INDIVIDUAL: timedelta(hours=72),  # "without undue delay"
            NotificationType.HIPAA_INDIVIDUAL: timedelta(days=60),
            NotificationType.HIPAA_HHS: timedelta(days=60),
            NotificationType.HIPAA_MEDIA: timedelta(days=60),
            NotificationType.INTERNAL_ALERT: timedelta(minutes=30)
        }
    
    def detect_breach(self, event_data: Dict[str, Any]) -> Optional[str]:
        """Detect potential breach from event data."""
        
        breach = self.detector.analyze_event(event_data)
        
        if breach:
            self.active_breaches[breach.event_id] = breach
            
            # Update metrics
            breach_detections.labels(
                breach_type=breach.breach_type.value,
                severity=breach.severity.value,
                source=breach.source
            ).inc()
            
            active_breaches.labels(severity=breach.severity.value).inc()
            
            # Start response timer
            breach.response_started_at = datetime.utcnow()
            
            # Send immediate internal alert
            self._send_immediate_alerts(breach)
            
            # Start investigation workflow
            self._start_investigation(breach)
            
            logger.warning(f"Breach detected: {breach.event_id} - {breach.breach_type.value}")
            
            return breach.event_id
        
        return None
    
    def _send_immediate_alerts(self, breach: BreachEvent) -> None:
        """Send immediate internal alerts."""
        
        # Always send internal alert
        recipients = ["security@genomevault.io", "privacy@genomevault.io"]
        
        if breach.severity in [BreachSeverity.HIGH, BreachSeverity.CRITICAL]:
            recipients.extend(["ciso@genomevault.io", "ceo@genomevault.io"])
        
        self.notification_manager.send_notification(
            NotificationType.INTERNAL_ALERT,
            breach,
            recipients
        )
    
    def _start_investigation(self, breach: BreachEvent) -> None:
        """Start breach investigation workflow."""
        breach.status = BreachStatus.INVESTIGATING
        breach.add_investigation_note("Investigation started by automated system")
        
        # Schedule follow-up actions
        self._schedule_notification_deadlines(breach)
    
    def _schedule_notification_deadlines(self, breach: BreachEvent) -> None:
        """Schedule notification deadline reminders."""
        
        for notification_type, deadline in self.notification_deadlines.items():
            deadline_time = breach.detected_at + deadline
            
            # Would schedule actual reminders/tasks
            logger.info(f"Scheduled {notification_type.value} deadline: {deadline_time}")
    
    def confirm_breach(self, 
                      event_id: str,
                      affected_individuals: Set[str],
                      data_elements: List[str],
                      root_cause: str) -> bool:
        """Confirm breach and trigger notifications."""
        
        if event_id not in self.active_breaches:
            return False
        
        breach = self.active_breaches[event_id]
        breach.status = BreachStatus.CONFIRMED
        breach.affected_individuals = affected_individuals
        breach.data_elements = data_elements
        breach.root_cause = root_cause
        breach.estimated_records = len(affected_individuals)
        
        breach.add_investigation_note(f"Breach confirmed. Root cause: {root_cause}")
        
        # Perform risk assessment
        risk_assessment = self.risk_assessor.assess_breach_risk(breach)
        
        # Send required notifications based on risk assessment
        if risk_assessment["notification_required"]:
            self._send_required_notifications(breach, risk_assessment)
        
        return True
    
    def _send_required_notifications(self, 
                                   breach: BreachEvent,
                                   risk_assessment: Dict[str, Any]) -> None:
        """Send all required notifications."""
        
        # GDPR notifications (72 hours)
        if breach.phi_involved:
            # Authority notification
            self.notification_manager.send_notification(
                NotificationType.GDPR_AUTHORITY,
                breach,
                ["privacy@genomevault.io"],  # Will route to appropriate authority
                risk_assessment=json.dumps(risk_assessment, indent=2)
            )
            
            # Individual notifications if high risk
            if risk_assessment["risk_level"] in ["HIGH", "MEDIUM"]:
                individual_emails = self._get_individual_contact_info(breach.affected_individuals)
                self.notification_manager.send_notification(
                    NotificationType.GDPR_INDIVIDUAL,
                    breach,
                    individual_emails
                )
        
        # HIPAA notifications (60 days, but send earlier)
        if breach.phi_involved and risk_assessment["notification_required"]:
            # HHS notification
            self.notification_manager.send_notification(
                NotificationType.HIPAA_HHS,
                breach,
                ["ocr@hhs.gov"]
            )
            
            # Individual notifications
            individual_emails = self._get_individual_contact_info(breach.affected_individuals)
            self.notification_manager.send_notification(
                NotificationType.HIPAA_INDIVIDUAL,
                breach,
                individual_emails
            )
            
            # Media notification if major breach (500+)
            if breach.is_major_breach():
                self.notification_manager.send_notification(
                    NotificationType.HIPAA_MEDIA,
                    breach,
                    ["media@genomevault.io"]
                )
    
    def _get_individual_contact_info(self, individual_ids: Set[str]) -> List[str]:
        """Get contact information for affected individuals."""
        # Would query database for contact info
        return [f"patient{i}@example.com" for i in list(individual_ids)[:10]]  # Mock data
    
    def mark_false_positive(self, event_id: str, reason: str) -> bool:
        """Mark breach detection as false positive."""
        
        if event_id not in self.active_breaches:
            return False
        
        breach = self.active_breaches[event_id]
        breach.status = BreachStatus.FALSE_POSITIVE
        breach.add_investigation_note(f"Marked as false positive: {reason}")
        
        # Move to resolved
        self.resolved_breaches[event_id] = breach
        del self.active_breaches[event_id]
        
        active_breaches.labels(severity=breach.severity.value).dec()
        
        return True
    
    def contain_breach(self, event_id: str, containment_actions: List[str]) -> bool:
        """Mark breach as contained."""
        
        if event_id not in self.active_breaches:
            return False
        
        breach = self.active_breaches[event_id]
        breach.status = BreachStatus.CONTAINED
        breach.contained_at = datetime.utcnow()
        
        for action in containment_actions:
            breach.add_investigation_note(f"Containment action: {action}")
        
        return True
    
    def resolve_breach(self, event_id: str, resolution_summary: str) -> bool:
        """Mark breach as resolved."""
        
        if event_id not in self.active_breaches:
            return False
        
        breach = self.active_breaches[event_id]
        breach.status = BreachStatus.RESOLVED
        breach.resolved_at = datetime.utcnow()
        breach.add_investigation_note(f"Resolution: {resolution_summary}")
        
        # Calculate response time
        if breach.response_started_at:
            response_time = (breach.resolved_at - breach.response_started_at).total_seconds()
            breach_response_time.labels(breach_type=breach.breach_type.value).observe(response_time)
        
        # Move to resolved
        self.resolved_breaches[event_id] = breach
        del self.active_breaches[event_id]
        
        active_breaches.labels(severity=breach.severity.value).dec()
        
        return True
    
    def get_breach_status(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get current breach status."""
        
        breach = self.active_breaches.get(event_id) or self.resolved_breaches.get(event_id)
        
        if not breach:
            return None
        
        return {
            "event_id": breach.event_id,
            "breach_type": breach.breach_type.value,
            "severity": breach.severity.value,
            "status": breach.status.value,
            "detected_at": breach.detected_at.isoformat(),
            "phi_involved": breach.phi_involved,
            "affected_individuals": len(breach.affected_individuals),
            "estimated_records": breach.estimated_records,
            "notifications_sent": len(breach.notifications_sent),
            "investigation_notes": breach.investigation_notes[-5:],  # Last 5 notes
            "response_time": self._calculate_response_time(breach)
        }
    
    def _calculate_response_time(self, breach: BreachEvent) -> Optional[str]:
        """Calculate breach response time."""
        if breach.response_started_at:
            end_time = breach.resolved_at or datetime.utcnow()
            delta = end_time - breach.response_started_at
            return f"{delta.total_seconds():.0f} seconds"
        return None
    
    def get_notification_requirements(self, event_id: str) -> Dict[str, Any]:
        """Get notification requirements for breach."""
        
        breach = self.active_breaches.get(event_id)
        if not breach:
            return {}
        
        risk_assessment = self.risk_assessor.assess_breach_risk(breach)
        
        requirements = {
            "gdpr_notifications": [],
            "hipaa_notifications": [],
            "internal_notifications": ["security@genomevault.io"]
        }
        
        # GDPR requirements
        if breach.phi_involved:
            requirements["gdpr_notifications"].append({
                "type": "authority_notification",
                "deadline": (breach.detected_at + timedelta(hours=72)).isoformat(),
                "required": True
            })
            
            if risk_assessment["risk_level"] in ["HIGH", "MEDIUM"]:
                requirements["gdpr_notifications"].append({
                    "type": "individual_notification", 
                    "deadline": (breach.detected_at + timedelta(hours=72)).isoformat(),
                    "required": True
                })
        
        # HIPAA requirements
        if risk_assessment["notification_required"]:
            requirements["hipaa_notifications"].extend([
                {
                    "type": "hhs_notification",
                    "deadline": (breach.detected_at + timedelta(days=60)).isoformat(),
                    "required": True
                },
                {
                    "type": "individual_notification",
                    "deadline": (breach.detected_at + timedelta(days=60)).isoformat(), 
                    "required": True
                }
            ])
            
            if breach.is_major_breach():
                requirements["hipaa_notifications"].append({
                    "type": "media_notification",
                    "deadline": (breach.detected_at + timedelta(days=60)).isoformat(),
                    "required": True
                })
        
        return requirements