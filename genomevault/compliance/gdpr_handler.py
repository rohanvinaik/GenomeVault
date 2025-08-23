"""
GDPR compliance handler for GenomeVault.

Implements data subject rights under GDPR Articles 15-22:
- Right to access (Article 15)
- Right to rectification (Article 16)  
- Right to erasure/right to be forgotten (Article 17)
- Right to restriction of processing (Article 18)
- Right to data portability (Article 20)
- Right to object (Article 21)
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
gdpr_requests = Counter(
    'genomevault_gdpr_requests_total',
    'Total GDPR data subject requests',
    ['request_type', 'status']
)

gdpr_request_duration = Histogram(
    'genomevault_gdpr_request_duration_seconds',
    'GDPR request processing duration',
    ['request_type'],
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600)
)

data_exports = Counter(
    'genomevault_data_exports_total',
    'Total data exports for portability',
    ['format']
)


class RequestType(Enum):
    """GDPR request types."""
    ACCESS = "access"  # Article 15
    RECTIFICATION = "rectification"  # Article 16
    ERASURE = "erasure"  # Article 17
    RESTRICTION = "restriction"  # Article 18
    PORTABILITY = "portability"  # Article 20
    OBJECTION = "objection"  # Article 21


class RequestStatus(Enum):
    """Request processing status."""
    PENDING = "pending"
    VERIFIED = "verified"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class DataFormat(Enum):
    """Export data formats."""
    JSON = "json"
    VCF = "vcf"
    FHIR = "fhir"
    CSV = "csv"
    XML = "xml"


@dataclass
class DataSubjectRequest:
    """Data subject request under GDPR."""
    request_id: str
    user_id: str
    request_type: RequestType
    status: RequestStatus
    created_at: datetime
    verified_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    verification_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_audit_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Add entry to audit log."""
        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "details": details
        })
    
    def is_expired(self) -> bool:
        """Check if request has expired (30 days)."""
        return datetime.utcnow() - self.created_at > timedelta(days=30)


class CryptographicErasure:
    """Implement cryptographic erasure for right to be forgotten."""
    
    def __init__(self, key_storage_path: str = "/secure/keys"):
        self.key_storage_path = Path(key_storage_path)
        self.key_storage_path.mkdir(parents=True, exist_ok=True)
    
    def generate_user_key(self, user_id: str) -> bytes:
        """Generate unique encryption key for user data."""
        salt = os.urandom(16)
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(user_id.encode())
        
        # Store key securely
        key_path = self.key_storage_path / f"{user_id}.key"
        with open(key_path, 'wb') as f:
            f.write(salt + key)
        
        return key
    
    def erase_user_data(self, user_id: str) -> bool:
        """Cryptographically erase user data by destroying key."""
        key_path = self.key_storage_path / f"{user_id}.key"
        
        if not key_path.exists():
            logger.warning(f"Key not found for user {user_id}")
            return False
        
        # Overwrite key multiple times before deletion
        key_size = key_path.stat().st_size
        with open(key_path, 'wb') as f:
            # DoD 5220.22-M standard: 3 passes
            for _ in range(3):
                f.write(os.urandom(key_size))
                f.flush()
                os.fsync(f.fileno())
        
        # Remove the file
        key_path.unlink()
        
        # Verify deletion
        if key_path.exists():
            raise RuntimeError(f"Failed to delete key for user {user_id}")
        
        logger.info(f"Cryptographically erased data for user {user_id}")
        return True


class DataPortability:
    """Handle data portability requests (Article 20)."""
    
    def __init__(self):
        self.supported_formats = {
            DataFormat.JSON: self._export_json,
            DataFormat.VCF: self._export_vcf,
            DataFormat.FHIR: self._export_fhir,
            DataFormat.CSV: self._export_csv,
            DataFormat.XML: self._export_xml
        }
    
    def export_user_data(self, 
                        user_id: str,
                        format: DataFormat,
                        include_derived: bool = False) -> Path:
        """Export user data in requested format."""
        with gdpr_request_duration.labels(request_type="portability").time():
            # Collect user data
            user_data = self._collect_user_data(user_id, include_derived)
            
            # Export in requested format
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}")
            
            export_path = self.supported_formats[format](user_data, user_id)
            
            data_exports.labels(format=format.value).inc()
            
            return export_path
    
    def _collect_user_data(self, user_id: str, include_derived: bool) -> Dict[str, Any]:
        """Collect all user data for export."""
        data = {
            "user_id": user_id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "genomic_data": self._get_genomic_data(user_id),
            "clinical_data": self._get_clinical_data(user_id),
            "consent_records": self._get_consent_records(user_id),
            "access_logs": self._get_access_logs(user_id)
        }
        
        if include_derived:
            data["derived_insights"] = self._get_derived_insights(user_id)
            data["hypervectors"] = self._get_hypervectors(user_id)
        
        return data
    
    def _export_json(self, data: Dict[str, Any], user_id: str) -> Path:
        """Export data as JSON."""
        export_dir = Path(tempfile.mkdtemp())
        export_path = export_dir / f"gdpr_export_{user_id}_{datetime.utcnow().strftime('%Y%m%d')}.json"
        
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return export_path
    
    def _export_vcf(self, data: Dict[str, Any], user_id: str) -> Path:
        """Export genomic data as VCF."""
        export_dir = Path(tempfile.mkdtemp())
        export_path = export_dir / f"genomic_data_{user_id}.vcf"
        
        with open(export_path, 'w') as f:
            # VCF header
            f.write("##fileformat=VCFv4.3\n")
            f.write(f"##fileDate={datetime.utcnow().strftime('%Y%m%d')}\n")
            f.write("##source=GenomeVault\n")
            f.write("##reference=GRCh38\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
            
            # Write variants
            genomic_data = data.get("genomic_data", {})
            for variant in genomic_data.get("variants", []):
                f.write(f"{variant['chrom']}\t{variant['pos']}\t{variant['id']}\t")
                f.write(f"{variant['ref']}\t{variant['alt']}\t{variant['qual']}\t")
                f.write(f"{variant['filter']}\t{variant['info']}\tGT\t{variant['genotype']}\n")
        
        return export_path
    
    def _export_fhir(self, data: Dict[str, Any], user_id: str) -> Path:
        """Export data as FHIR bundle."""
        export_dir = Path(tempfile.mkdtemp())
        export_path = export_dir / f"fhir_bundle_{user_id}.json"
        
        # Create FHIR bundle
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": datetime.utcnow().isoformat(),
            "entry": []
        }
        
        # Add Patient resource
        patient = {
            "resourceType": "Patient",
            "id": user_id,
            "identifier": [{
                "system": "https://genomevault.io",
                "value": user_id
            }]
        }
        bundle["entry"].append({"resource": patient})
        
        # Add Observation resources for genomic data
        for variant in data.get("genomic_data", {}).get("variants", []):
            observation = {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": "69548-6",
                        "display": "Genetic variant assessment"
                    }]
                },
                "subject": {"reference": f"Patient/{user_id}"},
                "component": [
                    {
                        "code": {"text": "Chromosome"},
                        "valueString": variant["chrom"]
                    },
                    {
                        "code": {"text": "Position"},
                        "valueInteger": variant["pos"]
                    }
                ]
            }
            bundle["entry"].append({"resource": observation})
        
        with open(export_path, 'w') as f:
            json.dump(bundle, f, indent=2)
        
        return export_path
    
    def _export_csv(self, data: Dict[str, Any], user_id: str) -> Path:
        """Export data as CSV files."""
        export_dir = Path(tempfile.mkdtemp())
        
        # Export variants as CSV
        variants_path = export_dir / f"variants_{user_id}.csv"
        with open(variants_path, 'w') as f:
            f.write("chrom,pos,ref,alt,genotype\n")
            for variant in data.get("genomic_data", {}).get("variants", []):
                f.write(f"{variant['chrom']},{variant['pos']},{variant['ref']},")
                f.write(f"{variant['alt']},{variant['genotype']}\n")
        
        # Create ZIP archive
        zip_path = export_dir / f"gdpr_export_{user_id}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(variants_path, arcname="variants.csv")
        
        return zip_path
    
    def _export_xml(self, data: Dict[str, Any], user_id: str) -> Path:
        """Export data as XML."""
        import xml.etree.ElementTree as ET
        
        export_dir = Path(tempfile.mkdtemp())
        export_path = export_dir / f"gdpr_export_{user_id}.xml"
        
        # Create XML structure
        root = ET.Element("GDPRExport")
        root.set("userId", user_id)
        root.set("timestamp", datetime.utcnow().isoformat())
        
        # Add genomic data
        genomic = ET.SubElement(root, "GenomicData")
        for variant in data.get("genomic_data", {}).get("variants", []):
            var_elem = ET.SubElement(genomic, "Variant")
            var_elem.set("chrom", str(variant["chrom"]))
            var_elem.set("pos", str(variant["pos"]))
            var_elem.set("ref", variant["ref"])
            var_elem.set("alt", variant["alt"])
        
        # Write XML
        tree = ET.ElementTree(root)
        tree.write(export_path, encoding="utf-8", xml_declaration=True)
        
        return export_path
    
    def _get_genomic_data(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user's genomic data."""
        # Implementation would connect to actual data store
        return {
            "variants": [],
            "quality_scores": [],
            "coverage": []
        }
    
    def _get_clinical_data(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user's clinical data."""
        return {
            "diagnoses": [],
            "medications": [],
            "test_results": []
        }
    
    def _get_consent_records(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve consent records."""
        return []
    
    def _get_access_logs(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve access logs for user's data."""
        return []
    
    def _get_derived_insights(self, user_id: str) -> Dict[str, Any]:
        """Retrieve derived insights."""
        return {}
    
    def _get_hypervectors(self, user_id: str) -> Dict[str, Any]:
        """Retrieve hypervector encodings."""
        return {}


class DataResidencyController:
    """Control data residency for GDPR compliance."""
    
    # EU countries requiring data residency
    EU_COUNTRIES = {
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "ES", "SE"
    }
    
    # EEA countries (EU + Iceland, Liechtenstein, Norway)
    EEA_COUNTRIES = EU_COUNTRIES | {"IS", "LI", "NO"}
    
    # Countries with GDPR adequacy decisions
    ADEQUATE_COUNTRIES = {
        "AD", "AR", "CA", "FO", "GG", "IL", "IM", "JP", "JE", "NZ", 
        "CH", "UY", "GB", "KR"  # As of 2024
    }
    
    # Approved data transfer mechanisms
    TRANSFER_MECHANISMS = {
        "adequacy_decision",
        "standard_contractual_clauses", 
        "binding_corporate_rules",
        "explicit_consent",
        "approved_codes_of_conduct",
        "approved_certification"
    }
    
    # Data localization requirements by country
    LOCALIZATION_REQUIREMENTS = {
        "RU": {"health_data": True, "personal_data": True},  # Russia
        "CN": {"health_data": True, "personal_data": True},  # China
        "IN": {"health_data": True, "financial_data": True},  # India
        "VN": {"personal_data": True},  # Vietnam
        "NG": {"government_data": True},  # Nigeria
        "ID": {"system_data": True},  # Indonesia
    }
    
    def __init__(self, enable_monitoring: bool = True):
        self.residency_rules: Dict[str, Dict[str, Any]] = {}
        self.storage_regions: Dict[str, str] = {}
        self.transfer_logs: List[Dict[str, Any]] = []
        self.monitoring_enabled = enable_monitoring
        
        # Initialize regional mappings
        self._init_regional_mappings()
        
        # Initialize transfer impact assessments
        self.transfer_assessments: Dict[str, Dict[str, Any]] = {}
    
    def _init_regional_mappings(self) -> None:
        """Initialize cloud region mappings."""
        self.cloud_regions = {
            # AWS regions
            "eu-west-1": {"location": "Ireland", "provider": "AWS", "gdpr_compliant": True},
            "eu-west-2": {"location": "London", "provider": "AWS", "gdpr_compliant": True},
            "eu-west-3": {"location": "Paris", "provider": "AWS", "gdpr_compliant": True},
            "eu-central-1": {"location": "Frankfurt", "provider": "AWS", "gdpr_compliant": True},
            "eu-north-1": {"location": "Stockholm", "provider": "AWS", "gdpr_compliant": True},
            
            "us-east-1": {"location": "N. Virginia", "provider": "AWS", "gdpr_compliant": False},
            "us-west-2": {"location": "Oregon", "provider": "AWS", "gdpr_compliant": False},
            
            "ca-central-1": {"location": "Canada", "provider": "AWS", "gdpr_compliant": True},
            "ap-southeast-2": {"location": "Sydney", "provider": "AWS", "gdpr_compliant": True},
            "ap-northeast-1": {"location": "Tokyo", "provider": "AWS", "gdpr_compliant": True},
            
            # Azure regions
            "westeurope": {"location": "Netherlands", "provider": "Azure", "gdpr_compliant": True},
            "northeurope": {"location": "Ireland", "provider": "Azure", "gdpr_compliant": True},
            "uksouth": {"location": "London", "provider": "Azure", "gdpr_compliant": True},
            
            # GCP regions
            "europe-west1": {"location": "Belgium", "provider": "GCP", "gdpr_compliant": True},
            "europe-west2": {"location": "London", "provider": "GCP", "gdpr_compliant": True},
            "europe-west3": {"location": "Frankfurt", "provider": "GCP", "gdpr_compliant": True},
        }
    
    def check_residency_compliance(self, 
                                  user_country: str,
                                  storage_region: str,
                                  data_type: str = "personal_data",
                                  transfer_mechanism: Optional[str] = None) -> Dict[str, Any]:
        """Check comprehensive data residency compliance."""
        
        compliance_result = {
            "compliant": True,
            "warnings": [],
            "violations": [],
            "required_mechanisms": [],
            "recommendations": []
        }
        
        # Check EEA/EU requirements
        if user_country in self.EEA_COUNTRIES:
            result = self._check_eea_compliance(user_country, storage_region, data_type, transfer_mechanism)
            compliance_result["compliant"] &= result["compliant"]
            compliance_result["violations"].extend(result.get("violations", []))
        
        # Check adequacy decisions
        elif user_country in self.ADEQUATE_COUNTRIES:
            result = self._check_adequate_country_compliance(user_country, storage_region, data_type)
            compliance_result["compliant"] &= result["compliant"]
            compliance_result["warnings"].extend(result.get("warnings", []))
        
        # Check local data localization laws
        if user_country in self.LOCALIZATION_REQUIREMENTS:
            result = self._check_localization_compliance(user_country, storage_region, data_type)
            compliance_result["compliant"] &= result["compliant"]
            compliance_result["violations"].extend(result.get("violations", []))
        
        # Check custom residency rules
        if user_country in self.residency_rules:
            result = self._check_custom_rules(user_country, storage_region, data_type)
            compliance_result["compliant"] &= result["compliant"]
            compliance_result["violations"].extend(result.get("violations", []))
        
        # Add recommendations
        if not compliance_result["compliant"]:
            compliance_result["recommendations"] = self._get_compliance_recommendations(
                user_country, storage_region, data_type
            )
        
        # Log transfer if cross-border
        if self._is_cross_border_transfer(user_country, storage_region):
            self._log_transfer(user_country, storage_region, data_type, compliance_result)
        
        return compliance_result
    
    def _check_eea_compliance(self, 
                             user_country: str, 
                             storage_region: str, 
                             data_type: str,
                             transfer_mechanism: Optional[str]) -> Dict[str, Any]:
        """Check EEA/EU GDPR compliance."""
        region_info = self.cloud_regions.get(storage_region, {})
        
        # Data can stay within EEA
        if region_info.get("gdpr_compliant", False):
            return {"compliant": True}
        
        # Outside EEA - need valid transfer mechanism
        if transfer_mechanism in self.TRANSFER_MECHANISMS:
            # Still need to check if mechanism is appropriate
            if transfer_mechanism == "adequacy_decision":
                target_country = self._get_region_country(storage_region)
                if target_country not in self.ADEQUATE_COUNTRIES:
                    return {
                        "compliant": False,
                        "violations": [f"No adequacy decision for {target_country}"]
                    }
            return {"compliant": True}
        else:
            return {
                "compliant": False,
                "violations": [
                    f"EEA data transferred to {storage_region} without valid mechanism"
                ]
            }
    
    def _check_adequate_country_compliance(self,
                                         user_country: str,
                                         storage_region: str,
                                         data_type: str) -> Dict[str, Any]:
        """Check compliance for countries with adequacy decisions."""
        # Generally more flexible, but still need to respect local laws
        return {
            "compliant": True,
            "warnings": [f"Data from {user_country} stored in {storage_region} - monitor for changes in adequacy status"]
        }
    
    def _check_localization_compliance(self,
                                     user_country: str,
                                     storage_region: str,
                                     data_type: str) -> Dict[str, Any]:
        """Check local data localization requirements."""
        requirements = self.LOCALIZATION_REQUIREMENTS.get(user_country, {})
        
        if requirements.get(data_type, False):
            # Data must be stored in the country
            if not self._is_data_in_country(user_country, storage_region):
                return {
                    "compliant": False,
                    "violations": [
                        f"{user_country} requires {data_type} to be stored locally, but found in {storage_region}"
                    ]
                }
        
        return {"compliant": True}
    
    def _check_custom_rules(self,
                          user_country: str,
                          storage_region: str,
                          data_type: str) -> Dict[str, Any]:
        """Check custom organizational rules."""
        rule = self.residency_rules[user_country]
        allowed_regions = rule.get("allowed_regions", [])
        
        if storage_region not in allowed_regions:
            return {
                "compliant": False,
                "violations": [
                    f"Custom rule violation: {user_country} data not allowed in {storage_region}"
                ]
            }
        
        return {"compliant": True}
    
    def _is_cross_border_transfer(self, user_country: str, storage_region: str) -> bool:
        """Check if this constitutes a cross-border transfer."""
        if user_country in self.EEA_COUNTRIES:
            region_info = self.cloud_regions.get(storage_region, {})
            return not region_info.get("gdpr_compliant", False)
        
        return not self._is_data_in_country(user_country, storage_region)
    
    def _is_data_in_country(self, country: str, region: str) -> bool:
        """Check if data is stored within the specified country."""
        region_info = self.cloud_regions.get(region, {})
        region_location = region_info.get("location", "")
        
        # Simple mapping - would be more sophisticated in practice
        country_mappings = {
            "US": ["N. Virginia", "Oregon", "California"],
            "CA": ["Canada"],
            "GB": ["London"],
            "DE": ["Frankfurt"],
            "FR": ["Paris"],
            "AU": ["Sydney"],
            "JP": ["Tokyo"]
        }
        
        return region_location in country_mappings.get(country, [])
    
    def _get_region_country(self, region: str) -> str:
        """Get country for cloud region."""
        region_info = self.cloud_regions.get(region, {})
        location = region_info.get("location", "")
        
        # Map location to country code
        location_mapping = {
            "N. Virginia": "US", "Oregon": "US",
            "Ireland": "IE", "London": "GB", "Paris": "FR", 
            "Frankfurt": "DE", "Stockholm": "SE", "Netherlands": "NL",
            "Belgium": "BE", "Canada": "CA", "Sydney": "AU", "Tokyo": "JP"
        }
        
        return location_mapping.get(location, "UNKNOWN")
    
    def _get_compliance_recommendations(self,
                                     user_country: str,
                                     storage_region: str,
                                     data_type: str) -> List[str]:
        """Get recommendations for compliance."""
        recommendations = []
        
        if user_country in self.EEA_COUNTRIES:
            recommendations.append("Consider using EU-based cloud regions")
            recommendations.append("Implement Standard Contractual Clauses (SCCs)")
            recommendations.append("Conduct Transfer Impact Assessment (TIA)")
        
        if user_country in self.LOCALIZATION_REQUIREMENTS:
            requirements = self.LOCALIZATION_REQUIREMENTS[user_country]
            if requirements.get(data_type, False):
                recommendations.append(f"Store {data_type} within {user_country}")
        
        recommendations.append("Implement additional technical safeguards")
        recommendations.append("Document legal basis for transfer")
        
        return recommendations
    
    def _log_transfer(self,
                     user_country: str,
                     storage_region: str,
                     data_type: str,
                     compliance_result: Dict[str, Any]) -> None:
        """Log international data transfer."""
        if not self.monitoring_enabled:
            return
        
        transfer_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_country": user_country,
            "storage_region": storage_region,
            "data_type": data_type,
            "compliant": compliance_result["compliant"],
            "violations": compliance_result["violations"],
            "target_country": self._get_region_country(storage_region)
        }
        
        self.transfer_logs.append(transfer_log)
        
        if not compliance_result["compliant"]:
            logger.warning(f"Non-compliant cross-border transfer: {user_country} -> {storage_region}")
    
    def get_compliant_storage_region(self, 
                                   user_country: str,
                                   data_type: str = "personal_data",
                                   performance_preference: str = "balanced") -> str:
        """Get compliant storage region with performance considerations."""
        
        # Check localization requirements first
        if user_country in self.LOCALIZATION_REQUIREMENTS:
            requirements = self.LOCALIZATION_REQUIREMENTS[user_country]
            if requirements.get(data_type, False):
                return self._get_local_region(user_country)
        
        # EEA countries - prefer EU regions
        if user_country in self.EEA_COUNTRIES:
            eu_regions = [
                "eu-west-1", "eu-west-2", "eu-west-3", 
                "eu-central-1", "eu-north-1"
            ]
            return self._select_optimal_region(eu_regions, performance_preference)
        
        # Countries with adequacy decisions - more flexible
        if user_country in self.ADEQUATE_COUNTRIES:
            preferred_regions = self._get_preferred_regions(user_country)
            return self._select_optimal_region(preferred_regions, performance_preference)
        
        # Default regional mapping
        region_mapping = {
            "US": ["us-east-1", "us-west-2"],
            "CA": ["ca-central-1"],
            "JP": ["ap-northeast-1"],
            "AU": ["ap-southeast-2"],
            "IN": ["ap-south-1"],
            "SG": ["ap-southeast-1"],
            "BR": ["sa-east-1"]
        }
        
        regions = region_mapping.get(user_country, ["us-east-1"])
        return self._select_optimal_region(regions, performance_preference)
    
    def _get_local_region(self, country: str) -> str:
        """Get local region for data localization requirements."""
        local_regions = {
            "RU": "ru-central-1",
            "CN": "cn-north-1", 
            "IN": "ap-south-1",
            "ID": "ap-southeast-3"
        }
        return local_regions.get(country, "us-east-1")
    
    def _get_preferred_regions(self, country: str) -> List[str]:
        """Get preferred regions for country with adequacy decision."""
        # Prefer geographically close regions
        preferences = {
            "GB": ["eu-west-2", "eu-west-1"],  # London, Ireland
            "CA": ["ca-central-1", "us-east-1"],
            "AU": ["ap-southeast-2"],
            "JP": ["ap-northeast-1"],
            "KR": ["ap-northeast-2"],
            "NZ": ["ap-southeast-2"]
        }
        return preferences.get(country, ["us-east-1"])
    
    def _select_optimal_region(self, 
                              regions: List[str],
                              performance_preference: str) -> str:
        """Select optimal region based on performance preference."""
        if not regions:
            return "us-east-1"
        
        if performance_preference == "balanced":
            return regions[0]  # Default to first option
        elif performance_preference == "latency":
            # Would implement latency-based selection
            return regions[0]
        elif performance_preference == "cost":
            # Would implement cost-based selection
            return regions[-1]
        
        return regions[0]
    
    def conduct_transfer_impact_assessment(self,
                                         source_country: str,
                                         target_country: str,
                                         data_categories: List[str]) -> Dict[str, Any]:
        """Conduct Transfer Impact Assessment (TIA) as per EDPB guidance."""
        
        assessment_id = f"tia_{source_country}_{target_country}_{int(time.time())}"
        
        # Assess legal framework in target country
        legal_assessment = self._assess_target_country_laws(target_country)
        
        # Assess technical safeguards
        technical_assessment = self._assess_technical_safeguards(data_categories)
        
        # Assess organizational measures
        organizational_assessment = self._assess_organizational_measures()
        
        # Overall risk assessment
        overall_risk = self._calculate_transfer_risk(
            legal_assessment, technical_assessment, organizational_assessment
        )
        
        assessment = {
            "assessment_id": assessment_id,
            "date": datetime.utcnow().isoformat(),
            "source_country": source_country,
            "target_country": target_country,
            "data_categories": data_categories,
            "legal_framework": legal_assessment,
            "technical_safeguards": technical_assessment,
            "organizational_measures": organizational_assessment,
            "overall_risk": overall_risk,
            "recommendations": self._get_tia_recommendations(overall_risk),
            "next_review": (datetime.utcnow() + timedelta(days=365)).isoformat()
        }
        
        self.transfer_assessments[assessment_id] = assessment
        return assessment
    
    def _assess_target_country_laws(self, country: str) -> Dict[str, Any]:
        """Assess legal framework in target country."""
        # Simplified assessment - would be more comprehensive
        high_risk_countries = {"CN", "RU", "IR", "KP"}  # China, Russia, Iran, North Korea
        
        if country in high_risk_countries:
            risk_level = "HIGH"
            concerns = ["Government surveillance laws", "Data localization requirements"]
        elif country in self.ADEQUATE_COUNTRIES:
            risk_level = "LOW"
            concerns = []
        else:
            risk_level = "MEDIUM"
            concerns = ["Unknown privacy framework"]
        
        return {
            "risk_level": risk_level,
            "concerns": concerns,
            "adequacy_decision": country in self.ADEQUATE_COUNTRIES,
            "surveillance_laws": country in high_risk_countries
        }
    
    def _assess_technical_safeguards(self, data_categories: List[str]) -> Dict[str, Any]:
        """Assess technical safeguards for transfer."""
        safeguards = {
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "pseudonymization": "genetic_data" in data_categories,
            "anonymization": False,  # Generally not possible for genomic data
            "access_controls": True,
            "audit_logging": True
        }
        
        # Calculate effectiveness score
        effectiveness = sum(safeguards.values()) / len(safeguards) * 100
        
        return {
            "safeguards": safeguards,
            "effectiveness_score": effectiveness,
            "risk_mitigation": "HIGH" if effectiveness > 80 else "MEDIUM"
        }
    
    def _assess_organizational_measures(self) -> Dict[str, Any]:
        """Assess organizational measures."""
        measures = {
            "data_processing_agreements": True,
            "staff_training": True,
            "incident_response_plan": True,
            "data_breach_procedures": True,
            "regular_audits": True,
            "vendor_management": True
        }
        
        effectiveness = sum(measures.values()) / len(measures) * 100
        
        return {
            "measures": measures,
            "effectiveness_score": effectiveness,
            "risk_mitigation": "HIGH" if effectiveness > 80 else "MEDIUM"
        }
    
    def _calculate_transfer_risk(self,
                               legal: Dict[str, Any],
                               technical: Dict[str, Any],
                               organizational: Dict[str, Any]) -> str:
        """Calculate overall transfer risk."""
        
        # Weight the assessments
        legal_weight = 0.4
        technical_weight = 0.3
        organizational_weight = 0.3
        
        # Convert risk levels to scores
        risk_scores = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        
        legal_score = risk_scores[legal["risk_level"]]
        technical_score = 4 - (technical["effectiveness_score"] / 33.33)  # Convert to 1-3 scale
        org_score = 4 - (organizational["effectiveness_score"] / 33.33)
        
        overall_score = (
            legal_score * legal_weight +
            technical_score * technical_weight +
            org_score * organizational_weight
        )
        
        if overall_score <= 1.5:
            return "LOW"
        elif overall_score <= 2.5:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _get_tia_recommendations(self, risk_level: str) -> List[str]:
        """Get Transfer Impact Assessment recommendations."""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "Consider alternative storage locations",
                "Implement additional technical safeguards",
                "Obtain explicit consent for transfer",
                "Regular monitoring and review required"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Implement supplementary measures",
                "Enhanced monitoring required",
                "Regular review of legal developments"
            ])
        else:
            recommendations.extend([
                "Standard safeguards sufficient",
                "Annual review recommended"
            ])
        
        return recommendations
    
    def add_residency_rule(self, 
                          country: str,
                          allowed_regions: List[str],
                          data_types: List[str] = None,
                          requires_encryption: bool = True) -> None:
        """Add enhanced data residency rule for country."""
        self.residency_rules[country] = {
            "allowed_regions": allowed_regions,
            "data_types": data_types or ["personal_data"],
            "requires_encryption": requires_encryption,
            "added_at": datetime.utcnow().isoformat(),
            "requires_tia": country not in self.ADEQUATE_COUNTRIES
        }
        
        logger.info(f"Added residency rule for {country}: {allowed_regions}")
    
    def get_transfer_logs(self, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get transfer logs for audit purposes."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        return [
            log for log in self.transfer_logs
            if start_date <= datetime.fromisoformat(log["timestamp"]) <= end_date
        ]
    
    def generate_residency_report(self) -> Dict[str, Any]:
        """Generate data residency compliance report."""
        recent_logs = self.get_transfer_logs()
        
        violations = [log for log in recent_logs if not log["compliant"]]
        
        # Count transfers by country
        transfer_counts = {}
        for log in recent_logs:
            key = f"{log['user_country']}->{log['target_country']}"
            transfer_counts[key] = transfer_counts.get(key, 0) + 1
        
        return {
            "report_date": datetime.utcnow().isoformat(),
            "total_transfers": len(recent_logs),
            "violations": len(violations),
            "compliance_rate": (len(recent_logs) - len(violations)) / len(recent_logs) * 100 if recent_logs else 100,
            "transfer_patterns": transfer_counts,
            "violation_details": violations[:10],  # Top 10 violations
            "active_assessments": len(self.transfer_assessments),
            "regions_in_use": list(set(log["storage_region"] for log in recent_logs))
        }


class GDPRHandler:
    """Main GDPR compliance handler."""
    
    def __init__(self, 
                 storage_path: str = "/var/genomevault/gdpr",
                 verification_url: str = "https://genomevault.io/verify"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.verification_url = verification_url
        
        self.crypto_erasure = CryptographicErasure()
        self.data_portability = DataPortability()
        self.residency_controller = DataResidencyController()
        
        self.pending_requests: Dict[str, DataSubjectRequest] = {}
        self.completed_requests: Dict[str, DataSubjectRequest] = {}
    
    def create_request(self, 
                       user_id: str,
                       request_type: RequestType,
                       metadata: Optional[Dict[str, Any]] = None) -> DataSubjectRequest:
        """Create new data subject request."""
        request_id = hashlib.sha256(
            f"{user_id}:{request_type.value}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        request = DataSubjectRequest(
            request_id=request_id,
            user_id=user_id,
            request_type=request_type,
            status=RequestStatus.PENDING,
            created_at=datetime.utcnow(),
            metadata=metadata or {},
            verification_token=os.urandom(32).hex()
        )
        
        request.add_audit_entry("request_created", {
            "type": request_type.value,
            "metadata": metadata
        })
        
        self.pending_requests[request_id] = request
        
        gdpr_requests.labels(
            request_type=request_type.value,
            status="created"
        ).inc()
        
        # Send verification email
        self._send_verification_email(request)
        
        return request
    
    def verify_request(self, request_id: str, token: str) -> bool:
        """Verify data subject request."""
        if request_id not in self.pending_requests:
            return False
        
        request = self.pending_requests[request_id]
        
        if request.verification_token != token:
            request.add_audit_entry("verification_failed", {"reason": "invalid_token"})
            return False
        
        request.status = RequestStatus.VERIFIED
        request.verified_at = datetime.utcnow()
        request.add_audit_entry("request_verified", {})
        
        gdpr_requests.labels(
            request_type=request.request_type.value,
            status="verified"
        ).inc()
        
        # Process request
        self._process_request(request)
        
        return True
    
    def _process_request(self, request: DataSubjectRequest) -> None:
        """Process verified request."""
        request.status = RequestStatus.IN_PROGRESS
        
        with gdpr_request_duration.labels(
            request_type=request.request_type.value
        ).time():
            
            if request.request_type == RequestType.ACCESS:
                self._process_access_request(request)
            elif request.request_type == RequestType.ERASURE:
                self._process_erasure_request(request)
            elif request.request_type == RequestType.PORTABILITY:
                self._process_portability_request(request)
            elif request.request_type == RequestType.RECTIFICATION:
                self._process_rectification_request(request)
            elif request.request_type == RequestType.RESTRICTION:
                self._process_restriction_request(request)
            elif request.request_type == RequestType.OBJECTION:
                self._process_objection_request(request)
        
        request.status = RequestStatus.COMPLETED
        request.completed_at = datetime.utcnow()
        
        self.completed_requests[request.request_id] = request
        del self.pending_requests[request.request_id]
        
        gdpr_requests.labels(
            request_type=request.request_type.value,
            status="completed"
        ).inc()
    
    def _process_access_request(self, request: DataSubjectRequest) -> None:
        """Process right to access request (Article 15)."""
        # Generate comprehensive data report
        report = {
            "request_id": request.request_id,
            "user_id": request.user_id,
            "generated_at": datetime.utcnow().isoformat(),
            "purposes_of_processing": [
                "Clinical diagnosis",
                "Research (with consent)",
                "Quality improvement"
            ],
            "categories_of_data": [
                "Genomic variants",
                "Clinical phenotypes",
                "Consent records",
                "Access logs"
            ],
            "recipients": self._get_data_recipients(request.user_id),
            "retention_period": "7 years per HIPAA requirements",
            "data_sources": ["Direct upload", "Clinical partners"],
            "automated_decision_making": False,
            "cross_border_transfers": self._get_transfer_info(request.user_id)
        }
        
        # Store report
        report_path = self.storage_path / f"access_report_{request.request_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        request.metadata["report_path"] = str(report_path)
        request.add_audit_entry("access_report_generated", {"path": str(report_path)})
    
    def _process_erasure_request(self, request: DataSubjectRequest) -> None:
        """Process right to erasure request (Article 17)."""
        user_id = request.user_id
        
        # Check if erasure can be performed
        if self._has_legal_obligation(user_id):
            request.status = RequestStatus.REJECTED
            request.add_audit_entry("erasure_rejected", {
                "reason": "legal_obligation",
                "details": "HIPAA 7-year retention requirement"
            })
            return
        
        # Perform cryptographic erasure
        success = self.crypto_erasure.erase_user_data(user_id)
        
        if success:
            request.add_audit_entry("data_erased", {
                "method": "cryptographic_erasure",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Notify downstream systems
            self._notify_erasure(user_id)
        else:
            request.status = RequestStatus.REJECTED
            request.add_audit_entry("erasure_failed", {})
    
    def _process_portability_request(self, request: DataSubjectRequest) -> None:
        """Process data portability request (Article 20)."""
        format_str = request.metadata.get("format", "json")
        format = DataFormat[format_str.upper()]
        
        export_path = self.data_portability.export_user_data(
            request.user_id,
            format,
            include_derived=request.metadata.get("include_derived", False)
        )
        
        request.metadata["export_path"] = str(export_path)
        request.add_audit_entry("data_exported", {
            "format": format.value,
            "path": str(export_path)
        })
    
    def _process_rectification_request(self, request: DataSubjectRequest) -> None:
        """Process rectification request (Article 16)."""
        corrections = request.metadata.get("corrections", {})
        
        for field, new_value in corrections.items():
            # Validate and apply corrections
            old_value = self._get_field_value(request.user_id, field)
            self._update_field_value(request.user_id, field, new_value)
            
            request.add_audit_entry("data_rectified", {
                "field": field,
                "old_value": str(old_value),
                "new_value": str(new_value)
            })
    
    def _process_restriction_request(self, request: DataSubjectRequest) -> None:
        """Process restriction of processing request (Article 18)."""
        restriction_type = request.metadata.get("restriction_type", "all")
        
        # Apply processing restrictions
        self._apply_restrictions(request.user_id, restriction_type)
        
        request.add_audit_entry("processing_restricted", {
            "type": restriction_type
        })
    
    def _process_objection_request(self, request: DataSubjectRequest) -> None:
        """Process objection to processing request (Article 21)."""
        objection_scope = request.metadata.get("scope", "marketing")
        
        # Record objection
        self._record_objection(request.user_id, objection_scope)
        
        request.add_audit_entry("objection_recorded", {
            "scope": objection_scope
        })
    
    def _send_verification_email(self, request: DataSubjectRequest) -> None:
        """Send verification email for request."""
        verification_link = f"{self.verification_url}?id={request.request_id}&token={request.verification_token}"
        logger.info(f"Verification link for {request.user_id}: {verification_link}")
    
    def _get_data_recipients(self, user_id: str) -> List[str]:
        """Get list of data recipients."""
        return ["Clinical partners (with BAA)", "Research institutions (with consent)"]
    
    def _get_transfer_info(self, user_id: str) -> Dict[str, Any]:
        """Get cross-border transfer information."""
        return {
            "transfers_outside_eu": False,
            "safeguards": ["Standard Contractual Clauses", "Encryption"]
        }
    
    def _has_legal_obligation(self, user_id: str) -> bool:
        """Check if there's legal obligation to retain data."""
        # Check HIPAA 7-year requirement
        return True  # Simplified - would check actual retention requirements
    
    def _notify_erasure(self, user_id: str) -> None:
        """Notify downstream systems of erasure."""
        logger.info(f"Notifying systems of erasure for user {user_id}")
    
    def _get_field_value(self, user_id: str, field: str) -> Any:
        """Get current field value."""
        return None
    
    def _update_field_value(self, user_id: str, field: str, value: Any) -> None:
        """Update field value."""
        pass
    
    def _apply_restrictions(self, user_id: str, restriction_type: str) -> None:
        """Apply processing restrictions."""
        pass
    
    def _record_objection(self, user_id: str, scope: str) -> None:
        """Record processing objection."""
        pass