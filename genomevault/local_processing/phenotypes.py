"""
GenomeVault Phenotypes Processing

Handles clinical and phenotypic data processing including EHR integration,
FHIR data parsing, and phenotype standardization.

"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from genomevault.utils import get_config, get_logger, secure_hash
from genomevault.utils.logging import log_operation

logger = get_logger(__name__)
config = get_config()


class PhenotypeCategory(Enum):
    """Categories of phenotypic data"""

    DEMOGRAPHIC = "demographic"
    CLINICAL = "clinical"
    LABORATORY = "laboratory"
    MEDICATION = "medication"
    FAMILY_HISTORY = "family_history"
    LIFESTYLE = "lifestyle"
    ENVIRONMENTAL = "environmental"


@dataclass
class ClinicalMeasurement:
    """Single clinical measurement or observation"""

    measurement_id: str
    measurement_type: str
    value: str | float | bool
    unit: str | None = None
    date: datetime.datetime | None = None
    source: str = "EHR"
    code_system: str | None = None  # LOINC, SNOMED, etc.
    code: str | None = None
    reference_range: dict[str, float] | None = None
    abnormal_flag: str | None = None

    def is_abnormal(self) -> bool:
        """Check if measurement is abnormal"""
        if self.abnormal_flag:
            return self.abnormal_flag.upper() in ["H", "L", "A", "ABNORMAL"]

        if self.reference_range and isinstance(self.value, (int, float)):
            if "low" in self.reference_range and self.value < self.reference_range["low"]:
                return True
            if "high" in self.reference_range and self.value > self.reference_range["high"]:
                return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.measurement_id,
            "type": self.measurement_type,
            "value": self.value,
            "unit": self.unit,
            "date": self.date.isoformat() if self.date else None,
            "source": self.source,
            "code_system": self.code_system,
            "code": self.code,
            "reference_range": self.reference_range,
            "abnormal": self.is_abnormal(),
        }


@dataclass
class Diagnosis:
    """Clinical diagnosis"""

    diagnosis_id: str
    name: str
    icd10_code: str | None = None
    snomed_code: str | None = None
    date_diagnosed: datetime.datetime | None = None
    severity: str | None = None
    status: str = "active"  # active, resolved, inactive
    certainty: float = 1.0  # 0-1 confidence
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.diagnosis_id,
            "name": self.name,
            "icd10": self.icd10_code,
            "snomed": self.snomed_code,
            "date": self.date_diagnosed.isoformat() if self.date_diagnosed else None,
            "severity": self.severity,
            "status": self.status,
            "certainty": self.certainty,
            "metadata": self.metadata,
        }


@dataclass
class Medication:
    """Medication information"""

    medication_id: str
    name: str
    rxnorm_code: str | None = None
    dose: str | None = None
    frequency: str | None = None
    route: str | None = None
    start_date: datetime.datetime | None = None
    end_date: datetime.datetime | None = None
    status: str = "active"  # active, completed, discontinued
    indication: str | None = None

    @property
    def is_active(self) -> bool:
        """Check if medication is currently active"""
        if self.status != "active":
            return False

        if self.end_date:
            return datetime.datetime.now() <= self.end_date

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.medication_id,
            "name": self.name,
            "rxnorm": self.rxnorm_code,
            "dose": self.dose,
            "frequency": self.frequency,
            "route": self.route,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status,
            "indication": self.indication,
            "active": self.is_active,
        }


@dataclass
class FamilyHistory:
    """Family history entry"""

    relationship: str  # mother, father, sibling, etc.
    condition: str
    age_at_onset: int | None = None
    outcome: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "relationship": self.relationship,
            "condition": self.condition,
            "age_at_onset": self.age_at_onset,
            "outcome": self.outcome,
        }


@dataclass
class PhenotypeProfile:
    """Complete phenotypic profile"""

    sample_id: str
    demographics: dict[str, Any]
    measurements: list[ClinicalMeasurement]
    diagnoses: list[Diagnosis]
    medications: list[Medication]
    family_history: list[FamilyHistory]
    lifestyle_factors: dict[str, Any]
    environmental_exposures: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_measurement_summary(self) -> dict[str, Any]:
        """Get summary of measurements by type"""
        summary = {}
        for measurement in self.measurements:
            if measurement.measurement_type not in summary:
                summary[measurement.measurement_type] = []
            summary[measurement.measurement_type].append(
                {
                    "value": measurement.value,
                    "unit": measurement.unit,
                    "date": measurement.date,
                    "abnormal": measurement.is_abnormal(),
                }
            )
        return summary

    def get_active_conditions(self) -> list[Diagnosis]:
        """Get list of active diagnoses"""
        return [d for d in self.diagnoses if d.status == "active"]

    def get_active_medications(self) -> list[Medication]:
        """Get list of active medications"""
        return [m for m in self.medications if m.is_active]

    def calculate_risk_factors(self) -> dict[str, float]:
        """Calculate basic risk factors from phenotype data"""
        risk_factors = {}

        # Age risk
        if "age" in self.demographics:
            age = self.demographics["age"]
            if age > 65:
                risk_factors["age"] = min((age - 65) / 20, 1.0)
            else:
                risk_factors["age"] = 0.0

        # BMI risk
        if "height" in self.demographics and "weight" in self.demographics:
            height_m = self.demographics["height"] / 100  # cm to m
            weight_kg = self.demographics["weight"]
            bmi = weight_kg / (height_m**2)

            if bmi < 18.5 or bmi > 30:
                risk_factors["bmi"] = min(abs(bmi - 25) / 10, 1.0)
            else:
                risk_factors["bmi"] = 0.0

        # Smoking risk
        if self.lifestyle_factors.get("smoking_status") == "current":
            risk_factors["smoking"] = 1.0
        elif self.lifestyle_factors.get("smoking_status") == "former":
            risk_factors["smoking"] = 0.5
        else:
            risk_factors["smoking"] = 0.0

        # Family history risk
        high_risk_conditions = {"cancer", "heart disease", "diabetes", "stroke"}
        family_risk = 0.0
        for history in self.family_history:
            if any(condition in history.condition.lower() for condition in high_risk_conditions):
                if history.relationship in ["mother", "father"]:
                    family_risk += 0.3
                elif history.relationship == "sibling":
                    family_risk += 0.2
        risk_factors["family_history"] = min(family_risk, 1.0)

        return risk_factors


class PhenotypeProcessor:
    """Process clinical and phenotypic data"""

    # Standard code mappings
    LOINC_CODES = {
        "glucose": "2345-7",
        "hemoglobin_a1c": "4548-4",
        "cholesterol_total": "2093-3",
        "cholesterol_ldl": "13457-7",
        "cholesterol_hdl": "2085-9",
        "triglycerides": "2571-8",
        "blood_pressure_systolic": "8480-6",
        "blood_pressure_diastolic": "8462-4",
        "bmi": "39156-5",
        "creatinine": "2160-0",
    }

    ICD10_PATTERNS = {
        "diabetes": r"^E1[0-4]\.",
        "hypertension": r"^I1[0-5]\.",
        "heart_disease": r"^I[2-5]\d\.",
        "cancer": r"^C\d{2}\.",
        "copd": r"^J4[0-4]\.",
        "asthma": r"^J45\.",
    }

    def __init__(self):
        """Initialize phenotype processor"""
        self.terminology_mapper = self._load_terminology_mappings()

    def _load_terminology_mappings(self) -> dict[str, dict[str, str]]:
        """Load terminology mappings"""
        # In production, would load from comprehensive terminology services
        return {"loinc": self.LOINC_CODES, "icd10_patterns": self.ICD10_PATTERNS}

    @log_operation("process_phenotypes")
    def process(
        self,
        input_data: dict[str, Any] | Path,
        sample_id: str,
        data_format: str = "fhir",
    ) -> PhenotypeProfile:
        """
        Process phenotypic data

        Args:
            input_data: Input data (dict or path to file)
            sample_id: Sample identifier
            data_format: Data format (fhir, csv, custom)

        Returns:
            PhenotypeProfile with structured phenotypic data
        """
        logger.info("Processing phenotypic data for sample %ssample_id")

        # Load data if path provided
        if isinstance(input_data, Path):
            with open(input_data) as f:
                if input_data.suffix == ".json":
                    input_data = json.load(f)
                else:
                    raise ValueError("Unsupported file format: {input_data.suffix}")

        # Process based on format
        if data_format.lower() == "fhir":
            profile = self._process_fhir_data(input_data, sample_id)
        elif data_format.lower() == "csv":
            profile = self._process_csv_data(input_data, sample_id)
        else:
            profile = self._process_custom_data(input_data, sample_id)

        # Standardize codes
        profile = self._standardize_codes(profile)

        # Calculate derived values
        profile = self._calculate_derived_values(profile)

        logger.info(
            "Phenotype processing complete. "
            "%slen(profile.measurements) measurements, "
            "%slen(profile.diagnoses)} diagnoses, "
            "{len(profile.medications)} medications"
        )

        return profile

    def _process_fhir_data(self, fhir_bundle: dict[str, Any], sample_id: str) -> PhenotypeProfile:
        """Process FHIR bundle data"""
        demographics = {}
        measurements = []
        diagnoses = []
        medications = []
        family_history = []

        # Process FHIR entries
        for entry in fhir_bundle.get("entry", []):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")

            if resource_type == "Patient":
                # Extract demographics
                demographics = self._extract_fhir_demographics(resource)

            elif resource_type == "Observation":
                # Extract measurements
                measurement = self._extract_fhir_observation(resource)
                if measurement:
                    measurements.append(measurement)

            elif resource_type == "Condition":
                # Extract diagnoses
                diagnosis = self._extract_fhir_condition(resource)
                if diagnosis:
                    diagnoses.append(diagnosis)

            elif resource_type == "MedicationStatement":
                # Extract medications
                medication = self._extract_fhir_medication(resource)
                if medication:
                    medications.append(medication)

            elif resource_type == "FamilyMemberHistory":
                # Extract family history
                history = self._extract_fhir_family_history(resource)
                if history:
                    family_history.extend(history)

        return PhenotypeProfile(
            sample_id=sample_id,
            demographics=demographics,
            measurements=measurements,
            diagnoses=diagnoses,
            medications=medications,
            family_history=family_history,
            lifestyle_factors={},
            environmental_exposures={},
            metadata={
                "source": "fhir",
                "version": fhir_bundle.get("meta", {}).get("versionId"),
            },
        )

    def _extract_fhir_demographics(self, patient: dict[str, Any]) -> dict[str, Any]:
        """Extract demographics from FHIR Patient resource"""
        demographics = {}

        # Birth date and age
        if "birthDate" in patient:
            birth_date = datetime.datetime.strptime(patient["birthDate"], "%Y-%m-%d")
            age = (datetime.datetime.now() - birth_date).days // 365
            demographics["birth_date"] = birth_date
            demographics["age"] = age

        # Gender
        if "gender" in patient:
            demographics["gender"] = patient["gender"]

        # Race and ethnicity from extensions
        for extension in patient.get("extension", []):
            url = extension.get("url", "")
            if "race" in url:
                demographics["race"] = extension.get("valueString")
            elif "ethnicity" in url:
                demographics["ethnicity"] = extension.get("valueString")

        return demographics

    def _extract_fhir_observation(self, observation: dict[str, Any]) -> ClinicalMeasurement | None:
        """Extract measurement from FHIR Observation"""
        # Get observation code
        coding = observation.get("code", {}).get("coding", [])
        if not coding:
            return None

        code_info = coding[0]

        # Extract value
        value = None
        unit = None

        if "valueQuantity" in observation:
            value = observation["valueQuantity"].get("value")
            unit = observation["valueQuantity"].get("unit")
        elif "valueString" in observation:
            value = observation["valueString"]
        elif "valueBoolean" in observation:
            value = observation["valueBoolean"]

        if value is None:
            return None

        # Parse date
        date = None
        if "effectiveDateTime" in observation:
            date = datetime.datetime.fromisoformat(
                observation["effectiveDateTime"].replace("Z", "+00:00")
            )

        # Create measurement
        measurement = ClinicalMeasurement(
            measurement_id=observation.get("id", secure_hash(str(observation).encode())[:8]),
            measurement_type=code_info.get("display", code_info.get("code")),
            value=value,
            unit=unit,
            date=date,
            source="FHIR",
            code_system=code_info.get("system"),
            code=code_info.get("code"),
        )

        # Add reference range if available
        reference_range = observation.get("referenceRange", [])
        if reference_range:
            range_info = reference_range[0]
            measurement.reference_range = {}
            if "low" in range_info:
                measurement.reference_range["low"] = range_info["low"].get("value")
            if "high" in range_info:
                measurement.reference_range["high"] = range_info["high"].get("value")

        # Add interpretation
        interpretation = observation.get("interpretation", [])
        if interpretation:
            coding = interpretation[0].get("coding", [])
            if coding:
                measurement.abnormal_flag = coding[0].get("code")

        return measurement

    def _extract_fhir_condition(self, condition: dict[str, Any]) -> Diagnosis | None:
        """Extract diagnosis from FHIR Condition"""
        # Get condition code
        coding = condition.get("code", {}).get("coding", [])
        if not coding:
            return None

        code_info = coding[0]

        # Parse dates
        date_diagnosed = None
        if "onsetDateTime" in condition:
            date_diagnosed = datetime.datetime.fromisoformat(
                condition["onsetDateTime"].replace("Z", "+00:00")
            )

        # Create diagnosis
        diagnosis = Diagnosis(
            diagnosis_id=condition.get("id", secure_hash(str(condition).encode())[:8]),
            name=code_info.get("display", code_info.get("code")),
            date_diagnosed=date_diagnosed,
            status=(
                "active"
                if condition.get("clinicalStatus", {}).get("coding", [{}])[0].get("code")
                == "active"
                else "resolved"
            ),
        )

        # Add ICD-10 code if available
        for coding in condition.get("code", {}).get("coding", []):
            if "icd" in coding.get("system", "").lower():
                diagnosis.icd10_code = coding.get("code")
            elif "snomed" in coding.get("system", "").lower():
                diagnosis.snomed_code = coding.get("code")

        # Add severity
        if "severity" in condition:
            severity_coding = condition["severity"].get("coding", [])
            if severity_coding:
                diagnosis.severity = severity_coding[0].get("display")

        return diagnosis

    def _extract_fhir_medication(self, medication: dict[str, Any]) -> Medication | None:
        """Extract medication from FHIR MedicationStatement"""
        # Get medication code
        med_codeable = medication.get("medicationCodeableConcept", {})
        coding = med_codeable.get("coding", [])

        if not coding:
            return None

        code_info = coding[0]

        # Parse dates
        start_date = None
        end_date = None

        effective_period = medication.get("effectivePeriod", {})
        if "start" in effective_period:
            start_date = datetime.datetime.fromisoformat(
                effective_period["start"].replace("Z", "+00:00")
            )
        if "end" in effective_period:
            end_date = datetime.datetime.fromisoformat(
                effective_period["end"].replace("Z", "+00:00")
            )

        # Create medication
        med = Medication(
            medication_id=medication.get("id", secure_hash(str(medication).encode())[:8]),
            name=code_info.get("display", code_info.get("code")),
            start_date=start_date,
            end_date=end_date,
            status=medication.get("status", "active"),
        )

        # Add RxNorm code if available
        for coding in med_codeable.get("coding", []):
            if "rxnorm" in coding.get("system", "").lower():
                med.rxnorm_code = coding.get("code")

        # Extract dosage information
        dosage = medication.get("dosage", [])
        if dosage:
            dosage_info = dosage[0]
            if "text" in dosage_info:
                med.dose = dosage_info["text"]
            elif "doseAndRate" in dosage_info:
                dose_rate = dosage_info["doseAndRate"][0]
                if "doseQuantity" in dose_rate:
                    dose_rate["doseQuantity"]
                    med.dose = "{dose_qty.get('value')} {dose_qty.get('unit')}"

            if "route" in dosage_info:
                route_coding = dosage_info["route"].get("coding", [])
                if route_coding:
                    med.route = route_coding[0].get("display")

        # Extract reason/indication
        reason = medication.get("reasonCode", [])
        if reason:
            reason_coding = reason[0].get("coding", [])
            if reason_coding:
                med.indication = reason_coding[0].get("display")

        return med

    def _extract_fhir_family_history(self, history: dict[str, Any]) -> list[FamilyHistory]:
        """Extract family history from FHIR FamilyMemberHistory"""
        family_history = []

        # Get relationship
        relationship_coding = history.get("relationship", {}).get("coding", [])
        if not relationship_coding:
            return family_history

        relationship = relationship_coding[0].get("display", relationship_coding[0].get("code"))

        # Get conditions
        for condition in history.get("condition", []):
            condition_coding = condition.get("code", {}).get("coding", [])
            if not condition_coding:
                continue

            condition_name = condition_coding[0].get("display", condition_coding[0].get("code"))

            # Get age at onset
            age_at_onset = None
            onset = condition.get("onsetAge")
            if onset:
                age_at_onset = onset.get("value")

            # Get outcome
            outcome = None
            outcome_coding = condition.get("outcome", {}).get("coding", [])
            if outcome_coding:
                outcome = outcome_coding[0].get("display")

            family_history.append(
                FamilyHistory(
                    relationship=relationship,
                    condition=condition_name,
                    age_at_onset=age_at_onset,
                    outcome=outcome,
                )
            )

        return family_history

    def _process_csv_data(self, csv_data: Any, sample_id: str) -> PhenotypeProfile:
        """Process CSV format data (placeholder)"""
        # This would be implemented based on specific CSV format
        logger.warning("CSV processing not fully implemented")
        return PhenotypeProfile(
            sample_id=sample_id,
            demographics={},
            measurements=[],
            diagnoses=[],
            medications=[],
            family_history=[],
            lifestyle_factors={},
            environmental_exposures={},
            metadata={"source": "csv"},
        )

    def _process_custom_data(self, custom_data: dict[str, Any], sample_id: str) -> PhenotypeProfile:
        """Process custom format data"""
        return PhenotypeProfile(
            sample_id=sample_id,
            demographics=custom_data.get("demographics", {}),
            measurements=[ClinicalMeasurement(**m) for m in custom_data.get("measurements", [])],
            diagnoses=[Diagnosis(**d) for d in custom_data.get("diagnoses", [])],
            medications=[Medication(**m) for m in custom_data.get("medications", [])],
            family_history=[FamilyHistory(**f) for f in custom_data.get("family_history", [])],
            lifestyle_factors=custom_data.get("lifestyle_factors", {}),
            environmental_exposures=custom_data.get("environmental_exposures", {}),
            metadata={"source": "custom"},
        )

    def _standardize_codes(self, profile: PhenotypeProfile) -> PhenotypeProfile:
        """Standardize medical codes"""
        # Add LOINC codes to measurements
        for measurement in profile.measurements:
            if not measurement.code and measurement.measurement_type.lower() in self.LOINC_CODES:
                measurement.code_system = "http://loinc.org"
                measurement.code = self.LOINC_CODES[measurement.measurement_type.lower()]

        # Categorize diagnoses by ICD-10 patterns
        for diagnosis in profile.diagnoses:
            if diagnosis.icd10_code:
                for category, pattern in self.ICD10_PATTERNS.items():
                    if re.match(pattern, diagnosis.icd10_code):
                        diagnosis.metadata = diagnosis.metadata or {}
                        diagnosis.metadata["category"] = category
                        break

        return profile

    def _calculate_derived_values(self, profile: PhenotypeProfile) -> PhenotypeProfile:
        """Calculate derived values from measurements"""
        # Calculate BMI if height and weight available
        height = None
        weight = None

        for measurement in profile.measurements:
            if measurement.measurement_type.lower() == "height" and measurement.unit in [
                "cm",
                "centimeter",
            ]:
                height = measurement.value
            elif measurement.measurement_type.lower() == "weight" and measurement.unit in [
                "kg",
                "kilogram",
            ]:
                weight = measurement.value

        if (
            height
            and weight
            and isinstance(height, (int, float))
            and isinstance(weight, (int, float))
        ):
            height_m = height / 100
            bmi = weight / (height_m**2)

            # Add BMI measurement
            bmi_measurement = ClinicalMeasurement(
                measurement_id="derived_bmi_{secure_hash('{height}{weight}'.encode())[:8]}",
                measurement_type="BMI",
                value=round(bmi, 1),
                unit="kg/m2",
                date=datetime.datetime.now(),
                source="derived",
                code_system="http://loinc.org",
                code=self.LOINC_CODES.get("bmi"),
                reference_range={"low": 18.5, "high": 25.0},
            )
            profile.measurements.append(bmi_measurement)

            # Update demographics
            profile.demographics["height"] = height
            profile.demographics["weight"] = weight
            profile.demographics["bmi"] = round(bmi, 1)

        return profile

    def merge_profiles(
        self, profiles: list[PhenotypeProfile], merge_strategy: str = "most_recent"
    ) -> PhenotypeProfile:
        """
        Merge multiple phenotype profiles

        Args:
            profiles: List of profiles to merge
            merge_strategy: Strategy for handling conflicts (most_recent, union, intersection)

        Returns:
            Merged phenotype profile
        """
        if not profiles:
            raise ValueError("No profiles to merge")

        if len(profiles) == 1:
            return profiles[0]

        # Start with first profile as base
        merged = profiles[0]

        for profile in profiles[1:]:
            if merge_strategy == "most_recent":
                self._merge_most_recent(merged, profile)
            elif merge_strategy == "union":
                self._merge_union(merged, profile)
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")

        # Update metadata
        merged.metadata["merged_from"] = [p.sample_id for p in profiles]
        merged.metadata["merge_strategy"] = merge_strategy

        return merged

    def _merge_most_recent(self, merged: PhenotypeProfile, profile: PhenotypeProfile) -> None:
        """Merge using most recent strategy"""
        # Merge measurements
        self._merge_measurements_most_recent(merged, profile)

        # Merge diagnoses and medications
        self._merge_diagnoses(merged, profile)
        self._merge_medications(merged, profile)

        # Merge family history
        merged.family_history.extend(profile.family_history)

        # Merge demographics
        self._merge_demographics(merged, profile)

    def _merge_union(self, merged: PhenotypeProfile, profile: PhenotypeProfile) -> None:
        """Merge using union strategy - keep all data"""
        merged.measurements.extend(profile.measurements)
        merged.diagnoses.extend(profile.diagnoses)
        merged.medications.extend(profile.medications)
        merged.family_history.extend(profile.family_history)

    def _merge_measurements_most_recent(
        self, merged: PhenotypeProfile, profile: PhenotypeProfile
    ) -> None:
        """Merge measurements keeping most recent ones"""
        measurement_dict = {m.measurement_type: m for m in merged.measurements}

        for measurement in profile.measurements:
            existing = measurement_dict.get(measurement.measurement_type)
            if self._should_replace_measurement(existing, measurement):
                measurement_dict[measurement.measurement_type] = measurement

        merged.measurements = list(measurement_dict.values())

    def _should_replace_measurement(
        self, existing: ClinicalMeasurement | None, new: ClinicalMeasurement
    ) -> bool:
        """Check if measurement should be replaced"""
        if not existing:
            return True
        if not new.date or not existing.date:
            return True
        return new.date > existing.date

    def _merge_diagnoses(self, merged: PhenotypeProfile, profile: PhenotypeProfile) -> None:
        """Merge diagnoses avoiding duplicates"""
        diagnosis_ids = {d.diagnosis_id for d in merged.diagnoses}
        merged.diagnoses.extend(
            [d for d in profile.diagnoses if d.diagnosis_id not in diagnosis_ids]
        )

    def _merge_medications(self, merged: PhenotypeProfile, profile: PhenotypeProfile) -> None:
        """Merge medications avoiding duplicates"""
        medication_ids = {m.medication_id for m in merged.medications}
        merged.medications.extend(
            [m for m in profile.medications if m.medication_id not in medication_ids]
        )

    def _merge_demographics(self, merged: PhenotypeProfile, profile: PhenotypeProfile) -> None:
        """Merge demographics with non-empty values"""
        for key, value in profile.demographics.items():
            if value and (key not in merged.demographics or not merged.demographics[key]):
                merged.demographics[key] = value
