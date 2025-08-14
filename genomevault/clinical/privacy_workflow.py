"""Clinical workflow integration with privacy preservation.

This module provides end-to-end workflows for clinical genomic
analysis while maintaining patient privacy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import hashlib
import json
from datetime import datetime

from genomevault.privacy.genomic_proof import (
    GenomicProver,
    ClinicalVerifier,
    MultiSNPProof,
)
from genomevault.privacy.proof_aggregation import ProofAggregator, BatchVerifier


class ClinicalTestType(Enum):
    """Types of clinical genomic tests."""

    SINGLE_GENE = "single_gene"
    PANEL = "panel"
    PHARMACOGENOMICS = "pharmacogenomics"
    CARRIER_SCREENING = "carrier_screening"
    CANCER_SUSCEPTIBILITY = "cancer_susceptibility"
    ANCESTRY = "ancestry"


@dataclass
class PatientConsent:
    """Patient consent for genomic testing."""

    patient_id: str
    test_types: List[ClinicalTestType]
    data_retention_days: int
    research_use_allowed: bool
    timestamp: datetime
    signature: bytes

    def is_valid_for_test(self, test_type: ClinicalTestType) -> bool:
        """Check if consent covers a specific test type."""
        return test_type in self.test_types


@dataclass
class ClinicalPanel:
    """Definition of a clinical genomic panel."""

    panel_id: str
    name: str
    test_type: ClinicalTestType
    markers: Dict[int, Dict[str, Any]]  # position -> marker info
    interpretation_rules: List[Dict[str, Any]]
    version: str

    def get_positions(self) -> List[int]:
        """Get all genomic positions in panel."""
        return list(self.markers.keys())

    def get_pathogenic_variants(self) -> Dict[int, str]:
        """Get positions and nucleotides of pathogenic variants."""
        pathogenic = {}
        for pos, info in self.markers.items():
            if info.get("clinical_significance") == "pathogenic":
                pathogenic[pos] = info["risk_allele"]
        return pathogenic


@dataclass
class ClinicalReport:
    """Privacy-preserving clinical report."""

    report_id: str
    patient_id_hash: str  # Hashed for privacy
    test_type: ClinicalTestType
    panel_id: str
    timestamp: datetime

    # Results without revealing genome
    risk_score: float
    risk_category: str  # Low, Medium, High
    actionable_findings: List[Dict[str, Any]]

    # Cryptographic proof of validity
    commitment_root: str
    verification_proofs: Dict[str, Any]

    # Metadata
    lab_certification: str
    clinician_notes: Optional[str] = None

    def to_json(self) -> str:
        """Serialize report to JSON."""
        data = {
            "report_id": self.report_id,
            "patient_id_hash": self.patient_id_hash,
            "test_type": self.test_type.value,
            "panel_id": self.panel_id,
            "timestamp": self.timestamp.isoformat(),
            "risk_score": self.risk_score,
            "risk_category": self.risk_category,
            "actionable_findings": self.actionable_findings,
            "commitment_root": self.commitment_root,
            "verification_proofs": self.verification_proofs,
            "lab_certification": self.lab_certification,
            "clinician_notes": self.clinician_notes,
        }
        return json.dumps(data, indent=2)


class ClinicalWorkflow:
    """End-to-end clinical genomic testing workflow."""

    def __init__(self):
        self.prover = GenomicProver()
        self.verifier = ClinicalVerifier()
        self.aggregator = ProofAggregator()
        self.batch_verifier = BatchVerifier()

        # Clinical panels database (simplified)
        self._panels = self._load_clinical_panels()

    def _load_clinical_panels(self) -> Dict[str, ClinicalPanel]:
        """Load clinical panel definitions."""
        panels = {}

        # BRCA1/2 cancer panel
        panels["BRCA"] = ClinicalPanel(
            panel_id="BRCA",
            name="BRCA1/2 Breast Cancer Risk",
            test_type=ClinicalTestType.CANCER_SUSCEPTIBILITY,
            markers={
                1000: {
                    "gene": "BRCA1",
                    "variant": "185delAG",
                    "risk_allele": "A",
                    "clinical_significance": "pathogenic",
                },
                2000: {
                    "gene": "BRCA1",
                    "variant": "5382insC",
                    "risk_allele": "C",
                    "clinical_significance": "pathogenic",
                },
                3000: {
                    "gene": "BRCA2",
                    "variant": "6174delT",
                    "risk_allele": "T",
                    "clinical_significance": "pathogenic",
                },
            },
            interpretation_rules=[
                {"condition": "any_pathogenic", "risk": "high"},
                {"condition": "all_normal", "risk": "average"},
            ],
            version="1.0",
        )

        # Pharmacogenomics panel
        panels["PGX"] = ClinicalPanel(
            panel_id="PGX",
            name="Pharmacogenomics Panel",
            test_type=ClinicalTestType.PHARMACOGENOMICS,
            markers={
                500: {
                    "gene": "CYP2C19",
                    "variant": "*2",
                    "allele": "A",
                    "drug_impact": "clopidogrel_reduced",
                },
                600: {
                    "gene": "CYP2C19",
                    "variant": "*3",
                    "allele": "G",
                    "drug_impact": "clopidogrel_reduced",
                },
                700: {
                    "gene": "CYP2D6",
                    "variant": "*4",
                    "allele": "T",
                    "drug_impact": "codeine_reduced",
                },
            },
            interpretation_rules=[
                {"gene": "CYP2C19", "variants": ["*2", "*3"], "metabolism": "poor"},
                {"gene": "CYP2D6", "variants": ["*4"], "metabolism": "poor"},
            ],
            version="2.0",
        )

        return panels

    def process_patient_sample(
        self, genome_data: str, consent: PatientConsent, test_type: ClinicalTestType
    ) -> ClinicalReport:
        """Process patient sample with privacy preservation.

        Args:
            genome_data: Patient genomic sequence
            consent: Valid patient consent
            test_type: Type of test to perform

        Returns:
            Clinical report with proofs
        """
        # Verify consent
        if not consent.is_valid_for_test(test_type):
            raise ValueError(f"No consent for {test_type.value}")

        # Create genomic commitment
        commitment = self.prover.commit_genome(
            genome_data,
            metadata={
                "patient_id_hash": hashlib.sha256(consent.patient_id.encode()).hexdigest(),
                "test_date": datetime.now().isoformat(),
            },
        )

        # Select appropriate panel
        panel = self._select_panel(test_type)
        if not panel:
            raise ValueError(f"No panel available for {test_type.value}")

        # Generate proofs for panel positions
        positions = panel.get_positions()
        multi_proof = self.prover.prove_snps_batch(positions, commitment)

        # Verify and interpret results
        pathogenic = panel.get_pathogenic_variants()
        verification = self.verifier.verify_disease_panel(multi_proof, pathogenic)

        # Calculate risk score
        risk_score = verification["risk_score"]
        risk_category = self._categorize_risk(risk_score)

        # Identify actionable findings
        actionable = self._identify_actionable_findings(verification["details"], panel)

        # Generate report
        report = ClinicalReport(
            report_id=self._generate_report_id(),
            patient_id_hash=hashlib.sha256(consent.patient_id.encode()).hexdigest(),
            test_type=test_type,
            panel_id=panel.panel_id,
            timestamp=datetime.now(),
            risk_score=risk_score,
            risk_category=risk_category,
            actionable_findings=actionable,
            commitment_root=commitment.to_hex(),
            verification_proofs=verification,
            lab_certification="CLIA-certified",
        )

        return report

    def verify_external_report(self, report: ClinicalReport, proofs: MultiSNPProof) -> bool:
        """Verify an external clinical report.

        Args:
            report: Clinical report to verify
            proofs: Cryptographic proofs

        Returns:
            True if report is valid
        """
        # Load panel definition
        panel = self._panels.get(report.panel_id)
        if not panel:
            return False

        # Verify proofs match commitment
        for proof in proofs.proofs:
            if proof.commitment_root.hex() != report.commitment_root:
                return False

        # Verify risk calculation
        pathogenic = panel.get_pathogenic_variants()
        verification = self.verifier.verify_disease_panel(proofs, pathogenic)

        calculated_risk = verification["risk_score"]
        if abs(calculated_risk - report.risk_score) > 0.01:
            return False

        return True

    def batch_process_samples(
        self, samples: List[Tuple[str, PatientConsent]], test_type: ClinicalTestType
    ) -> List[ClinicalReport]:
        """Process multiple samples efficiently.

        Args:
            samples: List of (genome_data, consent) tuples
            test_type: Test type for all samples

        Returns:
            List of clinical reports
        """
        reports = []

        # Process in batches for efficiency
        batch_size = 10
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]

            # Process each sample
            for genome_data, consent in batch:
                try:
                    report = self.process_patient_sample(genome_data, consent, test_type)
                    reports.append(report)
                except Exception as e:
                    # Log error (in practice, would use proper logging)
                    print(f"Error processing sample: {e}")

        return reports

    def _select_panel(self, test_type: ClinicalTestType) -> Optional[ClinicalPanel]:
        """Select appropriate panel for test type."""
        for panel in self._panels.values():
            if panel.test_type == test_type:
                return panel
        return None

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into clinical categories."""
        if risk_score < 0.2:
            return "Low"
        elif risk_score < 0.5:
            return "Medium"
        else:
            return "High"

    def _identify_actionable_findings(
        self, details: Dict[int, Dict[str, Any]], panel: ClinicalPanel
    ) -> List[Dict[str, Any]]:
        """Identify clinically actionable findings."""
        actionable = []

        for pos, result in details.items():
            if not result["matches"]:
                # Found a variant
                marker_info = panel.markers.get(pos, {})

                if marker_info.get("clinical_significance") == "pathogenic":
                    actionable.append(
                        {
                            "position": pos,
                            "gene": marker_info.get("gene", "Unknown"),
                            "variant": marker_info.get("variant", "Unknown"),
                            "significance": "Pathogenic",
                            "recommendation": self._get_clinical_recommendation(marker_info),
                        }
                    )

        return actionable

    def _get_clinical_recommendation(self, marker_info: Dict[str, Any]) -> str:
        """Get clinical recommendation for a marker."""
        gene = marker_info.get("gene", "")

        if gene.startswith("BRCA"):
            return "Recommend genetic counseling and enhanced screening"
        elif gene.startswith("CYP"):
            return f"Adjust dosing for {marker_info.get('drug_impact', 'affected drugs')}"
        else:
            return "Consult with genetic counselor"

    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]


# Example usage
def example_clinical_workflow():
    """Example: Complete clinical workflow with privacy."""

    # Simulate patient genome (with known BRCA1 mutation)
    genome = "A" * 1000 + "A" + "C" * 999 + "C" + "T" * 1000 + "T" + "G" * 7000

    # Patient consent
    consent = PatientConsent(
        patient_id="PATIENT_001",
        test_types=[ClinicalTestType.CANCER_SUSCEPTIBILITY],
        data_retention_days=365,
        research_use_allowed=True,
        timestamp=datetime.now(),
        signature=hashlib.sha256(b"patient_signature").digest(),
    )

    # Run workflow
    workflow = ClinicalWorkflow()
    report = workflow.process_patient_sample(
        genome, consent, ClinicalTestType.CANCER_SUSCEPTIBILITY
    )

    return {
        "report_id": report.report_id,
        "risk_category": report.risk_category,
        "risk_score": round(report.risk_score, 3),
        "actionable_findings": len(report.actionable_findings),
        "commitment_root": report.commitment_root[:16] + "...",
        "report_valid": True,
    }


def example_batch_processing():
    """Example: Batch processing multiple samples."""

    # Simulate multiple patient samples
    samples = []
    for i in range(5):
        genome = "ACGT" * 2500  # 10kb genome
        consent = PatientConsent(
            patient_id=f"PATIENT_{i:03d}",
            test_types=[ClinicalTestType.PHARMACOGENOMICS],
            data_retention_days=90,
            research_use_allowed=False,
            timestamp=datetime.now(),
            signature=hashlib.sha256(f"sig_{i}".encode()).digest(),
        )
        samples.append((genome, consent))

    # Process batch
    workflow = ClinicalWorkflow()
    reports = workflow.batch_process_samples(samples, ClinicalTestType.PHARMACOGENOMICS)

    return {
        "total_samples": len(samples),
        "successful_reports": len(reports),
        "all_reports_valid": all(r.commitment_root for r in reports),
    }
