# Clinical Use Cases and Examples

This guide demonstrates how to use GenomeVault for clinical genomic applications while maintaining HIPAA compliance and patient privacy.

## Overview

GenomeVault enables clinical genomics applications through:
- **HIPAA-compliant processing**: All PHI handling meets healthcare regulations
- **Privacy-preserving analysis**: Mathematical privacy guarantees protect patient data
- **Audit trails**: Comprehensive logging for regulatory compliance
- **Clinical-grade accuracy**: Validated algorithms for diagnostic applications

## Clinical Workflow Integration

### Electronic Health Record (EHR) Integration

```python
import hashlib
from genomevault_sdk import GenomeVaultClient
from genomevault_sdk.clinical import ClinicalVariant

class EHRIntegration:
    def __init__(self, api_key: str):
        self.client = GenomeVaultClient(api_key=api_key)

    def process_patient_variants(self, patient_id: str, vcf_data: str, consent_id: str):
        """Process patient genomic data with full privacy protection."""

        # Hash patient ID for privacy (never send raw patient IDs)
        patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()
        consent_hash = hashlib.sha256(consent_id.encode()).hexdigest()

        # Parse VCF and extract clinical variants
        variants = self.parse_clinical_vcf(vcf_data)

        # Perform privacy-preserving clinical analysis
        return await self.client.clinical_analysis(
            patient_id_hash=patient_hash,
            variants=variants,
            analysis_type="diagnostic",
            consent_hash=consent_hash,
            population_reference="gnomAD"
        )

    def parse_clinical_vcf(self, vcf_data: str) -> List[ClinicalVariant]:
        """Extract clinically relevant variants from VCF."""
        variants = []

        # Focus on clinically actionable genes
        actionable_genes = [
            "BRCA1", "BRCA2", "TP53", "PALB2", "CHEK2",
            "ATM", "CDH1", "PTEN", "STK11", "MLH1", "MSH2"
        ]

        for variant in parse_vcf(vcf_data):
            if variant.gene in actionable_genes:
                clinical_variant = ClinicalVariant(
                    gene=variant.gene,
                    variant=variant.hgvs_notation,
                    classification=variant.clinical_significance,
                    evidence_level=variant.evidence_level
                )
                variants.append(clinical_variant)

        return variants
```

## Use Case 1: Hereditary Cancer Risk Assessment

### Scenario
A patient has a family history of breast and ovarian cancer. The clinician needs to assess hereditary cancer risk while protecting patient privacy.

```python
async def hereditary_cancer_assessment():
    """Assess hereditary cancer risk with privacy protection."""

    client = GenomeVaultClient(api_key="your-clinical-api-key")

    # Patient variants (BRCA1/2, other hereditary cancer genes)
    variants = [
        ClinicalVariant(
            gene="BRCA1",
            variant="c.68_69delAG",
            classification="pathogenic",
            evidence_level="A"
        ),
        ClinicalVariant(
            gene="BRCA2",
            variant="c.5946delT",
            classification="pathogenic",
            evidence_level="A"
        ),
        ClinicalVariant(
            gene="PALB2",
            variant="c.3113G>A",
            classification="likely_pathogenic",
            evidence_level="B"
        )
    ]

    # Generate patient ID hash
    patient_id_hash = hashlib.sha256("PATIENT_12345".encode()).hexdigest()
    consent_hash = hashlib.sha256("CONSENT_67890".encode()).hexdigest()

    # Perform privacy-preserving risk assessment
    analysis = await client.clinical_analysis(
        patient_id_hash=patient_id_hash,
        variants=variants,
        analysis_type="risk_assessment",
        population_reference="gnomAD",
        consent_hash=consent_hash
    )

    # Process results for clinical decision support
    if analysis.risk_score > 0.8:
        recommendations = [
            "Consider genetic counseling referral",
            "Enhanced breast cancer screening (MRI + mammography)",
            "Consider prophylactic interventions",
            "Cascade testing for family members"
        ]
    elif analysis.risk_score > 0.5:
        recommendations = [
            "Genetic counseling recommended",
            "Enhanced screening protocol",
            "Regular follow-up assessments"
        ]
    else:
        recommendations = [
            "Standard screening guidelines apply",
            "Reassess if family history changes"
        ]

    return {
        "analysis_id": analysis.analysis_id,
        "risk_score": analysis.risk_score,
        "confidence_interval": analysis.confidence_interval,
        "clinical_recommendations": recommendations,
        "audit_trail": analysis.audit_trail_hash
    }

# Execute assessment
result = await hereditary_cancer_assessment()
print(f"Patient risk score: {result['risk_score']:.2%}")
print("Recommendations:")
for rec in result['clinical_recommendations']:
    print(f"  • {rec}")
```

### Privacy Protections
- Patient ID hashed before transmission
- Variants encoded using HDC for privacy
- Risk analysis uses differential privacy
- Audit trail cryptographically signed
- No raw genomic data stored on servers

## Use Case 2: Pharmacogenomics Decision Support

### Scenario
A patient is being prescribed warfarin. The system needs to determine optimal dosing based on genetic variants while maintaining privacy.

```python
async def pharmacogenomics_dosing():
    """Determine optimal drug dosing with privacy protection."""

    client = GenomeVaultClient(api_key="your-clinical-api-key")

    # Pharmacogenomically relevant variants
    pgx_variants = [
        ClinicalVariant(
            gene="CYP2C9",
            variant="*2/*3",
            classification="significant",
            evidence_level="A"
        ),
        ClinicalVariant(
            gene="VKORC1",
            variant="-1639G>A",
            classification="significant",
            evidence_level="A"
        ),
        ClinicalVariant(
            gene="CYP4F2",
            variant="V433M",
            classification="moderate",
            evidence_level="B"
        )
    ]

    patient_hash = hashlib.sha256("PATIENT_54321".encode()).hexdigest()

    # Perform pharmacogenomic analysis
    analysis = await client.clinical_analysis(
        patient_id_hash=patient_hash,
        variants=pgx_variants,
        analysis_type="pharmacogenomics",
        population_reference="gnomAD"
    )

    # Calculate warfarin dosing recommendation
    # This uses privacy-preserving algorithms that don't expose variant details
    base_dose = 5.0  # mg/day

    # Apply genetic factors (algorithm runs on encrypted data)
    if analysis.risk_score < 0.3:
        recommended_dose = base_dose * 0.5  # Sensitive metabolizer
        monitoring = "Frequent INR monitoring required"
    elif analysis.risk_score < 0.7:
        recommended_dose = base_dose * 0.75  # Intermediate metabolizer
        monitoring = "Standard INR monitoring"
    else:
        recommended_dose = base_dose * 1.0  # Normal metabolizer
        monitoring = "Standard INR monitoring"

    return {
        "drug": "warfarin",
        "recommended_dose": recommended_dose,
        "confidence": analysis.confidence_interval,
        "monitoring_plan": monitoring,
        "analysis_id": analysis.analysis_id
    }

result = await pharmacogenomics_dosing()
print(f"Recommended warfarin dose: {result['recommended_dose']:.1f} mg/day")
print(f"Monitoring plan: {result['monitoring_plan']}")
```

## Use Case 3: Rare Disease Diagnosis

### Scenario
A patient presents with symptoms suggesting a rare genetic disorder. The system needs to analyze variants across multiple genes while protecting patient privacy.

```python
async def rare_disease_analysis():
    """Analyze variants for rare disease diagnosis."""

    client = GenomeVaultClient(api_key="your-clinical-api-key")

    # Variants from whole exome sequencing
    candidate_variants = [
        ClinicalVariant(
            gene="CFTR",
            variant="p.Phe508del",
            classification="pathogenic",
            evidence_level="A"
        ),
        ClinicalVariant(
            gene="CFTR",
            variant="p.Gly542X",
            classification="pathogenic",
            evidence_level="A"
        ),
        ClinicalVariant(
            gene="SPINK1",
            variant="p.Asn34Ser",
            classification="uncertain_significance",
            evidence_level="C"
        )
    ]

    patient_hash = hashlib.sha256("RARE_DISEASE_PATIENT_789".encode()).hexdigest()

    # Perform comprehensive diagnostic analysis
    analysis = await client.clinical_analysis(
        patient_id_hash=patient_hash,
        variants=candidate_variants,
        analysis_type="diagnostic",
        population_reference="gnomAD"
    )

    # Interpret results for rare disease context
    diagnostic_confidence = analysis.risk_score

    if diagnostic_confidence > 0.9:
        diagnosis_status = "Confirmed"
        next_steps = [
            "Definitive genetic counseling",
            "Initiate disease-specific management",
            "Family cascade screening"
        ]
    elif diagnostic_confidence > 0.7:
        diagnosis_status = "Likely"
        next_steps = [
            "Additional functional studies recommended",
            "Genetic counseling advised",
            "Consider confirmatory testing"
        ]
    else:
        diagnosis_status = "Uncertain"
        next_steps = [
            "Review clinical phenotype",
            "Consider additional genetic testing",
            "Reassess in light of new evidence"
        ]

    return {
        "diagnosis_status": diagnosis_status,
        "confidence": diagnostic_confidence,
        "confidence_interval": analysis.confidence_interval,
        "next_steps": next_steps,
        "analysis_id": analysis.analysis_id
    }

result = await rare_disease_analysis()
print(f"Diagnosis: {result['diagnosis_status']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## Use Case 4: Population Health Screening

### Scenario
A healthcare system wants to identify patients at risk for hereditary conditions for proactive screening while maintaining patient privacy at scale.

```python
async def population_health_screening():
    """Screen population for hereditary disease risk."""

    client = GenomeVaultClient(api_key="your-clinical-api-key")

    # Batch processing for population screening
    screening_batches = []

    # Example: Screen 10,000 patients for Lynch syndrome
    for patient_batch in get_patient_batches(batch_size=100):
        batch_results = []

        for patient in patient_batch:
            # Hash patient identifier
            patient_hash = hashlib.sha256(patient.id.encode()).hexdigest()

            # Extract Lynch syndrome variants
            lynch_variants = extract_lynch_variants(patient.vcf_data)

            if lynch_variants:
                # Analyze variants with privacy protection
                analysis = await client.clinical_analysis(
                    patient_id_hash=patient_hash,
                    variants=lynch_variants,
                    analysis_type="carrier_screening",
                    population_reference="gnomAD"
                )

                batch_results.append({
                    "patient_hash": patient_hash,
                    "risk_score": analysis.risk_score,
                    "analysis_id": analysis.analysis_id
                })

        screening_batches.append(batch_results)

    # Aggregate results for population insights
    high_risk_patients = []
    for batch in screening_batches:
        for result in batch:
            if result["risk_score"] > 0.8:  # High risk threshold
                high_risk_patients.append(result)

    return {
        "total_screened": sum(len(batch) for batch in screening_batches),
        "high_risk_count": len(high_risk_patients),
        "screening_rate": len(high_risk_patients) / total_screened * 100,
        "privacy_preserved": True  # All analyses used encrypted patient IDs
    }

def extract_lynch_variants(vcf_data: str) -> List[ClinicalVariant]:
    """Extract variants in Lynch syndrome genes."""
    lynch_genes = ["MLH1", "MSH2", "MSH6", "PMS2", "EPCAM"]
    variants = []

    for variant in parse_vcf(vcf_data):
        if variant.gene in lynch_genes and variant.is_pathogenic():
            variants.append(ClinicalVariant(
                gene=variant.gene,
                variant=variant.hgvs,
                classification=variant.classification,
                evidence_level=variant.evidence
            ))

    return variants

result = await population_health_screening()
print(f"Screened {result['total_screened']} patients")
print(f"Identified {result['high_risk_count']} high-risk patients ({result['screening_rate']:.1f}%)")
```

## Clinical Decision Support Integration

### CDS Hooks Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
client = GenomeVaultClient(api_key="clinical-cds-key")

@app.route('/cds-services/genomic-risk-assessment', methods=['POST'])
async def genomic_cds_hook():
    """CDS Hook for genomic risk assessment."""

    hook_data = request.get_json()

    # Extract patient context from CDS Hook
    patient_id = hook_data['context']['patientId']
    patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()

    # Get genetic data from context (if available)
    genetic_data = hook_data.get('context', {}).get('genetic_variants', [])

    if genetic_data:
        # Convert to clinical variants
        variants = [
            ClinicalVariant(**variant) for variant in genetic_data
        ]

        # Perform privacy-preserving analysis
        analysis = await client.clinical_analysis(
            patient_id_hash=patient_hash,
            variants=variants,
            analysis_type="risk_assessment"
        )

        # Generate CDS cards based on results
        cards = []

        if analysis.risk_score > 0.8:
            cards.append({
                "summary": "High genetic risk detected",
                "detail": f"Patient has {analysis.risk_score:.0%} risk based on genetic variants",
                "indicator": "critical",
                "source": {
                    "label": "GenomeVault Clinical Analytics",
                    "url": f"https://genomevault.io/analysis/{analysis.analysis_id}"
                },
                "suggestions": [
                    {
                        "label": "Order genetic counseling",
                        "actions": [{
                            "type": "create",
                            "description": "Genetic counseling referral",
                            "resource": generate_counseling_order()
                        }]
                    }
                ]
            })

    return jsonify({"cards": cards})

if __name__ == '__main__':
    app.run(debug=True)
```

## Compliance and Audit Features

### HIPAA Compliance Monitoring

```python
async def hipaa_compliance_report():
    """Generate HIPAA compliance report for genomic analyses."""

    client = GenomeVaultClient(api_key="compliance-api-key")

    # Get compliance metrics
    report = await client.get_compliance_report(
        start_date="2024-01-01",
        end_date="2024-01-31",
        report_type="hipaa"
    )

    compliance_metrics = {
        "total_analyses": report.total_analyses,
        "phi_exposure_incidents": report.phi_incidents,
        "encryption_compliance": report.encryption_rate,
        "access_log_integrity": report.audit_trail_integrity,
        "patient_consent_verification": report.consent_verification_rate
    }

    # Check for compliance violations
    violations = []
    if report.phi_incidents > 0:
        violations.append(f"{report.phi_incidents} PHI exposure incidents detected")

    if report.encryption_rate < 1.0:
        violations.append(f"Encryption compliance: {report.encryption_rate:.1%}")

    return {
        "compliance_status": "COMPLIANT" if not violations else "NON-COMPLIANT",
        "metrics": compliance_metrics,
        "violations": violations,
        "audit_period": "January 2024"
    }

# Generate monthly compliance report
compliance = await hipaa_compliance_report()
print(f"HIPAA Compliance Status: {compliance['compliance_status']}")
```

### Audit Trail Verification

```python
async def verify_clinical_audit_trail(analysis_id: str):
    """Verify the integrity of clinical analysis audit trail."""

    client = GenomeVaultClient(api_key="audit-api-key")

    # Retrieve audit trail
    audit = await client.get_audit_trail(analysis_id)

    # Verify cryptographic signatures
    verification_results = {
        "timestamp_valid": verify_timestamp(audit.timestamp),
        "signature_valid": verify_signature(audit.signature, audit.data),
        "chain_integrity": verify_chain_integrity(audit.chain_hash),
        "privacy_parameters": audit.privacy_parameters,
        "compliance_flags": audit.compliance_flags
    }

    return verification_results

# Verify specific analysis
audit_result = await verify_clinical_audit_trail("analysis_12345")
print(f"Audit trail valid: {all(audit_result.values())}")
```

## Best Practices for Clinical Implementation

### 1. Patient Consent Management
```python
class ConsentManager:
    def __init__(self, client: GenomeVaultClient):
        self.client = client

    async def verify_consent(self, patient_id: str, analysis_type: str) -> bool:
        """Verify patient has consented to specific analysis type."""
        patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()

        consent = await self.client.get_patient_consent(patient_hash)
        return analysis_type in consent.authorized_analyses

    async def record_consent(self, patient_id: str, consent_form_id: str,
                           authorized_analyses: List[str]):
        """Record patient consent with cryptographic verification."""
        patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()
        consent_hash = hashlib.sha256(consent_form_id.encode()).hexdigest()

        await self.client.record_consent(
            patient_hash=patient_hash,
            consent_hash=consent_hash,
            authorized_analyses=authorized_analyses,
            timestamp=datetime.utcnow()
        )
```

### 2. Quality Control
```python
async def clinical_quality_control(variants: List[ClinicalVariant]) -> bool:
    """Perform quality control checks on clinical variants."""

    qc_checks = {
        "variant_format": all(validate_hgvs(v.variant) for v in variants),
        "gene_symbols": all(validate_gene_symbol(v.gene) for v in variants),
        "classifications": all(v.classification in VALID_CLASSIFICATIONS for v in variants),
        "evidence_levels": all(v.evidence_level in ["A", "B", "C", "D"] for v in variants)
    }

    return all(qc_checks.values())
```

### 3. Result Interpretation
```python
def interpret_clinical_results(analysis_result) -> Dict[str, Any]:
    """Interpret clinical analysis results for healthcare providers."""

    interpretation = {
        "risk_category": categorize_risk(analysis_result.risk_score),
        "clinical_significance": assess_clinical_significance(analysis_result),
        "recommended_actions": generate_clinical_recommendations(analysis_result),
        "confidence_assessment": interpret_confidence_interval(analysis_result.confidence_interval),
        "follow_up_needed": analysis_result.risk_score > 0.7
    }

    return interpretation

def categorize_risk(risk_score: float) -> str:
    """Categorize genetic risk score for clinical use."""
    if risk_score >= 0.8:
        return "High Risk"
    elif risk_score >= 0.5:
        return "Moderate Risk"
    elif risk_score >= 0.2:
        return "Low-Moderate Risk"
    else:
        return "Low Risk"
```

## Integration with Clinical Systems

### HL7 FHIR Integration

```python
from fhir.resources.observation import Observation
from fhir.resources.patient import Patient

def create_genomic_observation(analysis_result, patient_fhir_id: str) -> Observation:
    """Create HL7 FHIR Observation for genomic analysis result."""

    observation = Observation(
        status="final",
        category=[{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "survey",
                "display": "Survey"
            }]
        }],
        code={
            "coding": [{
                "system": "http://loinc.org",
                "code": "81247-9",
                "display": "Master HL7 genetic variant reporting panel"
            }]
        },
        subject={"reference": f"Patient/{patient_fhir_id}"},
        valueQuantity={
            "value": analysis_result.risk_score,
            "unit": "probability",
            "system": "http://unitsofmeasure.org"
        },
        component=[
            {
                "code": {
                    "coding": [{
                        "system": "http://genomevault.io/terms",
                        "code": "privacy-level",
                        "display": "Privacy Level"
                    }]
                },
                "valueString": analysis_result.privacy_level
            }
        ]
    )

    return observation
```

## Troubleshooting Clinical Workflows

### Common Issues and Solutions

```python
async def troubleshoot_clinical_analysis(error_code: str, context: Dict):
    """Troubleshoot common clinical analysis issues."""

    troubleshooting_guide = {
        "GV_CONSENT_REQUIRED": {
            "cause": "Patient consent not found or expired",
            "solution": "Verify patient consent status and re-obtain if necessary",
            "action": "consent_verification"
        },
        "GV_CLINICAL_DATA_INCOMPLETE": {
            "cause": "Required clinical variant information missing",
            "solution": "Ensure all variants have gene, HGVS notation, and classification",
            "action": "data_validation"
        },
        "GV_PHI_DETECTED": {
            "cause": "Protected health information in request",
            "solution": "Hash patient identifiers and remove direct PHI",
            "action": "phi_sanitization"
        }
    }

    if error_code in troubleshooting_guide:
        guide = troubleshooting_guide[error_code]
        return {
            "error": error_code,
            "cause": guide["cause"],
            "solution": guide["solution"],
            "recommended_action": guide["action"]
        }

    return {"error": "Unknown error code"}
```

## Performance Optimization

### Batch Clinical Processing

```python
async def batch_clinical_processing(patient_data: List[Dict]) -> List[Dict]:
    """Process multiple patients efficiently with privacy protection."""

    client = GenomeVaultClient(api_key="batch-clinical-key")

    # Prepare batch requests
    batch_requests = []
    for patient in patient_data:
        patient_hash = hashlib.sha256(patient["id"].encode()).hexdigest()

        request = {
            "patient_id_hash": patient_hash,
            "variants": patient["variants"],
            "analysis_type": patient.get("analysis_type", "risk_assessment"),
            "population_reference": patient.get("pop_ref", "gnomAD")
        }
        batch_requests.append(request)

    # Process in batches of 50 for optimal performance
    results = []
    for batch in chunk_list(batch_requests, 50):
        batch_results = await client.batch_clinical_analysis(batch)
        results.extend(batch_results)

    return results

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
```

This comprehensive guide demonstrates how GenomeVault can be integrated into clinical workflows while maintaining the highest standards of patient privacy and regulatory compliance. The privacy-preserving nature of the platform ensures that sensitive genomic information is protected throughout the analysis process.
