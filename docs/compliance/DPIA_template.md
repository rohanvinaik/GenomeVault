# Data Protection Impact Assessment (DPIA) Template
## GenomeVault Genomic Data Processing

---

**Document Information:**
- Template Version: 2.0
- Last Updated: {{assessment_date}}
- Next Review: {{next_review_date}}
- Classification: Confidential
- Prepared by: {{assessor_name}}
- Reviewed by: {{reviewer_name}}
- Approved by: Data Protection Officer

---

## Executive Summary

**Project Name:** {{project_name}}
**Assessment Date:** {{assessment_date}}
**Risk Level:** {{overall_risk_level}}
**Recommendation:** {{executive_recommendation}}

**Key Findings:**
- {{key_finding_1}}
- {{key_finding_2}}
- {{key_finding_3}}

**Required Actions:**
1. {{required_action_1}}
2. {{required_action_2}}
3. {{required_action_3}}

---

## 1. DESCRIPTION OF PROCESSING

### 1.1 Processing Overview

**Purpose of Processing:**
{{processing_purpose}}

**Legal Basis (GDPR Article 6):**
- [ ] Consent (6.1.a)
- [ ] Contract (6.1.b)
- [ ] Legal obligation (6.1.c)
- [ ] Vital interests (6.1.d)
- [ ] Public task (6.1.e)
- [ ] Legitimate interests (6.1.f)

**Special Category Legal Basis (GDPR Article 9):**
- [ ] Explicit consent (9.2.a)
- [ ] Employment/social security (9.2.b)
- [ ] Vital interests (9.2.c)
- [ ] Legitimate activities (9.2.d)
- [ ] Made public by data subject (9.2.e)
- [ ] Legal claims (9.2.f)
- [ ] Substantial public interest (9.2.g)
- [ ] Health/medicine (9.2.h)
- [ ] Public health (9.2.i)
- [ ] Research/statistics (9.2.j)

**Detailed Legal Basis Justification:**
{{legal_basis_justification}}

### 1.2 Data Categories

**Personal Data Categories:**
- [ ] Genetic data (DNA sequences, variants, SNPs)
- [ ] Health data (clinical diagnoses, medications, test results)
- [ ] Biometric data (fingerprints, voiceprints)
- [ ] Contact information (name, address, email, phone)
- [ ] Demographic data (age, gender, ethnicity)
- [ ] Family history data
- [ ] Lifestyle data (diet, exercise, habits)
- [ ] Insurance information
- [ ] Other: {{other_data_categories}}

**Sensitive Data Details:**
{{sensitive_data_details}}

### 1.3 Data Subjects

**Categories of Data Subjects:**
- [ ] Patients receiving clinical genomic testing
- [ ] Research participants
- [ ] Healthcare providers
- [ ] Family members (for hereditary analysis)
- [ ] Population study participants
- [ ] Other: {{other_data_subjects}}

**Vulnerable Groups:**
- [ ] Children (under 18)
- [ ] Elderly individuals
- [ ] Individuals with mental health conditions
- [ ] Individuals with rare diseases
- [ ] Economically disadvantaged populations

**Estimated Number of Data Subjects:** {{data_subject_count}}

### 1.4 Processing Activities

**Primary Processing Activities:**

1. **Data Collection:**
   - Source: {{data_collection_source}}
   - Method: {{data_collection_method}}
   - Frequency: {{data_collection_frequency}}

2. **Genomic Analysis:**
   - Hypervector encoding of genomic variants
   - Similarity analysis using Hamming distances
   - Clinical interpretation using KAN networks
   - Risk assessment and scoring

3. **Data Storage:**
   - Primary location: {{primary_storage_location}}
   - Backup location: {{backup_storage_location}}
   - Retention period: {{data_retention_period}}

4. **Data Sharing:**
   - Healthcare providers: {{healthcare_sharing}}
   - Research institutions: {{research_sharing}}
   - Commercial partners: {{commercial_sharing}}

5. **Automated Decision Making:**
   - Risk scoring algorithms: {{risk_scoring_details}}
   - Clinical recommendations: {{clinical_recommendations}}
   - Research matching: {{research_matching}}

### 1.5 Technology Architecture

**Core Technologies:**
- **Hyperdimensional Computing:** Privacy-preserving genomic vector encoding
- **Private Information Retrieval:** Multi-server PIR for query privacy
- **Zero-Knowledge Proofs:** Computation verification without data exposure
- **Differential Privacy:** Statistical privacy with ε-budget tracking
- **Homomorphic Encryption:** Computation on encrypted data

**Infrastructure:**
- Cloud provider: {{cloud_provider}}
- Data centers: {{data_center_locations}}
- Network architecture: {{network_architecture}}
- Access controls: {{access_control_system}}

---

## 2. STAKEHOLDER CONSULTATION

### 2.1 Internal Stakeholders

| Stakeholder | Role | Consultation Date | Key Concerns | Input Incorporated |
|-------------|------|-------------------|--------------|-------------------|
| {{stakeholder_1}} | {{role_1}} | {{date_1}} | {{concerns_1}} | {{input_1}} |
| {{stakeholder_2}} | {{role_2}} | {{date_2}} | {{concerns_2}} | {{input_2}} |
| {{stakeholder_3}} | {{role_3}} | {{date_3}} | {{concerns_3}} | {{input_3}} |

### 2.2 External Stakeholders

| Stakeholder | Organization | Consultation Method | Date | Input Summary |
|-------------|--------------|-------------------|------|---------------|
| {{ext_stakeholder_1}} | {{org_1}} | {{method_1}} | {{ext_date_1}} | {{ext_input_1}} |
| {{ext_stakeholder_2}} | {{org_2}} | {{method_2}} | {{ext_date_2}} | {{ext_input_2}} |

### 2.3 Data Subject Input

**Consultation Method:** {{data_subject_consultation_method}}

**Key Feedback:**
- {{feedback_1}}
- {{feedback_2}}
- {{feedback_3}}

**Privacy Concerns Raised:**
1. {{privacy_concern_1}}
2. {{privacy_concern_2}}
3. {{privacy_concern_3}}

**Design Changes Based on Feedback:**
- {{design_change_1}}
- {{design_change_2}}

---

## 3. NECESSITY AND PROPORTIONALITY

### 3.1 Necessity Assessment

**Is the processing necessary for the stated purpose?**
{{necessity_assessment}}

**Could the purpose be achieved with less intrusive means?**
{{alternative_means_assessment}}

**Alternative Processing Methods Considered:**
1. {{alternative_1}} - Rejected because: {{rejection_reason_1}}
2. {{alternative_2}} - Rejected because: {{rejection_reason_2}}
3. {{alternative_3}} - Rejected because: {{rejection_reason_3}}

### 3.2 Proportionality Assessment

**Balancing Test:**

| Factor | Weight (1-5) | Score (1-5) | Weighted Score | Justification |
|--------|--------------|-------------|----------------|---------------|
| Processing benefit | {{benefit_weight}} | {{benefit_score}} | {{benefit_weighted}} | {{benefit_justification}} |
| Privacy intrusion | {{intrusion_weight}} | {{intrusion_score}} | {{intrusion_weighted}} | {{intrusion_justification}} |
| Data subject rights | {{rights_weight}} | {{rights_score}} | {{rights_weighted}} | {{rights_justification}} |
| Societal benefit | {{society_weight}} | {{society_score}} | {{society_weighted}} | {{society_justification}} |

**Overall Proportionality Score:** {{proportionality_score}} / 25
**Assessment:** {{proportionality_assessment}}

### 3.3 Data Minimization

**Data Minimization Measures:**
- [ ] Only essential genomic variants processed
- [ ] Clinical data limited to relevant conditions
- [ ] Demographic data aggregated where possible
- [ ] Automatic deletion of temporary processing data
- [ ] Anonymization where identification not required

**Justification for Each Data Category:**
{{data_minimization_justification}}

---

## 4. PRIVACY RISKS ASSESSMENT

### 4.1 Risk Identification

**Risk Matrix:**

| Risk ID | Risk Description | Likelihood (1-5) | Impact (1-5) | Risk Score | Category |
|---------|------------------|------------------|--------------|------------|----------|
| R001 | {{risk_1_description}} | {{risk_1_likelihood}} | {{risk_1_impact}} | {{risk_1_score}} | {{risk_1_category}} |
| R002 | {{risk_2_description}} | {{risk_2_likelihood}} | {{risk_2_impact}} | {{risk_2_score}} | {{risk_2_category}} |
| R003 | {{risk_3_description}} | {{risk_3_likelihood}} | {{risk_3_impact}} | {{risk_3_score}} | {{risk_3_category}} |
| R004 | {{risk_4_description}} | {{risk_4_likelihood}} | {{risk_4_impact}} | {{risk_4_score}} | {{risk_4_category}} |
| R005 | {{risk_5_description}} | {{risk_5_likelihood}} | {{risk_5_impact}} | {{risk_5_score}} | {{risk_5_category}} |

### 4.2 Genomic-Specific Risks

**Genetic Discrimination:**
- Risk Level: {{genetic_discrimination_risk}}
- Mitigation: {{genetic_discrimination_mitigation}}

**Re-identification Risk:**
- Risk Level: {{reidentification_risk}}
- Population size: {{population_size}}
- Uniqueness of genetic markers: {{uniqueness_assessment}}
- Mitigation: {{reidentification_mitigation}}

**Family Privacy Impact:**
- Risk Level: {{family_privacy_risk}}
- Affected family members: {{affected_family_count}}
- Mitigation: {{family_privacy_mitigation}}

**Insurance/Employment Discrimination:**
- Risk Level: {{discrimination_risk}}
- Legal protections: {{legal_protections}}
- Mitigation: {{discrimination_mitigation}}

### 4.3 Technical Risks

**Data Breach Risk:**
- Risk Level: {{breach_risk}}
- Attack vectors: {{attack_vectors}}
- Potential impact: {{breach_impact}}
- Mitigation: {{breach_mitigation}}

**Inference Attack Risk:**
- Risk Level: {{inference_risk}}
- Attack scenarios: {{inference_scenarios}}
- Mitigation: {{inference_mitigation}}

**System Compromise Risk:**
- Risk Level: {{system_compromise_risk}}
- Critical components: {{critical_components}}
- Mitigation: {{system_mitigation}}

### 4.4 Legal/Regulatory Risks

**GDPR Compliance Risks:**
{{gdpr_compliance_risks}}

**HIPAA Compliance Risks:**
{{hipaa_compliance_risks}}

**Cross-Border Transfer Risks:**
{{transfer_risks}}

---

## 5. PRIVACY SAFEGUARDS

### 5.1 Technical Safeguards

**Encryption:**
- Data at rest: {{encryption_at_rest}}
- Data in transit: {{encryption_in_transit}}
- Key management: {{key_management}}

**Privacy-Preserving Technologies:**

1. **Hyperdimensional Computing:**
   - Vector dimension: {{hd_dimension}}
   - Encoding method: {{hd_encoding_method}}
   - Privacy guarantee: {{hd_privacy_guarantee}}

2. **Private Information Retrieval:**
   - Number of servers: {{pir_servers}}
   - Privacy threshold: {{pir_threshold}}
   - Query privacy: {{pir_query_privacy}}

3. **Zero-Knowledge Proofs:**
   - Proof system: {{zk_system}}
   - Verification method: {{zk_verification}}
   - Soundness parameter: {{zk_soundness}}

4. **Differential Privacy:**
   - Epsilon budget: {{dp_epsilon}}
   - Budget allocation: {{dp_allocation}}
   - Noise mechanism: {{dp_mechanism}}

**Access Controls:**
- Authentication: {{authentication_method}}
- Authorization: {{authorization_model}}
- Audit logging: {{audit_logging}}

### 5.2 Organizational Safeguards

**Data Governance:**
- Data steward: {{data_steward}}
- Data protection officer: {{dpo}}
- Privacy committee: {{privacy_committee}}

**Staff Training:**
- Privacy training frequency: {{training_frequency}}
- Specialized training: {{specialized_training}}
- Training records: {{training_records}}

**Policies and Procedures:**
- Data protection policy: {{data_protection_policy}}
- Incident response plan: {{incident_response_plan}}
- Data breach procedures: {{breach_procedures}}

### 5.3 Legal Safeguards

**Contractual Protections:**
- Data processing agreements: {{dpa_status}}
- Business associate agreements: {{baa_status}}
- Standard contractual clauses: {{scc_status}}

**Consent Management:**
- Consent collection method: {{consent_collection}}
- Consent withdrawal process: {{consent_withdrawal}}
- Consent records retention: {{consent_retention}}

**Data Subject Rights:**
- Access request process: {{access_process}}
- Rectification process: {{rectification_process}}
- Erasure process: {{erasure_process}}
- Portability process: {{portability_process}}

---

## 6. RESIDUAL RISKS

### 6.1 Risk Assessment After Safeguards

| Risk ID | Original Score | Residual Score | Risk Reduction | Justification |
|---------|----------------|----------------|----------------|---------------|
| R001 | {{orig_score_1}} | {{residual_score_1}} | {{reduction_1}} | {{justification_1}} |
| R002 | {{orig_score_2}} | {{residual_score_2}} | {{reduction_2}} | {{justification_2}} |
| R003 | {{orig_score_3}} | {{residual_score_3}} | {{reduction_3}} | {{justification_3}} |
| R004 | {{orig_score_4}} | {{residual_score_4}} | {{reduction_4}} | {{justification_4}} |
| R005 | {{orig_score_5}} | {{residual_score_5}} | {{reduction_5}} | {{justification_5}} |

### 6.2 Acceptable Risk Level

**Risk Tolerance:** {{risk_tolerance}}

**High Residual Risks (Score ≥ 15):**
{{high_residual_risks}}

**Risk Acceptance Justification:**
{{risk_acceptance_justification}}

### 6.3 Ongoing Risk Monitoring

**Risk Monitoring Plan:**
- Review frequency: {{risk_review_frequency}}
- Monitoring metrics: {{monitoring_metrics}}
- Escalation triggers: {{escalation_triggers}}
- Responsibility: {{monitoring_responsibility}}

---

## 7. RECOMMENDATIONS AND APPROVAL

### 7.1 Recommendations

**Primary Recommendation:** {{primary_recommendation}}

**Specific Actions Required:**

1. **High Priority (Complete before processing):**
   - {{high_priority_1}}
   - {{high_priority_2}}
   - {{high_priority_3}}

2. **Medium Priority (Complete within 30 days):**
   - {{medium_priority_1}}
   - {{medium_priority_2}}
   - {{medium_priority_3}}

3. **Low Priority (Complete within 90 days):**
   - {{low_priority_1}}
   - {{low_priority_2}}
   - {{low_priority_3}}

### 7.2 Alternative Recommendations

**If High-Risk Rating:**
- [ ] Redesign processing to reduce risk
- [ ] Implement additional safeguards
- [ ] Obtain supervisory authority consultation
- [ ] Delay processing until risks mitigated

**If Medium-Risk Rating:**
- [ ] Implement recommended safeguards
- [ ] Enhanced monitoring required
- [ ] Regular review schedule

**If Low-Risk Rating:**
- [ ] Standard safeguards sufficient
- [ ] Normal review schedule

### 7.3 Approval Decision

**Decision:**
- [ ] Approve processing as proposed
- [ ] Approve with conditions (see recommendations)
- [ ] Reject - risks too high
- [ ] Defer - additional assessment required

**Conditions (if applicable):**
{{approval_conditions}}

**Review Schedule:**
- Initial review: {{initial_review_date}}
- Ongoing reviews: {{ongoing_review_frequency}}
- Next scheduled review: {{next_review_date}}

---

## 8. SIGNATURES AND APPROVALS

**Prepared by:**
Name: {{assessor_name}}
Title: {{assessor_title}}
Date: {{assessment_date}}
Signature: _________________________

**Technical Review:**
Name: {{technical_reviewer}}
Title: Chief Technology Officer
Date: {{technical_review_date}}
Signature: _________________________

**Privacy Review:**
Name: {{privacy_reviewer}}
Title: Data Protection Officer
Date: {{privacy_review_date}}
Signature: _________________________

**Final Approval:**
Name: {{approver_name}}
Title: {{approver_title}}
Date: {{approval_date}}
Signature: _________________________

---

## 9. APPENDICES

### Appendix A: Technical Architecture Diagrams
{{technical_diagrams}}

### Appendix B: Data Flow Diagrams
{{data_flow_diagrams}}

### Appendix C: Risk Assessment Methodology
{{risk_methodology}}

### Appendix D: Stakeholder Consultation Records
{{consultation_records}}

### Appendix E: Legal Analysis
{{legal_analysis}}

### Appendix F: Privacy Engineering Requirements
{{privacy_engineering_requirements}}

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | {{version_1_date}} | {{version_1_author}} | Initial draft |
| 1.1 | {{version_1_1_date}} | {{version_1_1_author}} | Technical review comments |
| 2.0 | {{version_2_date}} | {{version_2_author}} | Final version |

**Distribution List:**
- Data Protection Officer
- Chief Technology Officer
- Privacy Committee
- Legal Counsel
- Project Team

**Retention:** 7 years after end of processing activities
**Classification:** Confidential - Internal Use Only

---

*This DPIA template complies with GDPR Article 35 requirements and ICO guidance. It should be customized for each specific processing activity and reviewed by qualified privacy professionals.*
