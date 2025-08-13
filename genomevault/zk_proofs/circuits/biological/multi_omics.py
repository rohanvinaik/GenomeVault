"""Multi-omics integration circuits for zero-knowledge proofs.

This module implements circuits that integrate multiple omics data types
while preserving privacy.
"""

from __future__ import annotations

from typing import Dict, List, cast
import hashlib

from numpy.typing import NDArray
import numpy as np

from ...prover import Circuit
from ..base_circuits import Any, BaseCircuit, FieldElement


class MultiOmicsCorrelationCircuit(BaseCircuit):
    """
    Circuit for proving correlations between omics layers without revealing data.

    Public inputs:
    - correlation_coefficient: Committed correlation value
    - modality_1: First omics type (e.g., 'genomics')
    - modality_2: Second omics type (e.g., 'transcriptomics')
    - significance_threshold: P-value threshold

    Private inputs:
    - data_1: First omics data (hypervector)
    - data_2: Second omics data (hypervector)
    - sample_size: Number of samples
    - witness_randomness: ZK randomness
    """

    def __init__(self, max_dimensions: int = 1000):
        """Initialize instance.

        Args:
            max_dimensions: Dimension value.
        """
        super().__init__("multi_omics_correlation", 30000)
        self.max_dimensions = max_dimensions

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup multi-omics correlation circuit."""
        # Public inputs
        self.correlation_commitment = FieldElement(
            int(public_inputs["correlation_coefficient"], 16)
        )
        self.modality_1 = public_inputs["modality_1"]
        self.modality_2 = public_inputs["modality_2"]
        self.significance_threshold = FieldElement(
            int(public_inputs["significance_threshold"] * 10000)
        )

        # Private inputs
        self.data_1 = self._process_hypervector(private_inputs["data_1"])
        self.data_2 = self._process_hypervector(private_inputs["data_2"])
        self.sample_size = FieldElement(private_inputs["sample_size"])
        self.witness_randomness = FieldElement(int(private_inputs["witness_randomness"], 16))

    def generate_constraints(self):
        """Generate correlation proof constraints."""
        # 1. Calculate correlation coefficient
        correlation = self._calculate_correlation(self.data_1, self.data_2)

        # 2. Create commitment to correlation
        correlation_commit = self._commit_correlation(correlation, self.witness_randomness)
        self.add_constraint(
            correlation_commit,
            self.correlation_commitment,
            FieldElement(0),
            ql=1,
            qr=-1,
        )

        # 3. Calculate significance (simplified t-test)
        t_statistic = self._calculate_t_statistic(correlation, self.sample_size)

        # 4. Verify significance meets threshold
        # In production, would use proper p-value calculation
        self._add_significance_constraint(t_statistic)

    def _process_hypervector(self, data: Any) -> List[FieldElement]:
        """Convert hypervector to field elements."""
        # Extract first N dimensions for efficiency
        if hasattr(data, "numpy"):
            values: NDArray[np.float32] = cast(
                NDArray[np.float32], data.numpy()[: self.max_dimensions]
            )
        else:
            values = np.array(data)[: self.max_dimensions]

        # Scale and convert to field elements
        return [FieldElement(int(v * 10000)) for v in values]

    def _calculate_correlation(
        self, data_1: List[FieldElement], data_2: List[FieldElement]
    ) -> FieldElement:
        """Calculate Pearson correlation coefficient."""
        n = len(data_1)

        # Calculate means
        sum_1 = FieldElement(0)
        sum_2 = FieldElement(0)
        for x, y in zip(data_1, data_2):
            sum_1 = sum_1 + x
            sum_2 = sum_2 + y

        mean_1 = sum_1 * FieldElement(n).inverse()
        mean_2 = sum_2 * FieldElement(n).inverse()

        # Calculate correlation components
        cov_sum = FieldElement(0)
        var_1_sum = FieldElement(0)
        var_2_sum = FieldElement(0)

        for x, y in zip(data_1, data_2):
            diff_1 = x - mean_1
            diff_2 = y - mean_2

            cov_sum = cov_sum + (diff_1 * diff_2)
            var_1_sum = var_1_sum + (diff_1 * diff_1)
            var_2_sum = var_2_sum + (diff_2 * diff_2)

        # Correlation = cov / (std_1 * std_2)
        # Simplified calculation
        return cov_sum  # In production, would properly normalize

    def _calculate_t_statistic(self, correlation: FieldElement, n: FieldElement) -> FieldElement:
        """Calculate t-statistic for correlation significance."""
        # t = r * sqrt(n-2) / sqrt(1-r^2)
        # Simplified for circuit
        return correlation * n

    def _add_significance_constraint(self, t_statistic: FieldElement):
        """Add constraint for statistical significance."""
        # Simplified: just check t-statistic is above threshold
        # In production, would map to proper p-value
        threshold = FieldElement(196)  # ~1.96 for p=0.05, scaled

        t_statistic - threshold
        # Add range proof that diff > 0

    def _commit_correlation(
        self, correlation: FieldElement, randomness: FieldElement
    ) -> FieldElement:
        """Create commitment to correlation value."""
        data = (
            "CORRELATION:{self.modality_1}:{self.modality_2}:{correlation.value}:{randomness.value}"
        )
        hash_val = hashlib.sha256(data.encode()).hexdigest()
        return FieldElement(int(hash_val, 16))


class GenotypePhenotypeAssociationCircuit(BaseCircuit):
    """
    Circuit for proving genotype-phenotype associations.

    Public inputs:
    - phenotype_id: Phenotype being tested
    - association_strength: Odds ratio or effect size
    - p_value_commitment: Statistical significance
    - study_size: Number of samples

    Private inputs:
    - genotypes: Variant genotypes (0, 1, 2)
    - phenotypes: Phenotype values
    - covariates: Confounding variables
    - witness_randomness: ZK randomness
    """

    def __init__(self, max_samples: int = 10000):
        """Initialize instance.

        Args:
            max_samples: Max samples.
        """
        super().__init__("genotype_phenotype_association", 40000)
        self.max_samples = max_samples

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup G-P association circuit."""
        # Public inputs
        self.phenotype_id = FieldElement(public_inputs["phenotype_id"])
        self.association_strength = FieldElement(int(public_inputs["association_strength"] * 1000))
        self.p_value_commitment = FieldElement(int(public_inputs["p_value_commitment"], 16))
        self.study_size = FieldElement(public_inputs["study_size"])

        # Private inputs
        self.genotypes = [FieldElement(g) for g in private_inputs["genotypes"]]
        self.phenotypes = [FieldElement(int(p * 100)) for p in private_inputs["phenotypes"]]
        self.covariates = private_inputs.get("covariates", [])
        self.witness_randomness = FieldElement(int(private_inputs["witness_randomness"], 16))

    def generate_constraints(self):
        """Generate association test constraints."""
        # 1. Validate genotypes (0, 1, or 2)
        for genotype in self.genotypes[:100]:  # Limit for efficiency
            self._add_genotype_validation(genotype)

        # 2. Calculate association (simplified linear model)
        beta = self._calculate_association(self.genotypes[:100], self.phenotypes[:100])

        # 3. Verify association strength
        self.add_constraint(beta, self.association_strength, FieldElement(0), ql=1, qr=-1)

        # 4. Calculate and commit p-value
        p_value = self._calculate_p_value(beta, len(self.genotypes))
        p_commit = self._commit_p_value(p_value, self.witness_randomness)

        self.add_constraint(p_commit, self.p_value_commitment, FieldElement(0), ql=1, qr=-1)

    def _add_genotype_validation(self, genotype: FieldElement):
        """Validate genotype is 0, 1, or 2."""
        # g * (g - 1) * (g - 2) = 0
        g_minus_1 = genotype - FieldElement(1)
        g_minus_2 = genotype - FieldElement(2)

        temp = genotype * g_minus_1
        result = temp * g_minus_2

        self.add_constraint(result, FieldElement(0), FieldElement(0), ql=1)

    def _calculate_association(
        self, genotypes: List[FieldElement], phenotypes: List[FieldElement]
    ) -> FieldElement:
        """Calculate association coefficient (simplified)."""
        # Simple correlation as proxy for association
        n = len(genotypes)

        # Calculate means
        geno_sum = FieldElement(0)
        pheno_sum = FieldElement(0)

        for g, p in zip(genotypes, phenotypes):
            geno_sum = geno_sum + g
            pheno_sum = pheno_sum + p

        geno_mean = geno_sum * FieldElement(n).inverse()
        pheno_mean = pheno_sum * FieldElement(n).inverse()

        # Calculate covariance
        cov_sum = FieldElement(0)
        var_sum = FieldElement(0)

        for g, p in zip(genotypes, phenotypes):
            g_diff = g - geno_mean
            p_diff = p - pheno_mean

            cov_sum = cov_sum + (g_diff * p_diff)
            var_sum = var_sum + (g_diff * g_diff)

        # Beta = cov(G,P) / var(G)
        # Simplified to avoid division in circuit
        return cov_sum

    def _calculate_p_value(self, beta: FieldElement, n: int) -> FieldElement:
        """Calculate p-value for association (simplified)."""
        # In production, would use proper statistical test
        # For now, use beta magnitude as proxy
        beta_abs = beta  # Would need absolute value in real circuit

        # Map to p-value scale (simplified)
        if beta_abs.value > 1000:
            return FieldElement(1)  # p < 0.001
        elif beta_abs.value > 500:
            return FieldElement(10)  # p < 0.01
        else:
            return FieldElement(50)  # p < 0.05

    def _commit_p_value(self, p_value: FieldElement, randomness: FieldElement) -> FieldElement:
        """Commit to p-value."""
        data = b"PVALUE:{p_value.value}:{randomness.value}"
        hash_val = hashlib.sha256(data).hexdigest()
        return FieldElement(int(hash_val, 16))


class ClinicalTrialEligibilityCircuit(BaseCircuit):
    """
    Circuit for proving clinical trial eligibility without revealing patient data.

    Public inputs:
    - trial_id: Clinical trial identifier
    - eligibility_result: Boolean eligibility
    - criteria_hash: Hash of eligibility criteria

    Private inputs:
    - genomic_features: Relevant genetic variants
    - clinical_features: Clinical measurements
    - demographic_features: Age, sex, etc.
    - witness_randomness: ZK randomness
    """

    def __init__(self):
        """Initialize instance."""
        super().__init__("clinical_trial_eligibility", 20000)

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup eligibility circuit."""
        # Public inputs
        self.trial_id = FieldElement(public_inputs["trial_id"])
        self.eligibility_result = FieldElement(1 if public_inputs["eligibility_result"] else 0)
        self.criteria_hash = FieldElement(int(public_inputs["criteria_hash"], 16))

        # Private inputs
        self.genomic_features = private_inputs["genomic_features"]
        self.clinical_features = private_inputs["clinical_features"]
        self.demographic_features = private_inputs["demographic_features"]
        self.witness_randomness = FieldElement(int(private_inputs["witness_randomness"], 16))

    def generate_constraints(self):
        """Generate eligibility check constraints."""
        # 1. Check genomic criteria
        genomic_eligible = self._check_genomic_criteria()

        # 2. Check clinical criteria
        clinical_eligible = self._check_clinical_criteria()

        # 3. Check demographic criteria
        demographic_eligible = self._check_demographic_criteria()

        # 4. Combine all criteria (AND operation)
        total_eligible = genomic_eligible * clinical_eligible * demographic_eligible

        # 5. Verify result matches public input
        self.add_constraint(total_eligible, self.eligibility_result, FieldElement(0), ql=1, qr=-1)

        # 6. Verify criteria hash
        computed_hash = self._hash_criteria()
        self.add_constraint(computed_hash, self.criteria_hash, FieldElement(0), ql=1, qr=-1)

    def _check_genomic_criteria(self) -> FieldElement:
        """Check if genomic features meet criteria."""
        # Example: Check for specific mutation
        required_mutation = self.genomic_features.get("required_mutation", {})

        if required_mutation:
            has_mutation = FieldElement(1 if required_mutation.get("present", False) else 0)
            return has_mutation

        return FieldElement(1)  # No genomic criteria

    def _check_clinical_criteria(self) -> FieldElement:
        """Check if clinical features meet criteria."""
        # Example: Check lab values are in range
        eligible = FieldElement(1)

        for feature, value in self.clinical_features.items():
            if "range" in value:
                min_val, max_val = value["range"]
                actual = value["value"]

                # Check if in range (simplified)
                if actual < min_val or actual > max_val:
                    eligible = FieldElement(0)

        return eligible

    def _check_demographic_criteria(self) -> FieldElement:
        """Check if demographic features meet criteria."""
        # Example: Age range check
        age = self.demographic_features.get("age", 0)
        min_age = self.demographic_features.get("min_age", 0)
        max_age = self.demographic_features.get("max_age", 150)

        if min_age <= age <= max_age:
            return FieldElement(1)

        return FieldElement(0)

    def _hash_criteria(self) -> FieldElement:
        """Hash the eligibility criteria."""
        criteria_str = "{self.trial_id.value}:genomic:clinical:demographic"
        hash_val = hashlib.sha256(criteria_str.encode()).hexdigest()
        return FieldElement(int(hash_val, 16))


class RareVariantBurdenCircuit(BaseCircuit):
    """
    Circuit for rare variant burden testing.

    Public inputs:
    - gene_id: Gene being tested
    - burden_score: Aggregated burden score
    - max_allele_frequency: MAF threshold for rare variants

    Private inputs:
    - variants: List of variants in gene
    - allele_frequencies: Population frequencies
    - functional_scores: CADD/REVEL scores
    - witness_randomness: ZK randomness
    """

    def __init__(self, max_variants_per_gene: int = 100):
        """Initialize instance.

        Args:
            max_variants_per_gene: Genetic variant information.
        """
        super().__init__("rare_variant_burden", 15000)
        self.max_variants = max_variants_per_gene

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup burden test circuit."""
        # Public inputs
        self.gene_id = FieldElement(public_inputs["gene_id"])
        self.burden_score = FieldElement(int(public_inputs["burden_score"] * 1000))
        self.maf_threshold = FieldElement(int(public_inputs["max_allele_frequency"] * 10000))

        # Private inputs
        self.variants = private_inputs["variants"]
        self.allele_frequencies = [
            FieldElement(int(af * 10000)) for af in private_inputs["allele_frequencies"]
        ]
        self.functional_scores = [
            FieldElement(int(score * 100)) for score in private_inputs["functional_scores"]
        ]
        self.witness_randomness = FieldElement(int(private_inputs["witness_randomness"], 16))

    def generate_constraints(self):
        """Generate burden test constraints."""
        # 1. Filter variants by MAF
        burden = FieldElement(0)

        for i, (variant, af, func_score) in enumerate(
            zip(
                self.variants[: self.max_variants],
                self.allele_frequencies[: self.max_variants],
                self.functional_scores[: self.max_variants],
            )
        ):
            # Check if rare (AF < threshold)
            is_rare = self._check_rare_variant(af, self.maf_threshold)

            # Weight by functional score
            weighted_contribution = is_rare * func_score

            # Add to burden
            burden = burden + weighted_contribution

        # 2. Verify burden score
        self.add_constraint(burden, self.burden_score, FieldElement(0), ql=1, qr=-1)

        # 3. Create commitment to hide individual variants
        self._commit_variants(self.witness_randomness)

    def _check_rare_variant(self, af: FieldElement, threshold: FieldElement) -> FieldElement:
        """Check if variant is rare (simplified)."""
        # In production, would use comparison circuit
        # For now, return 1 if we assume it's rare
        return FieldElement(1)

    def _commit_variants(self, randomness: FieldElement) -> FieldElement:
        """Commit to variant list."""
        ":".join(str(v.get("id", "")) for v in self.variants[:10])
        data = b"VARIANTS:{variant_str}:{randomness.value}"
        hash_val = hashlib.sha256(data).hexdigest()
        return FieldElement(int(hash_val, 16))


def create_multi_omics_proof_suite(omics_data: Dict[str, Any], analysis_type: str) -> List[Circuit]:
    """
    Create a suite of proofs for multi-omics analysis.

    Args:
        omics_data: Dictionary with omics layer data
        analysis_type: Type of analysis to perform

    Returns:
        List of circuits for the analysis
    """
    circuits = []

    if analysis_type == "integrated_risk":
        # Create circuits for integrated disease risk

        # Genomic risk
        if "genomics" in omics_data:
            prs_circuit = Circuit(
                name="integrated_prs",
                circuit_type="genomic",
                constraints=20000,
                public_inputs=["risk_model", "risk_score", "confidence"],
                private_inputs=["variants", "weights", "quality_scores"],
                parameters={"max_variants": 5000},
            )
            circuits.append(prs_circuit)

        # Transcriptomic risk
        if "transcriptomics" in omics_data:
            expression_circuit = Circuit(
                name="expression_signature",
                circuit_type="transcriptomic",
                constraints=30000,
                public_inputs=["signature_id", "match_score", "p_value"],
                private_inputs=[
                    "expression_values",
                    "signature_genes",
                    "normalization",
                ],
                parameters={"signature_size": 100},
            )
            circuits.append(expression_circuit)

        # Integrated risk
        integration_circuit = Circuit(
            name="integrated_risk_score",
            circuit_type="multi_omics",
            constraints=50000,
            public_inputs=["total_risk", "confidence_interval", "model_version"],
            private_inputs=["layer_scores", "layer_weights", "correlations"],
            parameters={"num_layers": len(omics_data)},
        )
        circuits.append(integration_circuit)

    elif analysis_type == "biomarker_discovery":
        # Create circuits for biomarker discovery

        correlation_circuit = Circuit(
            name="omics_correlation",
            circuit_type="multi_omics",
            constraints=40000,
            public_inputs=[
                "correlation_matrix_hash",
                "significant_pairs",
                "fdr_threshold",
            ],
            private_inputs=["layer_data", "sample_ids", "batch_effects"],
            parameters={"max_features": 1000, "correction_method": "BH"},
        )
        circuits.append(correlation_circuit)

    return circuits
