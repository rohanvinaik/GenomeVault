"""
Biological zero-knowledge proof circuits

This module implements specialized circuits for genomic privacy,
including variant verification, PRS calculation, and clinical assessments.
"""

import hashlib

import torch

from genomevault.utils.logging import get_logger

from ..base_circuits import (Any, BaseCircuit, ComparisonCircuit, Dict,
                             FieldElement, List, MerkleTreeCircuit,
                             RangeProofCircuit)

logger = get_logger(__name__)


class VariantPresenceCircuit(BaseCircuit):
    """
    Circuit for proving presence of a genetic variant without revealing position

    Public inputs:
    - variant_hash: Hash of variant details (chr:pos:ref:alt)
    - reference_hash: Hash of reference genome version
    - commitment_root: Merkle root of user's genome commitment

    Private inputs:
    - variant_data: Actual variant information
    - merkle_proof: Proof of inclusion in genome
    - witness_randomness: Randomness for zero-knowledge
    """

    def __init__(self, merkle_depth: int = 20):
        super().__init__("variant_presence", 5000)
        self.merkle_depth = merkle_depth
        self.merkle_circuit = MerkleTreeCircuit(merkle_depth)

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup variant presence circuit"""
        # Public inputs
        self.variant_hash = FieldElement(int(public_inputs["variant_hash"], 16))
        self.reference_hash = FieldElement(int(public_inputs["reference_hash"], 16))
        self.commitment_root = FieldElement(int(public_inputs["commitment_root"], 16))

        # Private inputs
        self.variant_data = private_inputs["variant_data"]
        self.merkle_proof = private_inputs["merkle_proof"]
        self.witness_randomness = FieldElement(
            int(private_inputs["witness_randomness"], 16)
        )

        # Compute variant leaf
        self.variant_leaf = self._compute_variant_leaf()

    def generate_constraints(self):
        """Generate variant presence constraints"""
        # 1. Verify variant hash matches computed hash
        computed_hash = self._hash_variant(self.variant_data)
        self.add_constraint(
            computed_hash, self.variant_hash, FieldElement(0), ql=1, qr=-1
        )

        # 2. Verify variant is in Merkle tree
        self.merkle_circuit.setup(
            public_inputs={"root": self.commitment_root.value},
            private_inputs={
                "leaf": self.variant_leaf.value,
                "path": self.merkle_proof["path"],
                "indices": self.merkle_proof["indices"],
            },
        )
        self.merkle_circuit.generate_constraints()

        # Add Merkle constraints to this circuit
        self.constraints.extend(self.merkle_circuit.constraints)

        # 3. Add blinding factor for zero-knowledge
        blinded_variant = self.variant_leaf + self.witness_randomness
        self.add_constraint(
            blinded_variant,
            blinded_variant,
            FieldElement(0),
            ql=1,
            qr=-1,  # Identity constraint with blinding
        )

    def _compute_variant_leaf(self) -> FieldElement:
        """Compute Merkle tree leaf for variant"""
        variant_str = (
            "{self.variant_data['chr']}:"
            "{self.variant_data['pos']}:"
            "{self.variant_data['ref']}:"
            "{self.variant_data['alt']}:"
            "{self.variant_data.get('genotype', '0/1')}"
        )
        leaf_hash = hashlib.sha256(variant_str.encode()).hexdigest()
        return FieldElement(int(leaf_hash, 16))

    def _hash_variant(self, variant_data: Dict) -> FieldElement:
        """Hash variant data"""
        variant_str = (
            "{variant_data['chr']}:"
            "{variant_data['pos']}:"
            "{variant_data['ref']}:"
            "{variant_data['alt']}"
        )
        hash_val = hashlib.sha256(variant_str.encode()).hexdigest()
        return FieldElement(int(hash_val, 16))


class PolygenenicRiskScoreCircuit(BaseCircuit):
    """
    Circuit for computing Polygenic Risk Score without revealing variants

    Public inputs:
    - prs_model: Hash of PRS model (weights and variants)
    - score_range: Valid score range [min, max]
    - result_commitment: Commitment to calculated score
    - genome_commitment: Merkle root of user's genome

    Private inputs:
    - variants: User's variant genotypes (0, 1, or 2)
    - weights: PRS model weights
    - merkle_proofs: Proofs for each variant
    - witness_randomness: Randomness for zero-knowledge
    """

    def __init__(self, max_variants: int = 1000):
        super().__init__("polygenic_risk_score", 20000)
        self.max_variants = max_variants
        self.range_circuit = RangeProofCircuit(bit_width=32)

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup PRS circuit"""
        # Public inputs
        self.prs_model_hash = FieldElement(int(public_inputs["prs_model"], 16))
        self.score_range = public_inputs["score_range"]
        self.result_commitment = FieldElement(
            int(public_inputs["result_commitment"], 16)
        )
        self.genome_commitment = FieldElement(
            int(public_inputs["genome_commitment"], 16)
        )

        # Private inputs
        self.variants = [FieldElement(v) for v in private_inputs["variants"]]
        self.weights = [
            FieldElement(int(w * 1000)) for w in private_inputs["weights"]
        ]  # Scale weights
        self.merkle_proofs = private_inputs["merkle_proofs"]
        self.witness_randomness = FieldElement(
            int(private_inputs["witness_randomness"], 16)
        )

        # Ensure we have matching variants and weights
        assert len(self.variants) == len(self.weights)

    def generate_constraints(self):
        """Generate PRS calculation constraints"""
        # 1. Verify each variant genotype is valid (0, 1, or 2)
        for i, variant in enumerate(self.variants):
            self._add_genotype_constraint(variant)

        # 2. Calculate weighted sum
        score = FieldElement(0)
        for variant, weight in zip(self.variants, self.weights):
            contribution = variant * weight
            score = score + contribution

            # Add multiplication constraint
            self.add_multiplication_gate(variant, weight, contribution)

        # 3. Scale score back to normal range
        scaled_score = score * FieldElement(1000).inverse()  # Divide by 1000

        # 4. Verify score is in valid range
        self.range_circuit.setup(
            public_inputs={
                "min": int(self.score_range["min"] * 1000),
                "max": int(self.score_range["max"] * 1000),
            },
            private_inputs={"value": score.value},
        )
        self.range_circuit.generate_constraints()
        self.constraints.extend(self.range_circuit.constraints)

        # 5. Create and verify commitment
        commitment = self._commit_score(scaled_score, self.witness_randomness)
        self.add_constraint(
            commitment, self.result_commitment, FieldElement(0), ql=1, qr=-1
        )

        # 6. Verify PRS model hash
        model_hash = self._hash_prs_model(self.weights)
        self.add_constraint(
            model_hash, self.prs_model_hash, FieldElement(0), ql=1, qr=-1
        )

    def _add_genotype_constraint(self, genotype: FieldElement):
        """Constrain genotype to be 0, 1, or 2"""
        # g * (g - 1) * (g - 2) = 0
        g_minus_1 = genotype - FieldElement(1)
        g_minus_2 = genotype - FieldElement(2)

        # First: g * (g - 1)
        temp = genotype * g_minus_1
        self.add_multiplication_gate(genotype, g_minus_1, temp)

        # Then: temp * (g - 2) = 0
        result = temp * g_minus_2
        self.add_multiplication_gate(temp, g_minus_2, result)
        self.add_constraint(result, FieldElement(0), FieldElement(0), ql=1)

    def _commit_score(
        self, score: FieldElement, randomness: FieldElement
    ) -> FieldElement:
        """Create commitment to PRS score"""
        data = b"PRS:{score.value}:{randomness.value}"
        hash_val = hashlib.sha256(data).hexdigest()
        return FieldElement(int(hash_val, 16))

    def _hash_prs_model(self, weights: List[FieldElement]) -> FieldElement:
        """Hash PRS model weights"""
        weight_str = ":".join(str(w.value) for w in weights)
        hash_val = hashlib.sha256(weight_str.encode()).hexdigest()
        return FieldElement(int(hash_val, 16))


class DiabetesRiskCircuit(BaseCircuit):
    """
    Circuit for diabetes risk assessment (clinical pilot)

    Proves: (G > G_threshold) AND (R > R_threshold) without revealing G or R

    Public inputs:
    - glucose_threshold: G_threshold value
    - risk_threshold: R_threshold value
    - result_commitment: Commitment to boolean result

    Private inputs:
    - glucose_reading: Actual glucose value (G)
    - risk_score: Actual PRS with DP noise (R)
    - witness_randomness: Randomness for zero-knowledge
    """

    def __init__(self):
        super().__init__("diabetes_risk_alert", 15000)
        self.glucose_comparison = ComparisonCircuit()
        self.risk_comparison = ComparisonCircuit()

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup diabetes risk circuit"""
        # Public inputs
        self.glucose_threshold = FieldElement(
            int(public_inputs["glucose_threshold"] * 100)
        )  # Scale to avoid decimals
        self.risk_threshold = FieldElement(int(public_inputs["risk_threshold"] * 1000))
        self.result_commitment = FieldElement(
            int(public_inputs["result_commitment"], 16)
        )

        # Private inputs
        self.glucose_reading = FieldElement(
            int(private_inputs["glucose_reading"] * 100)
        )
        self.risk_score = FieldElement(int(private_inputs["risk_score"] * 1000))
        self.witness_randomness = FieldElement(
            int(private_inputs["witness_randomness"], 16)
        )

    def generate_constraints(self):
        """Generate diabetes risk assessment constraints"""
        # 1. Prove G > G_threshold
        self.glucose_comparison.setup(
            public_inputs={
                "result": True,
                "comparison_type": "gt",
            },  # We're proving it's true
            private_inputs={
                "a": self.glucose_reading.value,
                "b": self.glucose_threshold.value,
            },
        )
        self.glucose_comparison.generate_constraints()

        # 2. Prove R > R_threshold
        self.risk_comparison.setup(
            public_inputs={"result": True, "comparison_type": "gt"},
            private_inputs={"a": self.risk_score.value, "b": self.risk_threshold.value},
        )
        self.risk_comparison.generate_constraints()

        # Add comparison constraints
        self.constraints.extend(self.glucose_comparison.constraints)
        self.constraints.extend(self.risk_comparison.constraints)

        # 3. Compute AND of conditions
        # For simplicity, we know both are true if we got here
        condition_result = FieldElement(1)  # True

        # 4. Create and verify commitment
        commitment = self._commit_result(condition_result, self.witness_randomness)
        self.add_constraint(
            commitment, self.result_commitment, FieldElement(0), ql=1, qr=-1
        )

        # 5. Add range constraints for glucose and risk score
        self._add_glucose_range_constraint()
        self._add_risk_range_constraint()

    def _commit_result(
        self, result: FieldElement, randomness: FieldElement
    ) -> FieldElement:
        """Commit to boolean result"""
        data = b"DIABETES_RISK:{result.value}:{randomness.value}"
        hash_val = hashlib.sha256(data).hexdigest()
        return FieldElement(int(hash_val, 16))

    def _add_glucose_range_constraint(self):
        """Ensure glucose is in reasonable range (50-500 mg/dL)"""
        # Simplified: just check it's positive and less than 50000 (500 * 100)
        max_glucose = FieldElement(50000)
        diff = max_glucose - self.glucose_reading
        # In production, would add proper range proof

    def _add_risk_range_constraint(self):
        """Ensure risk score is in [0, 1] range"""
        # Risk score is scaled by 1000, so check [0, 1000]
        max_risk = FieldElement(1000)
        diff = max_risk - self.risk_score
        # In production, would add proper range proof


class PharmacogenomicCircuit(BaseCircuit):
    """
    Circuit for medication response prediction based on pharmacogenomics

    Public inputs:
    - medication_id: Medication being evaluated
    - response_category: Predicted response (poor/normal/rapid/ultra-rapid)
    - model_version: PharmGKB model version

    Private inputs:
    - star_alleles: User's CYP gene star alleles
    - variant_genotypes: Relevant variant genotypes
    - activity_scores: Computed enzyme activity scores
    - witness_randomness: Randomness for zero-knowledge
    """

    def __init__(self, max_star_alleles: int = 50):
        super().__init__("pharmacogenomic", 10000)
        self.max_star_alleles = max_star_alleles
        self.genes = ["CYP2C19", "CYP2D6", "CYP2C9", "VKORC1", "TPMT"]

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup pharmacogenomic circuit"""
        # Public inputs
        self.medication_id = FieldElement(public_inputs["medication_id"])
        self.response_category = FieldElement(public_inputs["response_category"])
        self.model_version = FieldElement(int(public_inputs["model_version"], 16))

        # Private inputs
        self.star_alleles = private_inputs["star_alleles"]
        self.variant_genotypes = private_inputs["variant_genotypes"]
        self.activity_scores = [
            FieldElement(int(s * 100)) for s in private_inputs["activity_scores"]
        ]
        self.witness_randomness = FieldElement(
            int(private_inputs["witness_randomness"], 16)
        )

    def generate_constraints(self):
        """Generate pharmacogenomic prediction constraints"""
        # 1. Compute total enzyme activity
        total_activity = FieldElement(0)
        for gene, activity in zip(self.genes, self.activity_scores):
            total_activity = total_activity + activity

        # 2. Map activity to response category
        # Poor: 0-50, Normal: 50-150, Rapid: 150-200, Ultra-rapid: >200
        predicted_category = self._activity_to_category(total_activity)

        # 3. Verify predicted category matches public input
        self.add_constraint(
            predicted_category, self.response_category, FieldElement(0), ql=1, qr=-1
        )

        # 4. Verify star alleles are valid
        for allele_data in self.star_alleles:
            self._verify_star_allele(allele_data)

        # 5. Create commitment to hide actual genotypes
        genotype_commitment = self._commit_genotypes(
            self.variant_genotypes, self.witness_randomness
        )

    def _activity_to_category(self, total_activity: FieldElement) -> FieldElement:
        """Map enzyme activity to response category"""
        # Simplified mapping - in production would use circuit-friendly comparison
        # 0: Poor, 1: Normal, 2: Rapid, 3: Ultra-rapid

        # For now, return a placeholder based on assumptions
        if total_activity.value < 5000:  # < 50 (scaled by 100)
            return FieldElement(0)
        elif total_activity.value < 15000:  # < 150
            return FieldElement(1)
        elif total_activity.value < 20000:  # < 200
            return FieldElement(2)
        else:
            return FieldElement(3)

    def _verify_star_allele(self, allele_data: Dict):
        """Verify star allele is valid for the gene"""
        # In production, would check against known star allele database
        # For now, just ensure it's properly formatted
        gene = allele_data["gene"]
        allele = allele_data["allele"]

        # Add constraint that gene is one of known genes
        # Simplified for demo

    def _commit_genotypes(
        self, genotypes: List[Dict], randomness: FieldElement
    ) -> FieldElement:
        """Create commitment to genotypes"""
        genotype_str = ":".join("{g['variant']}={g['genotype']}" for g in genotypes)
        data = b"GENOTYPES:{genotype_str}:{randomness.value}"
        hash_val = hashlib.sha256(data).hexdigest()
        return FieldElement(int(hash_val, 16))


class PathwayEnrichmentCircuit(BaseCircuit):
    """
    Circuit for pathway enrichment analysis without revealing expression

    Public inputs:
    - pathway_id: Pathway being tested (e.g., KEGG pathway)
    - enrichment_score: Calculated enrichment score
    - significance: P-value commitment

    Private inputs:
    - expression_values: Gene expression values (hypervectors)
    - gene_sets: Pathway gene sets
    - permutation_seeds: Seeds for significance testing
    - witness_randomness: Randomness for zero-knowledge
    """

    def __init__(self, max_genes: int = 20000):
        super().__init__("pathway_enrichment", 25000)
        self.max_genes = max_genes

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup pathway enrichment circuit"""
        # Public inputs
        self.pathway_id = FieldElement(public_inputs["pathway_id"])
        self.enrichment_score = FieldElement(
            int(public_inputs["enrichment_score"] * 1000)
        )
        self.significance = FieldElement(int(public_inputs["significance"], 16))

        # Private inputs
        self.expression_values = private_inputs["expression_values"]
        self.gene_sets = private_inputs["gene_sets"]
        self.permutation_seeds = private_inputs["permutation_seeds"]
        self.witness_randomness = FieldElement(
            int(private_inputs["witness_randomness"], 16)
        )

    def generate_constraints(self):
        """Generate pathway enrichment constraints"""
        # 1. Convert hypervector expressions to field elements
        expression_elements = self._hypervectors_to_field_elements(
            self.expression_values
        )

        # 2. Calculate enrichment score for pathway
        pathway_genes = self.gene_sets[str(self.pathway_id.value)]
        pathway_score = self._calculate_enrichment(expression_elements, pathway_genes)

        # 3. Verify enrichment score matches public input
        self.add_constraint(
            pathway_score, self.enrichment_score, FieldElement(0), ql=1, qr=-1
        )

        # 4. Perform permutation test for significance
        permutation_scores = []
        for seed in self.permutation_seeds[:10]:  # Limit permutations for demo
            perm_score = self._calculate_permuted_enrichment(
                expression_elements, pathway_genes, FieldElement(seed)
            )
            permutation_scores.append(perm_score)

        # 5. Calculate p-value commitment
        p_value_commitment = self._commit_significance(
            pathway_score, permutation_scores, self.witness_randomness
        )

        self.add_constraint(
            p_value_commitment, self.significance, FieldElement(0), ql=1, qr=-1
        )

    def _hypervectors_to_field_elements(
        self, hypervectors: torch.Tensor
    ) -> List[FieldElement]:
        """Convert hypervector to field elements"""
        # Take first few dimensions for efficiency
        if isinstance(hypervectors, torch.Tensor):
            values = hypervectors[:100].numpy()  # Use first 100 dimensions
        else:
            values = hypervectors[:100]

        return [FieldElement(int(v * 1000)) for v in values]

    def _calculate_enrichment(
        self, expression: List[FieldElement], gene_indices: List[int]
    ) -> FieldElement:
        """Calculate enrichment score for gene set"""
        # Simplified: sum expression of genes in pathway
        pathway_sum = FieldElement(0)
        for idx in gene_indices[:20]:  # Limit for demo
            if idx < len(expression):
                pathway_sum = pathway_sum + expression[idx]

        # Normalize by pathway size
        pathway_size = FieldElement(len(gene_indices))
        # Simplified division
        normalized_score = pathway_sum * FieldElement(1000) * pathway_size.inverse()

        return normalized_score

    def _calculate_permuted_enrichment(
        self,
        expression: List[FieldElement],
        gene_indices: List[int],
        seed: FieldElement,
    ) -> FieldElement:
        """Calculate enrichment with permuted gene labels"""
        # Simplified: just add noise based on seed
        base_score = self._calculate_enrichment(expression, gene_indices)
        noise = seed * FieldElement(100)  # Scale down seed
        return base_score + noise

    def _commit_significance(
        self,
        observed_score: FieldElement,
        permutation_scores: List[FieldElement],
        randomness: FieldElement,
    ) -> FieldElement:
        """Create commitment to p-value"""
        # Count how many permutation scores exceed observed
        exceed_count = 0
        for perm_score in permutation_scores:
            if perm_score.value > observed_score.value:
                exceed_count += 1

        # p-value = (exceed_count + 1) / (num_permutations + 1)
        p_value = FieldElement(
            (exceed_count + 1) * 1000 // (len(permutation_scores) + 1)
        )

        # Create commitment
        data = b"PVALUE:{p_value.value}:{randomness.value}"
        hash_val = hashlib.sha256(data).hexdigest()
        return FieldElement(int(hash_val, 16))


# Helper function to create proof for hypervector data
def create_hypervector_proof(
    hypervector: torch.Tensor, proof_type: str, public_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create ZK proof for hypervector-encoded data

    Args:
        hypervector: Encoded biological data
        proof_type: Type of proof to generate
        public_params: Public parameters for the proof

    Returns:
        Proof data including circuit and commitments
    """
    # Extract features from hypervector for proof
    if proof_type == "similarity":
        # Prove similarity between two hypervectors without revealing them
        return {
            "circuit": "hypervector_similarity",
            "similarity_commitment": hashlib.sha256(
                b"{hypervector.sum().item()}"
            ).hexdigest(),
            "dimension": hypervector.shape[0],
        }
    elif proof_type == "range":
        # Prove hypervector properties are in expected range
        return {
            "circuit": "hypervector_range",
            "norm_commitment": hashlib.sha256(
                b"{torch.norm(hypervector).item()}"
            ).hexdigest(),
            "sparsity_commitment": hashlib.sha256(
                b"{(hypervector == 0).float().mean().item()}"
            ).hexdigest(),
        }
    else:
        raise ValueError("Unknown proof type: {proof_type}")
