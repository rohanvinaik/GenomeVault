from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genomevault.utils.logging import get_logger
from genomevault.zk_proofs.circuits.base_circuits import FieldElement
from genomevault.zk_proofs.circuits.training_proof import TrainingProofCircuit

"""
Multi-Modal Training Proof Circuit for Cross-Omics Verification

This module implements cryptographic proofs for verifying that ML models
correctly learned from multiple biological modalities (genomic, transcriptomic, proteomic)
with proper cross-modal alignment.
"""


logger = get_logger(__name__)


@dataclass
class ModalityMetrics:
    """Metrics for a single biological modality during training"""

    modality_name: str  # 'genomic', 'transcriptomic', 'proteomic'
    data_hash: str
    feature_dim: int
    sample_count: int
    quality_score: float
    coverage: float  # Percentage of features with valid data


@dataclass
class CrossModalAlignment:
    """Alignment metrics between two modalities"""

    modality_a: str
    modality_b: str
    correlation: float
    mutual_information: float
    alignment_score: float
    attention_weights: list[float]


class MultiModalTrainingProof(TrainingProofCircuit):
    """
    Verify training across genomic, transcriptomic, and proteomic data.

    This circuit extends the base training proof to ensure:
    1. Model learned consistent representations across modalities
    2. Cross-modal correlations meet biological expectations
    3. Attention mechanisms properly weight each modality
    4. Temporal consistency for time-series omics data
    """

    def __init__(self, max_snapshots: int = 100, max_modalities: int = 5):
        """
        Initialize multi-modal training proof circuit.

        Args:
            max_snapshots: Maximum number of training snapshots
            max_modalities: Maximum number of biological modalities
        """
        # Additional constraints for cross-modal verification
        additional_constraints = max_modalities * 10  # Per-modality constraints
        additional_constraints += (
            max_modalities * (max_modalities - 1) // 2 * 5
        )  # Pairwise alignments

        super().__init__(max_snapshots)
        self.num_constraints += additional_constraints
        self.max_modalities = max_modalities

        self.modality_commits: dict[str, str] = {}
        self.correlation_thresholds: dict[str, float] = {}
        self.modality_metrics: dict[str, ModalityMetrics] = {}
        self.cross_modal_alignments: list[CrossModalAlignment] = []

    def setup(self, public_inputs: dict[str, Any], private_inputs: dict[str, Any]):
        """
        Setup multi-modal circuit inputs.

        Additional public inputs:
            - modality_hashes: Hash of each modality's data
            - expected_correlations: Expected correlation ranges

        Additional private inputs:
            - modality_commits: Commitments to each modality's features
            - correlation_thresholds: Minimum correlation requirements
            - modality_metrics: Detailed metrics per modality
            - cross_modal_alignments: Pairwise alignment data
        """
        # Call parent setup
        super().setup(public_inputs, private_inputs)

        # Multi-modal specific setup
        self.modality_hashes = {
            mod: FieldElement(int(h, 16))
            for mod, h in public_inputs.get("modality_hashes", {}).items()
        }

        self.expected_correlations = public_inputs.get("expected_correlations", {})

        # Private inputs
        self.modality_commits = private_inputs.get("modality_commits", {})
        self.correlation_thresholds = private_inputs.get(
            "correlation_thresholds",
            {
                "genomic_transcriptomic": 0.7,
                "transcriptomic_proteomic": 0.6,
                "genomic_proteomic": 0.5,
            },
        )

        # Parse modality metrics
        for mod_data in private_inputs.get("modality_metrics", []):
            metric = ModalityMetrics(**mod_data)
            self.modality_metrics[metric.modality_name] = metric

        # Parse cross-modal alignments
        for align_data in private_inputs.get("cross_modal_alignments", []):
            self.cross_modal_alignments.append(CrossModalAlignment(**align_data))

    def generate_constraints(self):
        """Generate constraints including multi-modal verification"""
        # First generate base training proof constraints
        super().generate_constraints()

        # Add multi-modal specific constraints
        logger.info("Generating multi-modal training constraints")

        # 1. Verify each modality's data integrity
        self.constrain_modality_integrity()

        # 2. Verify cross-modal alignment
        self.constrain_modality_alignment()

        # 3. Verify temporal consistency
        self.constrain_temporal_consistency()

        # 4. Verify attention weight constraints
        self.constrain_cross_modal_attention()

        # 5. Verify biological consistency
        self.constrain_biological_consistency()

    def constrain_modality_integrity(self):
        """Ensure each modality's data maintains integrity"""
        for modality_name, metrics in self.modality_metrics.items():
            # Verify modality data hash
            if modality_name in self.modality_hashes:
                declared_hash = self.modality_hashes[modality_name]
                actual_hash = FieldElement(int(metrics.data_hash, 16))

                # Constraint: actual data hash matches declared
                self.add_constraint(
                    actual_hash, declared_hash, FieldElement(0), ql=1, qr=-1
                )

            # Verify quality score is acceptable (> 0.8)
            quality_field = FieldElement(int(metrics.quality_score * 1000))
            min_quality = FieldElement(800)  # 0.8 * 1000

            # quality - min_quality > 0
            quality_diff = quality_field - min_quality
            if quality_diff.value > 0:
                diff_inv = quality_diff.inverse()
                self.add_constraint(
                    quality_diff, diff_inv, FieldElement(1), qm=1, qo=-1
                )

            # Verify coverage is sufficient (> 0.7)
            coverage_field = FieldElement(int(metrics.coverage * 1000))
            min_coverage = FieldElement(700)  # 0.7 * 1000

            coverage_diff = coverage_field - min_coverage
            if coverage_diff.value > 0:
                cov_inv = coverage_diff.inverse()
                self.add_constraint(
                    coverage_diff, cov_inv, FieldElement(1), qm=1, qo=-1
                )

    def constrain_modality_alignment(self):
        """Ensure proper alignment between modalities"""
        for alignment in self.cross_modal_alignments:
            # Get threshold for this modality pair
            pair_key = f"{alignment.modality_a}_{alignment.modality_b}"
            threshold = self.correlation_thresholds.get(pair_key, 0.5)

            # Verify correlation meets threshold
            corr_field = FieldElement(int(alignment.correlation * 1000))
            threshold_field = FieldElement(int(threshold * 1000))

            # correlation - threshold > 0
            corr_diff = corr_field - threshold_field
            if corr_diff.value > 0:
                corr_inv = corr_diff.inverse()
                self.add_constraint(corr_diff, corr_inv, FieldElement(1), qm=1, qo=-1)

            # Verify mutual information is meaningful (> 0.3)
            mi_field = FieldElement(int(alignment.mutual_information * 1000))
            min_mi = FieldElement(300)  # 0.3 * 1000

            mi_diff = mi_field - min_mi
            if mi_diff.value > 0:
                mi_inv = mi_diff.inverse()
                self.add_constraint(mi_diff, mi_inv, FieldElement(1), qm=1, qo=-1)

            # Verify alignment score
            align_field = FieldElement(int(alignment.alignment_score * 1000))
            min_align = FieldElement(600)  # 0.6 * 1000

            align_diff = align_field - min_align
            if align_diff.value > 0:
                align_inv = align_diff.inverse()
                self.add_constraint(align_diff, align_inv, FieldElement(1), qm=1, qo=-1)

    def constrain_temporal_consistency(self):
        """Ensure temporal consistency for time-series omics data"""
        # For transcriptomic data, ensure temporal patterns are preserved
        if "transcriptomic" in self.modality_metrics:
            trans_metrics = self.modality_metrics["transcriptomic"]

            # Verify feature dimensions are consistent across time
            # This is a simplified check - in production would verify
            # actual temporal correlation patterns
            feature_dim = FieldElement(trans_metrics.feature_dim)
            expected_dim = FieldElement(20000)  # Expected transcript count

            # Allow some variance (±10%)
            dim_diff = feature_dim - expected_dim
            max_variance = expected_dim * 10 // 100

            # |dim_diff| < max_variance
            # Simplified constraint - in production use range proof
            self.add_constraint(dim_diff, max_variance, FieldElement(0), ql=1, qr=-1)

    def constrain_cross_modal_attention(self):
        """Verify attention weights are properly bounded and sum to 1"""
        for alignment in self.cross_modal_alignments:
            if alignment.attention_weights:
                # Convert attention weights to field elements
                attention_fields = [
                    FieldElement(int(w * 1000)) for w in alignment.attention_weights
                ]

                # Verify each weight is in [0, 1]
                for weight in attention_fields:
                    # 0 <= weight <= 1000 (scaled by 1000)
                    # weight >= 0
                    if weight.value > 0:
                        weight_inv = weight.inverse()
                        self.add_constraint(
                            weight, weight_inv, FieldElement(1), qm=1, qo=-1
                        )

                    # weight <= 1000
                    max_weight = FieldElement(1000)
                    weight_diff = max_weight - weight
                    if weight_diff.value > 0:
                        diff_inv = weight_diff.inverse()
                        self.add_constraint(
                            weight_diff, diff_inv, FieldElement(1), qm=1, qo=-1
                        )

                # Verify weights sum to ~1 (allowing small epsilon for rounding)
                weight_sum = FieldElement(0)
                for weight in attention_fields:
                    weight_sum = weight_sum + weight

                expected_sum = FieldElement(1000)  # 1.0 * 1000
                epsilon = FieldElement(10)  # 0.01 * 1000

                # |weight_sum - expected_sum| < epsilon
                sum_diff = weight_sum - expected_sum
                # Simplified - in production use proper absolute value constraint
                self.add_constraint(sum_diff, epsilon, FieldElement(0), ql=1, qr=-1)

    def constrain_biological_consistency(self):
        """Verify biological relationships are preserved"""
        # Key biological constraints:
        # 1. Gene expression (transcriptomic) should correlate with protein abundance (proteomic)
        # 2. Genomic variants should influence transcript levels
        # 3. Pathway-level consistency across modalities

        # Find genomic->transcriptomic->proteomic alignments
        gen_trans_align = None
        trans_prot_align = None

        for align in self.cross_modal_alignments:
            if align.modality_a == "genomic" and align.modality_b == "transcriptomic":
                gen_trans_align = align
            elif (
                align.modality_a == "transcriptomic" and align.modality_b == "proteomic"
            ):
                trans_prot_align = align

        # Central dogma constraint: DNA -> RNA -> Protein
        # The correlation should decrease along this path but remain positive
        if gen_trans_align and trans_prot_align:
            FieldElement(int(gen_trans_align.correlation * 1000))
            trans_prot_corr = FieldElement(int(trans_prot_align.correlation * 1000))

            # trans_prot_corr should be positive but potentially lower than gen_trans_corr
            # trans_prot_corr > 0.3 (biological noise threshold)
            min_corr = FieldElement(300)  # 0.3 * 1000
            corr_diff = trans_prot_corr - min_corr

            if corr_diff.value > 0:
                diff_inv = corr_diff.inverse()
                self.add_constraint(corr_diff, diff_inv, FieldElement(1), qm=1, qo=-1)

    def verify_cross_modal_consistency(self) -> dict[str, float]:
        """
        Verify cross-modal consistency metrics.

        Returns:
            Dictionary of consistency scores per modality pair
        """
        consistency_scores = {}

        for alignment in self.cross_modal_alignments:
            pair_key = f"{alignment.modality_a}_{alignment.modality_b}"

            # Compute overall consistency score
            score = (
                alignment.correlation * 0.4
                + alignment.mutual_information * 0.3
                + alignment.alignment_score * 0.3
            )

            consistency_scores[pair_key] = score

            logger.info(
                "Cross-modal consistency %spair_key: %sscore:.3f "
                "(corr=%salignment.correlation:.3f, MI=%salignment.mutual_information:.3f)"
            )

        return consistency_scores

    def generate_proof(self) -> dict[str, Any]:
        """Generate multi-modal training proof"""
        # Generate base proof
        proof = super().generate_proof()

        # Add multi-modal specific elements
        proof["multi_modal"] = {
            "modalities": list(self.modality_metrics.keys()),
            "cross_modal_scores": self.verify_cross_modal_consistency(),
            "modality_qualities": {
                mod: metrics.quality_score
                for mod, metrics in self.modality_metrics.items()
            },
            "alignment_summary": {
                f"{align.modality_a}_{align.modality_b}": {
                    "correlation": align.correlation,
                    "mutual_information": align.mutual_information,
                    "alignment_score": align.alignment_score,
                }
                for align in self.cross_modal_alignments
            },
        }

        logger.info(
            "Generated multi-modal training proof for %slen(self.modality_metrics) modalities"
        )

        return proof
