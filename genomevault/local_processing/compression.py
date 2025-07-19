"""
GenomeVault Compression System - Multi-tier implementation

Implements the three-tier compression system as specified:
- Mini tier: ~25KB - 5,000 most-studied SNPs
- Clinical tier: ~300KB - ACMG + PharmGKB variants (~120k)
- Full HDC tier: 100-200KB per modality - 10,000-D vectors
"""

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from genomevault.core.constants import CompressionTier, OmicsType
from genomevault.utils import get_config, get_logger

logger = get_logger(__name__)
config = get_config()


@dataclass
class CompressionProfile:
    """Compression profile for a specific tier"""

    tier: CompressionTier
    max_size_kb: int
    feature_count: int
    description: str
    omics_types: List[OmicsType] = field(default_factory=list)


@dataclass
class CompressedData:
    """Compressed genomic data package"""

    sample_id: str
    tier: CompressionTier
    data: bytes
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any]
    omics_included: List[OmicsType]

    def verify_checksum(self) -> bool:
        """Verify data integrity"""
        calculated = hashlib.sha256(self.data).hexdigest()
        return calculated == self.checksum


class CompressionEngine:
    """
    Multi-tier compression engine for genomic data.
    Implements the specification's storage requirements.
    """

    # Compression profiles as per specification
    COMPRESSION_PROFILES = {
        CompressionTier.MINI: CompressionProfile(
            tier=CompressionTier.MINI,
            max_size_kb=25,
            feature_count=5000,
            description="Most-studied SNPs only",
            omics_types=[OmicsType.GENOMIC],
        ),
        CompressionTier.CLINICAL: CompressionProfile(
            tier=CompressionTier.CLINICAL,
            max_size_kb=300,
            feature_count=120000,
            description="ACMG + PharmGKB variants",
            omics_types=[OmicsType.GENOMIC, OmicsType.PHENOTYPIC],
        ),
        CompressionTier.FULL: CompressionProfile(
            tier=CompressionTier.FULL,
            max_size_kb=200,  # per modality
            feature_count=10000,  # hypervector dimensions
            description="Full HDC vectors per modality",
            omics_types=[
                OmicsType.GENOMIC,
                OmicsType.TRANSCRIPTOMIC,
                OmicsType.EPIGENETIC,
                OmicsType.PROTEOMIC,
                OmicsType.PHENOTYPIC,
            ],
        ),
    }

    def __init__(self):
        """Initialize compression engine"""
        self.variant_databases = self._load_variant_databases()
        logger.info("Compression engine initialized")

    def _load_variant_databases(self) -> Dict[str, List[str]]:
        """Load reference variant databases for compression tiers"""
        # In production, these would be loaded from actual databases
        databases = {
            "most_studied_snps": [],  # Top 5,000 SNPs
            "acmg_variants": [],  # ACMG secondary findings
            "pharmgkb_variants": [],  # PharmGKB pharmacogenomic variants
        }

        # Placeholder: Generate example variant IDs
        databases["most_studied_snps"] = [f"rs{i}" for i in range(1, 5001)]

        return databases

    def compress(
        self, data: Dict[str, Any], tier: CompressionTier, sample_id: str
    ) -> CompressedData:
        """
        Compress multi-omics data according to specified tier.

        Args:
            data: Multi-omics data dictionary
            tier: Compression tier to use
            sample_id: Sample identifier

        Returns:
            Compressed data package
        """
        profile = self.COMPRESSION_PROFILES[tier]
        logger.info(f"Compressing data for {sample_id} using {tier.value} tier")

        # Select compression method based on tier
        if tier == CompressionTier.MINI:
            compressed_dict = self._compress_mini_tier(data)
        elif tier == CompressionTier.CLINICAL:
            compressed_dict = self._compress_clinical_tier(data)
        else:  # FULL
            compressed_dict = self._compress_full_tier(data)

        # Convert to binary format
        json_str = json.dumps(compressed_dict, separators=(",", ":"))
        compressed_bytes = gzip.compress(json_str.encode("utf-8"))

        # Verify size constraints
        size_kb = len(compressed_bytes) / 1024
        max_size = profile.max_size_kb

        # For full tier, multiply by number of modalities
        if tier == CompressionTier.FULL:
            modalities_included = len(
                [
                    k
                    for k in data.keys()
                    if k in ["genomic", "transcriptomic", "epigenetic", "proteomic"]
                ]
            )
            max_size = profile.max_size_kb * modalities_included

        if size_kb > max_size:
            logger.warning(f"Compressed size {size_kb:.1f}KB exceeds target {max_size}KB")

        # Create compressed data package
        compressed_data = CompressedData(
            sample_id=sample_id,
            tier=tier,
            data=compressed_bytes,
            size_bytes=len(compressed_bytes),
            checksum=hashlib.sha256(compressed_bytes).hexdigest(),
            metadata={
                "compression_version": "1.0",
                "profile": profile.description,
                "features_included": len(compressed_dict.get("features", [])),
            },
            omics_included=[o for o in OmicsType if o.value in data],
        )

        logger.info(f"Compression complete: {size_kb:.1f}KB ({tier.value} tier)")
        return compressed_data

    def _compress_mini_tier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mini tier compression: ~25KB with 5,000 most-studied SNPs.
        """
        compressed = {"tier": CompressionTier.MINI.value, "features": []}

        # Extract genomic variants only
        if "genomic" in data and "variants" in data["genomic"]:
            variants = data["genomic"]["variants"]

            # Filter to most studied SNPs
            for variant in variants:
                if variant.get("rsid") in self.variant_databases["most_studied_snps"]:
                    compressed["features"].append(
                        {
                            "id": variant["rsid"],
                            "gt": variant.get("genotype", "0/0"),  # Compact genotype
                            "af": round(variant.get("allele_frequency", 0), 3),
                        }
                    )

                if len(compressed["features"]) >= 5000:
                    break

        # Add minimal metadata
        compressed["meta"] = {
            "ref": "GRCh38",
            "date": data.get("metadata", {}).get("date", ""),
            "n_vars": len(compressed["features"]),
        }

        return compressed

    def _compress_clinical_tier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clinical tier compression: ~300KB with ACMG + PharmGKB variants.
        """
        compressed = {"tier": CompressionTier.CLINICAL.value, "genomic": {}, "phenotypic": {}}

        # Include ACMG and PharmGKB variants
        if "genomic" in data and "variants" in data["genomic"]:
            clinical_variants = []

            for variant in data["genomic"]["variants"]:
                # Check if variant is clinically relevant
                is_acmg = self._is_acmg_variant(variant)
                is_pharmgkb = self._is_pharmgkb_variant(variant)

                if is_acmg or is_pharmgkb:
                    clinical_variants.append(
                        {
                            "chr": variant["chromosome"],
                            "pos": variant["position"],
                            "ref": variant["reference"],
                            "alt": variant["alternate"],
                            "gt": variant.get("genotype", "0/0"),
                            "qual": round(variant.get("quality", 0), 1),
                            "ann": {
                                "acmg": is_acmg,
                                "pgkb": is_pharmgkb,
                                "gene": variant.get("gene", ""),
                                "impact": variant.get("impact", ""),
                            },
                        }
                    )

            compressed["genomic"]["variants"] = clinical_variants[:120000]

        # Include key phenotypic data
        if "phenotypic" in data:
            compressed["phenotypic"] = {
                "conditions": data["phenotypic"].get("conditions", []),
                "medications": data["phenotypic"].get("medications", []),
                "labs": self._compress_lab_values(data["phenotypic"].get("lab_results", {})),
            }

        # Metadata
        compressed["meta"] = {
            "ref": "GRCh38",
            "date": data.get("metadata", {}).get("date", ""),
            "n_vars": len(compressed["genomic"].get("variants", [])),
            "clinical_version": "ACMG-v3.0,PharmGKB-2024",
        }

        return compressed

    def _compress_full_tier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full tier compression: 100-200KB per modality using HDC vectors.
        """
        compressed = {"tier": CompressionTier.FULL.value, "modalities": {}}

        # Process each modality into hypervectors
        for modality in ["genomic", "transcriptomic", "epigenetic", "proteomic"]:
            if modality in data:
                # In practice, this would use the actual hypervector encoder
                # For now, simulate with compressed representation
                compressed["modalities"][modality] = {
                    "hypervector": self._create_mock_hypervector(
                        data[modality], 10000  # 10,000-D as specified
                    ),
                    "stats": self._extract_modality_stats(data[modality]),
                }

        # Add integrated multi-omics features
        if len(compressed["modalities"]) > 1:
            compressed["integrated"] = {
                "cross_modal_binding": self._compute_cross_modal_features(compressed["modalities"])
            }

        # Metadata
        compressed["meta"] = {
            "ref": "GRCh38",
            "date": data.get("metadata", {}).get("date", ""),
            "modalities_included": list(compressed["modalities"].keys()),
            "hypervector_dim": 10000,
            "compression_version": "HDC-v1.0",
        }

        return compressed

    def _is_acmg_variant(self, variant: Dict[str, Any]) -> bool:
        """Check if variant is in ACMG secondary findings list"""
        # Simplified check - in production would use actual ACMG database
        acmg_genes = [
            "BRCA1",
            "BRCA2",
            "MLH1",
            "MSH2",
            "MSH6",
            "PMS2",
            "APC",
            "MUTYH",
            "VHL",
            "MEN1",
            "RET",
            "PTEN",
            "TP53",
            "STK11",
            "LDLR",
            "APOB",
            "PCSK9",
        ]
        return variant.get("gene", "") in acmg_genes

    def _is_pharmgkb_variant(self, variant: Dict[str, Any]) -> bool:
        """Check if variant is pharmacogenomically relevant"""
        # Simplified check - in production would use PharmGKB database
        pgx_genes = [
            "CYP2C19",
            "CYP2D6",
            "CYP2C9",
            "CYP3A4",
            "CYP3A5",
            "VKORC1",
            "TPMT",
            "NUDT15",
            "DPYD",
            "UGT1A1",
            "SLCO1B1",
            "CYP4F2",
            "HLA-B",
            "HLA-A",
        ]
        return variant.get("gene", "") in pgx_genes

    def _compress_lab_values(self, lab_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compress laboratory values for clinical tier"""
        compressed_labs = {}

        # Key labs for clinical use
        important_labs = [
            "glucose",
            "hba1c",
            "cholesterol_total",
            "ldl",
            "hdl",
            "triglycerides",
            "creatinine",
            "egfr",
            "ast",
            "alt",
            "tsh",
            "hemoglobin",
            "wbc",
            "platelets",
        ]

        for lab in important_labs:
            if lab in lab_results:
                value = lab_results[lab]
                compressed_labs[lab] = {
                    "v": round(value.get("value", 0), 2),
                    "u": value.get("unit", ""),
                    "d": value.get("date", "")[:10] if "date" in value else "",
                }

        return compressed_labs

    def _create_mock_hypervector(self, modality_data: Dict[str, Any], dimensions: int) -> str:
        """
        Create mock hypervector representation.
        In production, this would use the actual hypervector encoder.
        """
        # Simulate by creating a hash-based compact representation
        data_str = json.dumps(modality_data, sort_keys=True)
        base_hash = hashlib.sha256(data_str.encode()).digest()

        # Expand to create pseudo-hypervector (compressed representation)
        # In practice, this would be the actual HDC encoding
        expanded = hashlib.pbkdf2_hmac(
            "sha256", base_hash, b"genomevault", iterations=100, dklen=dimensions // 8
        )

        # Convert to base64 for compact string representation
        import base64

        return base64.b64encode(expanded).decode("ascii")

    def _extract_modality_stats(self, modality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key statistics from modality data"""
        stats = {}

        if "variants" in modality_data:
            stats["variant_count"] = len(modality_data["variants"])

        if "expression_matrix" in modality_data:
            expr = modality_data["expression_matrix"]
            if isinstance(expr, dict):
                stats["genes_measured"] = len(expr)

        if "methylation_levels" in modality_data:
            stats["cpg_sites"] = len(modality_data["methylation_levels"])

        if "protein_abundances" in modality_data:
            stats["proteins_measured"] = len(modality_data["protein_abundances"])

        return stats

    def _compute_cross_modal_features(self, modalities: Dict[str, Any]) -> Dict[str, Any]:
        """Compute cross-modal binding features for multi-omics integration"""
        # Placeholder for cross-modal analysis
        # In production, would compute actual biological relationships
        return {
            "modality_pairs": list(modalities.keys()),
            "binding_strength": 0.85,  # Mock correlation
            "integration_method": "circular_convolution",
        }

    def decompress(self, compressed: CompressedData) -> Dict[str, Any]:
        """
        Decompress data package back to usable format.

        Args:
            compressed: Compressed data package

        Returns:
            Decompressed data dictionary
        """
        # Verify integrity
        if not compressed.verify_checksum():
            raise ValueError("Compressed data integrity check failed")

        # Decompress
        json_str = gzip.decompress(compressed.data).decode("utf-8")
        data = json.loads(json_str)

        # Add decompression metadata
        data["_decompression_info"] = {
            "sample_id": compressed.sample_id,
            "tier": compressed.tier.value,
            "original_size_bytes": compressed.size_bytes,
            "omics_included": [o.value for o in compressed.omics_included],
        }

        return data

    def calculate_storage_requirements(
        self, tiers: List[CompressionTier], modalities: List[OmicsType]
    ) -> Dict[str, Any]:
        """
        Calculate storage requirements for given tiers and modalities.

        Args:
            tiers: List of compression tiers to use
            modalities: List of omics types to include

        Returns:
            Storage requirement details
        """
        total_size_kb = 0
        breakdown = {}

        for tier in tiers:
            profile = self.COMPRESSION_PROFILES[tier]

            if tier == CompressionTier.FULL:
                # Full tier is per-modality
                modality_count = len([m for m in modalities if m in profile.omics_types])
                size_kb = profile.max_size_kb * modality_count
            else:
                size_kb = profile.max_size_kb

            breakdown[tier.value] = {
                "size_kb": size_kb,
                "features": profile.feature_count,
                "description": profile.description,
            }

            total_size_kb += size_kb

        return {
            "total_size_kb": total_size_kb,
            "total_size_mb": total_size_kb / 1024,
            "breakdown": breakdown,
            "formula": "S_client = ∑modalities Size_tier",
        }


# Example usage
if __name__ == "__main__":
    # Initialize compression engine
    engine = CompressionEngine()

    # Example: Calculate storage for different configurations
    print("Storage Requirements Examples:")
    print("=" * 50)

    # Example 1: Mini genomics only
    req1 = engine.calculate_storage_requirements([CompressionTier.MINI], [OmicsType.GENOMIC])
    print(f"Mini genomics only: {req1['total_size_kb']} KB")

    # Example 2: Clinical pharmacogenomics
    req2 = engine.calculate_storage_requirements(
        [CompressionTier.CLINICAL], [OmicsType.GENOMIC, OmicsType.PHENOTYPIC]
    )
    print(f"Clinical tier: {req2['total_size_kb']} KB")

    # Example 3: Mini + Clinical (as specified in docs)
    req3 = engine.calculate_storage_requirements(
        [CompressionTier.MINI, CompressionTier.CLINICAL], [OmicsType.GENOMIC, OmicsType.PHENOTYPIC]
    )
    print(f"Mini + Clinical: {req3['total_size_kb']} KB")

    # Example 4: Full multi-omics
    req4 = engine.calculate_storage_requirements(
        [CompressionTier.FULL],
        [OmicsType.GENOMIC, OmicsType.TRANSCRIPTOMIC, OmicsType.EPIGENETIC, OmicsType.PROTEOMIC],
    )
    print(f"Full multi-omics (4 modalities): {req4['total_size_kb']} KB")

    print("\nDetailed breakdown:")
    print(json.dumps(req4, indent=2))
