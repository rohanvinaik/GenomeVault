#!/usr/bin/env python3
"""
Differential Encoding Metadata Demo

Demonstrates the differential encoding metadata functionality from Section 5.2
of the specification. Shows how to:
1. Create metadata for differential encoding results
2. Serialize/deserialize metadata to JSON
3. Verify cryptographic bindings
4. Use the factory function for convenient creation
5. Validate metadata integrity
"""

import json
from datetime import datetime

from genomevault.differential_encoding import (
    DifferentialEncodingMetadata,
    METADATA_SCHEMA,
    validate_metadata_schema,
    create_metadata_from_chunk,
    AnalysisType,
    CryptoRNG,
)


def demo_basic_metadata_creation():
    """Demonstrate basic metadata creation and validation."""
    print("=" * 70)
    print("1. Basic Metadata Creation")
    print("=" * 70)

    # Create basic metadata
    metadata = DifferentialEncodingMetadata(
        chunk_id=b"\x01" * 32,
        chromosome="chr7",
        start_position=117_120_000,  # CFTR gene region
        end_position=117_308_000,
        reference_genome_id="GRCh38_1000G_phase3_001",
        reference_seed=b"\x02" * 32,
        reference_hash=b"\x03" * 32,
        chunking_strategy="gene_region",
        chunking_seed=b"\x04" * 32,
        analysis_type="gene_region",
        difference_counts={
            "new_mutations": 15,
            "missing_variants": 8,
            "genotype_differences": 5,
            "total": 28,
        },
        cryptographic_binding=b"\x05" * 32,
        feature_associations=["CFTR"],
        metadata={
            "gene_name": "CFTR",
            "clinical_significance": "pathogenic_variants_detected",
            "coverage_mean": 45.2,
        },
    )

    print(f"\n📦 Created metadata:")
    print(f"  Region: {metadata.get_region_string()}")
    print(f"  Size: {metadata.get_region_size():,} bp")
    print(f"  Reference: {metadata.reference_genome_id}")
    print(f"  Analysis: {metadata.analysis_type}")
    print(f"  Total differences: {metadata.difference_counts['total']}")
    print(f"    - New mutations: {metadata.difference_counts['new_mutations']}")
    print(f"    - Missing variants: {metadata.difference_counts['missing_variants']}")
    print(f"    - Genotype differences: {metadata.difference_counts['genotype_differences']}")
    print(f"  Features: {', '.join(metadata.feature_associations)}")
    print(f"  Additional metadata: {metadata.metadata}")

    return metadata


def demo_serialization(metadata: DifferentialEncodingMetadata):
    """Demonstrate serialization and deserialization."""
    print("\n" + "=" * 70)
    print("2. Serialization and Deserialization")
    print("=" * 70)

    # Convert to dictionary
    data_dict = metadata.to_dict()
    print(f"\n📝 Dictionary representation:")
    print(f"  chunk_id: {data_dict['chunk_id'][:16]}... (truncated)")
    print(f"  chromosome: {data_dict['chromosome']}")
    print(f"  start_position: {data_dict['start_position']:,}")
    print(f"  end_position: {data_dict['end_position']:,}")

    # Convert to JSON
    json_str = metadata.to_json(indent=2)
    print(f"\n📄 JSON representation (first 500 chars):")
    print(json_str[:500] + "...")

    # Save to file
    output_file = "/tmp/metadata_demo.json"
    with open(output_file, "w") as f:
        f.write(json_str)
    print(f"\n💾 Saved to: {output_file}")

    # Deserialize from JSON
    with open(output_file, "r") as f:
        loaded_json = f.read()

    restored_metadata = DifferentialEncodingMetadata.from_json(loaded_json)
    print(f"\n✅ Successfully restored from JSON")
    print(f"  Original chunk_id == Restored chunk_id: {metadata.chunk_id == restored_metadata.chunk_id}")
    print(f"  Original region == Restored region: {metadata.get_region_string() == restored_metadata.get_region_string()}")

    return restored_metadata


def demo_cryptographic_binding():
    """Demonstrate cryptographic binding computation and verification."""
    print("\n" + "=" * 70)
    print("3. Cryptographic Binding Verification")
    print("=" * 70)

    # Simulate chunk and reference data
    chunk_data = b"experimental_genome_chunk_data_with_variants_chr7:117120000-117308000"
    reference_data = b"reference_genome_section_data_GRCh38_1000G_phase3_chr7:117120000-117308000"
    seed = b"\x04" * 32

    # Compute binding
    binding = DifferentialEncodingMetadata.compute_binding(
        chunk_data, reference_data, seed
    )

    print(f"\n🔐 Computed cryptographic binding:")
    print(f"  Binding (hex): {binding.hex()}")
    print(f"  Length: {len(binding)} bytes")

    # Create metadata with this binding
    metadata = DifferentialEncodingMetadata(
        chunk_id=b"\x01" * 32,
        chromosome="chr7",
        start_position=117_120_000,
        end_position=117_308_000,
        reference_genome_id="GRCh38_1000G_phase3_001",
        reference_seed=b"\x02" * 32,
        reference_hash=b"\x03" * 32,
        chunking_strategy="gene_region",
        chunking_seed=seed,
        analysis_type="gene_region",
        difference_counts={
            "new_mutations": 15,
            "missing_variants": 8,
            "genotype_differences": 5,
            "total": 28,
        },
        cryptographic_binding=binding,
    )

    # Verify binding with correct data
    print(f"\n✅ Verifying binding with correct data:")
    is_valid = metadata.verify_binding(chunk_data, reference_data)
    print(f"  Valid: {is_valid}")

    # Try with tampered data
    tampered_chunk = b"TAMPERED_experimental_genome_chunk_data"
    print(f"\n❌ Verifying binding with tampered chunk data:")
    is_valid_tampered = metadata.verify_binding(tampered_chunk, reference_data)
    print(f"  Valid: {is_valid_tampered}")

    tampered_reference = b"TAMPERED_reference_genome_section_data"
    print(f"\n❌ Verifying binding with tampered reference data:")
    is_valid_tampered_ref = metadata.verify_binding(chunk_data, tampered_reference)
    print(f"  Valid: {is_valid_tampered_ref}")


def demo_factory_function():
    """Demonstrate using the factory function for convenient metadata creation."""
    print("\n" + "=" * 70)
    print("4. Factory Function for Convenient Creation")
    print("=" * 70)

    # Generate random cryptographic values
    rng = CryptoRNG()
    chunk_id = rng.derive_seed(b"chunk_001")
    reference_seed = rng.derive_seed(b"reference_001")
    reference_hash = rng.derive_seed(b"reference_hash")
    chunking_seed = rng.derive_seed(b"chunking_seed")

    # Simulate chunk and reference data
    chunk_data = b"chunk_genomic_data_for_BRCA1_region"
    reference_data = b"reference_genomic_data_for_BRCA1_region"

    # Use factory function
    metadata = create_metadata_from_chunk(
        chunk_id=chunk_id,
        chromosome="chr17",
        start_position=43_044_295,  # BRCA1 gene start
        end_position=43_125_483,    # BRCA1 gene end
        reference_genome_id="GRCh38_gnomAD_v3.1_chr17",
        reference_seed=reference_seed,
        reference_hash=reference_hash,
        chunking_strategy="gene_region",
        chunking_seed=chunking_seed,
        analysis_type=AnalysisType.GENE_REGION,
        new_mutations=12,
        missing_variants=4,
        genotype_differences=3,
        chunk_data=chunk_data,
        reference_data=reference_data,
        feature_associations=["BRCA1"],
        additional_metadata={
            "gene_name": "BRCA1",
            "clinical_significance": "cancer_risk_assessment",
            "quality_score": 0.98,
        },
    )

    print(f"\n🏭 Created metadata using factory function:")
    print(f"  {metadata}")
    print(f"\n🔍 Details:")
    print(f"  Region: {metadata.get_region_string()}")
    print(f"  Size: {metadata.get_region_size():,} bp")
    print(f"  Total differences: {metadata.difference_counts['total']}")
    print(f"  Cryptographic binding computed: {metadata.cryptographic_binding.hex()[:32]}...")

    # Verify the binding
    is_valid = metadata.verify_binding(chunk_data, reference_data)
    print(f"\n✅ Binding verification: {is_valid}")

    return metadata


def demo_schema_validation():
    """Demonstrate JSON schema validation."""
    print("\n" + "=" * 70)
    print("5. JSON Schema Validation")
    print("=" * 70)

    # Create valid metadata dictionary
    valid_data = {
        "chunk_id": "01" * 32,
        "chromosome": "chr1",
        "start_position": 100000,
        "end_position": 200000,
        "reference_genome_id": "GRCh38_ref_001",
        "reference_seed": "02" * 32,
        "reference_hash": "03" * 32,
        "chunking_strategy": "sliding_window",
        "chunking_seed": "04" * 32,
        "analysis_type": "sliding_window",
        "difference_counts": {
            "new_mutations": 5,
            "missing_variants": 3,
            "genotype_differences": 2,
            "total": 10,
        },
        "cryptographic_binding": "05" * 32,
        "created_timestamp": datetime.utcnow().isoformat(),
    }

    print(f"\n📋 Validating valid metadata:")
    try:
        is_valid = validate_metadata_schema(valid_data)
        print(f"  ✅ Validation passed: {is_valid}")
    except ValueError as e:
        print(f"  ❌ Validation failed: {e}")

    # Try with invalid data (missing field)
    invalid_data = valid_data.copy()
    del invalid_data["chromosome"]

    print(f"\n📋 Validating invalid metadata (missing chromosome):")
    try:
        validate_metadata_schema(invalid_data)
        print(f"  ❌ Validation should have failed!")
    except ValueError as e:
        print(f"  ✅ Validation correctly failed: {e}")

    # Try with invalid difference counts
    invalid_data2 = valid_data.copy()
    invalid_data2["difference_counts"] = {
        "new_mutations": 5,
        # Missing "missing_variants"
        "genotype_differences": 2,
        "total": 7,
    }

    print(f"\n📋 Validating invalid metadata (missing difference count):")
    try:
        validate_metadata_schema(invalid_data2)
        print(f"  ❌ Validation should have failed!")
    except ValueError as e:
        print(f"  ✅ Validation correctly failed: {e}")


def demo_multiple_regions():
    """Demonstrate creating metadata for multiple genomic regions."""
    print("\n" + "=" * 70)
    print("6. Metadata for Multiple Genomic Regions")
    print("=" * 70)

    regions = [
        {
            "chromosome": "chr7",
            "start": 117_120_000,
            "end": 117_308_000,
            "gene": "CFTR",
            "new": 15, "missing": 8, "genotype": 5,
        },
        {
            "chromosome": "chr17",
            "start": 43_044_295,
            "end": 43_125_483,
            "gene": "BRCA1",
            "new": 12, "missing": 4, "genotype": 3,
        },
        {
            "chromosome": "chr13",
            "start": 32_315_086,
            "end": 32_400_268,
            "gene": "BRCA2",
            "new": 8, "missing": 6, "genotype": 4,
        },
    ]

    metadata_list = []
    rng = CryptoRNG()

    for i, region in enumerate(regions):
        chunk_data = f"chunk_data_for_{region['gene']}".encode()
        reference_data = f"reference_data_for_{region['gene']}".encode()

        metadata = create_metadata_from_chunk(
            chunk_id=rng.derive_seed(f"chunk_{i}".encode()),
            chromosome=region["chromosome"],
            start_position=region["start"],
            end_position=region["end"],
            reference_genome_id=f"GRCh38_ref_{i:03d}",
            reference_seed=rng.derive_seed(f"ref_seed_{i}".encode()),
            reference_hash=rng.derive_seed(f"ref_hash_{i}".encode()),
            chunking_strategy="gene_region",
            chunking_seed=rng.derive_seed(f"chunk_seed_{i}".encode()),
            analysis_type=AnalysisType.GENE_REGION,
            new_mutations=region["new"],
            missing_variants=region["missing"],
            genotype_differences=region["genotype"],
            chunk_data=chunk_data,
            reference_data=reference_data,
            feature_associations=[region["gene"]],
            additional_metadata={"gene_name": region["gene"]},
        )
        metadata_list.append(metadata)

    print(f"\n📊 Created metadata for {len(metadata_list)} regions:")
    print()
    for metadata in metadata_list:
        print(f"  {metadata}")

    # Save all metadata to JSON
    output_file = "/tmp/metadata_batch.json"
    batch_data = [m.to_dict() for m in metadata_list]
    with open(output_file, "w") as f:
        json.dump(batch_data, f, indent=2)

    print(f"\n💾 Saved batch metadata to: {output_file}")

    # Calculate statistics
    total_differences = sum(m.difference_counts["total"] for m in metadata_list)
    total_size = sum(m.get_region_size() for m in metadata_list)

    print(f"\n📈 Batch statistics:")
    print(f"  Total regions: {len(metadata_list)}")
    print(f"  Total genomic span: {total_size:,} bp")
    print(f"  Total differences: {total_differences}")
    print(f"  Average differences per region: {total_differences / len(metadata_list):.1f}")


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("Differential Encoding Metadata Demo")
    print("Section 5.2: Metadata Management")
    print("=" * 70)

    # Run all demos
    metadata = demo_basic_metadata_creation()
    demo_serialization(metadata)
    demo_cryptographic_binding()
    demo_factory_function()
    demo_schema_validation()
    demo_multiple_regions()

    print("\n" + "=" * 70)
    print("✅ All demos completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
