#!/usr/bin/env python3
"""
Generate Demo Data for GenomeVault

This script generates comprehensive synthetic genomic data for testing:
- Genomic variants (SNPs, INDELs, CNVs)
- VCF files with realistic variant calls
- Mock FAST5 files for nanopore sequencing
- PostgreSQL sample records
- Test hypervectors
- Sample ZK proofs

All data is generated deterministically using fixed seeds for reproducibility.
"""

import argparse
import hashlib
import json
import os
import random
import sqlite3
import struct
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


# Constants for deterministic generation
SEED = 42
CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
NUCLEOTIDES = ["A", "T", "G", "C"]
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Common disease variants for testing
DISEASE_VARIANTS = {
    "BRCA1": {"chr": "17", "pos": 41276045, "ref": "C", "alt": "T", "disease": "Breast Cancer"},
    "BRCA2": {"chr": "13", "pos": 32914438, "ref": "G", "alt": "A", "disease": "Breast Cancer"},
    "APOE4": {"chr": "19", "pos": 45411941, "ref": "T", "alt": "C", "disease": "Alzheimer"},
    "HFE": {"chr": "6", "pos": 26093141, "ref": "G", "alt": "A", "disease": "Hemochromatosis"},
    "CFTR": {"chr": "7", "pos": 117559590, "ref": "CTT", "alt": "C", "disease": "Cystic Fibrosis"},
}


@dataclass
class GenomicVariant:
    """Represents a genomic variant."""

    chromosome: str
    position: int
    ref_allele: str
    alt_allele: str
    variant_type: str  # SNP, INDEL, CNV, SV
    quality: float
    depth: int
    allele_frequency: float
    gene: Optional[str] = None
    impact: Optional[str] = None  # HIGH, MODERATE, LOW, MODIFIER
    clinical_significance: Optional[str] = None
    dbsnp_id: Optional[str] = None


class DemoDataGenerator:
    """Generate synthetic genomic demo data."""

    def __init__(self, seed: int = SEED, output_dir: str = "demo_data"):
        """Initialize generator with fixed seed."""
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        logger.info(f"Initialized demo data generator with seed {seed}")
        logger.info(f"Output directory: {self.output_dir}")

    def generate_snp(self, chromosome: str = None) -> GenomicVariant:
        """Generate a random SNP."""
        if not chromosome:
            chromosome = random.choice(CHROMOSOMES[:-1])  # Exclude MT

        position = random.randint(1000000, 250000000)
        ref = random.choice(NUCLEOTIDES)
        alt = random.choice([n for n in NUCLEOTIDES if n != ref])

        return GenomicVariant(
            chromosome=chromosome,
            position=position,
            ref_allele=ref,
            alt_allele=alt,
            variant_type="SNP",
            quality=random.uniform(20, 100),
            depth=random.randint(10, 100),
            allele_frequency=random.uniform(0.001, 0.5),
            gene=f"GENE{random.randint(1, 10000)}",
            impact=random.choice(["HIGH", "MODERATE", "LOW", "MODIFIER"]),
            dbsnp_id=f"rs{random.randint(1000000, 99999999)}",
        )

    def generate_indel(self) -> GenomicVariant:
        """Generate a random INDEL."""
        chromosome = random.choice(CHROMOSOMES[:-1])
        position = random.randint(1000000, 250000000)

        # Insertion or deletion
        if random.random() < 0.5:
            # Deletion
            ref_len = random.randint(1, 10)
            ref = "".join(random.choices(NUCLEOTIDES, k=ref_len))
            alt = ref[0]  # Keep first base
        else:
            # Insertion
            ref = random.choice(NUCLEOTIDES)
            alt_len = random.randint(2, 10)
            alt = ref + "".join(random.choices(NUCLEOTIDES, k=alt_len - 1))

        return GenomicVariant(
            chromosome=chromosome,
            position=position,
            ref_allele=ref,
            alt_allele=alt,
            variant_type="INDEL",
            quality=random.uniform(15, 80),
            depth=random.randint(8, 80),
            allele_frequency=random.uniform(0.001, 0.3),
            gene=f"GENE{random.randint(1, 10000)}",
            impact=random.choice(["HIGH", "MODERATE", "LOW"]),
        )

    def generate_cnv(self) -> GenomicVariant:
        """Generate a Copy Number Variant."""
        chromosome = random.choice(CHROMOSOMES[:-2])  # Exclude sex chromosomes
        start_pos = random.randint(1000000, 200000000)
        cnv_length = random.randint(1000, 1000000)

        return GenomicVariant(
            chromosome=chromosome,
            position=start_pos,
            ref_allele="N",
            alt_allele=f"<CN{random.randint(0, 5)}>",
            variant_type="CNV",
            quality=random.uniform(10, 60),
            depth=random.randint(20, 100),
            allele_frequency=random.uniform(0.01, 0.2),
            impact="MODERATE",
        )

    def generate_clinical_variant(self) -> GenomicVariant:
        """Generate a clinically significant variant."""
        variant_key = random.choice(list(DISEASE_VARIANTS.keys()))
        var = DISEASE_VARIANTS[variant_key]

        return GenomicVariant(
            chromosome=var["chr"],
            position=var["pos"],
            ref_allele=var["ref"],
            alt_allele=var["alt"],
            variant_type="SNP" if len(var["ref"]) == 1 else "INDEL",
            quality=random.uniform(30, 99),
            depth=random.randint(20, 100),
            allele_frequency=random.uniform(0.01, 0.5),
            gene=variant_key,
            impact="HIGH",
            clinical_significance=var["disease"],
        )

    def generate_variants(self, count: int = 1000) -> List[GenomicVariant]:
        """Generate a mixed set of variants."""
        variants = []

        # Distribution of variant types
        snp_count = int(count * 0.7)
        indel_count = int(count * 0.2)
        cnv_count = int(count * 0.05)
        clinical_count = int(count * 0.05)

        logger.info(f"Generating {count} variants...")

        # Generate SNPs
        for _ in range(snp_count):
            variants.append(self.generate_snp())

        # Generate INDELs
        for _ in range(indel_count):
            variants.append(self.generate_indel())

        # Generate CNVs
        for _ in range(cnv_count):
            variants.append(self.generate_cnv())

        # Generate clinical variants
        for _ in range(clinical_count):
            variants.append(self.generate_clinical_variant())

        # Sort by chromosome and position
        variants.sort(key=lambda v: (v.chromosome, v.position))

        logger.info(f"Generated {len(variants)} variants")
        return variants

    def write_vcf(self, variants: List[GenomicVariant], filename: str = "demo.vcf"):
        """Write variants to VCF format."""
        vcf_path = self.output_dir / filename

        logger.info(f"Writing VCF file: {vcf_path}")

        with open(vcf_path, "w") as f:
            # Write VCF header
            f.write("##fileformat=VCFv4.3\n")
            f.write(f"##fileDate={datetime.now().strftime('%Y%m%d')}\n")
            f.write("##source=GenomeVault_DemoDataGenerator\n")
            f.write(f"##reference=GRCh38\n")

            # INFO fields
            f.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">\n')
            f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Read Depth">\n')
            f.write('##INFO=<ID=GENE,Number=1,Type=String,Description="Gene Name">\n')
            f.write('##INFO=<ID=IMPACT,Number=1,Type=String,Description="Variant Impact">\n')
            f.write('##INFO=<ID=TYPE,Number=1,Type=String,Description="Variant Type">\n')

            # FORMAT fields
            f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            f.write('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n')
            f.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">\n')

            # Contig lines
            for chrom in CHROMOSOMES[:-1]:
                f.write(f"##contig=<ID={chrom}>\n")

            # Column header
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE001\n")

            # Write variants
            for var in variants:
                # ID field
                id_field = var.dbsnp_id if var.dbsnp_id else "."

                # INFO field
                info_parts = [
                    f"AF={var.allele_frequency:.4f}",
                    f"DP={var.depth}",
                    f"TYPE={var.variant_type}",
                ]
                if var.gene:
                    info_parts.append(f"GENE={var.gene}")
                if var.impact:
                    info_parts.append(f"IMPACT={var.impact}")
                info_field = ";".join(info_parts)

                # FORMAT and sample fields
                format_field = "GT:GQ:DP"

                # Random genotype
                if random.random() < var.allele_frequency:
                    if random.random() < 0.5:
                        genotype = "0/1"  # Heterozygous
                    else:
                        genotype = "1/1"  # Homozygous alt
                else:
                    genotype = "0/0"  # Homozygous ref

                sample_field = f"{genotype}:{int(var.quality)}:{var.depth}"

                # Write line
                f.write(f"{var.chromosome}\t{var.position}\t{id_field}\t")
                f.write(f"{var.ref_allele}\t{var.alt_allele}\t{var.quality:.1f}\t")
                f.write(f"PASS\t{info_field}\t{format_field}\t{sample_field}\n")

        logger.info(f"Wrote {len(variants)} variants to {vcf_path}")
        return vcf_path

    def generate_mock_fast5(self, read_count: int = 100):
        """Generate mock FAST5-like data structure."""
        fast5_dir = self.output_dir / "fast5"
        fast5_dir.mkdir(exist_ok=True)

        logger.info(f"Generating {read_count} mock FAST5 reads...")

        reads = []
        for i in range(read_count):
            read_id = f"read_{i:06d}"

            # Generate random sequence
            seq_length = random.randint(500, 5000)
            sequence = "".join(random.choices(NUCLEOTIDES, k=seq_length))

            # Generate mock quality scores
            quality = np.random.randint(10, 40, size=seq_length)

            # Generate mock signal data (simplified)
            signal_length = seq_length * random.randint(8, 12)
            signal = np.random.normal(100, 20, signal_length).astype(np.float32)

            # Create read metadata
            read_data = {
                "read_id": read_id,
                "sequence": sequence,
                "quality": quality.tolist(),
                "signal_mean": float(np.mean(signal)),
                "signal_std": float(np.std(signal)),
                "sequence_length": seq_length,
                "signal_length": signal_length,
                "channel": random.randint(1, 512),
                "mux": random.randint(1, 4),
                "start_time": i * 1000,
                "duration": seq_length * 0.4,
                "sampling_rate": 4000,
                "median_before": random.uniform(60, 80),
                "median_after": random.uniform(60, 80),
            }

            reads.append(read_data)

            # Save individual read file
            read_file = fast5_dir / f"{read_id}.json"
            with open(read_file, "w") as f:
                json.dump(read_data, f, indent=2)

        # Save summary
        summary_file = fast5_dir / "reads_summary.json"
        with open(summary_file, "w") as f:
            summary = {
                "total_reads": read_count,
                "total_bases": sum(r["sequence_length"] for r in reads),
                "mean_length": np.mean([r["sequence_length"] for r in reads]),
                "generated_at": datetime.now().isoformat(),
                "seed": self.seed,
            }
            json.dump(summary, f, indent=2)

        logger.info(f"Generated {read_count} mock FAST5 reads in {fast5_dir}")
        return reads

    def create_sqlite_db(self, variants: List[GenomicVariant]):
        """Create SQLite database with variant data (PostgreSQL alternative)."""
        db_path = self.output_dir / "genomevault_demo.db"

        logger.info(f"Creating SQLite database: {db_path}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create variants table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS variants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chromosome TEXT NOT NULL,
                position INTEGER NOT NULL,
                ref_allele TEXT NOT NULL,
                alt_allele TEXT NOT NULL,
                variant_type TEXT,
                quality REAL,
                depth INTEGER,
                allele_frequency REAL,
                gene TEXT,
                impact TEXT,
                clinical_significance TEXT,
                dbsnp_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chromosome, position, ref_allele, alt_allele)
            )
        """
        )

        # Create index
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_variants_chr_pos
            ON variants(chromosome, position)
        """
        )

        # Insert variants
        for var in variants:
            cursor.execute(
                """
                INSERT OR IGNORE INTO variants
                (chromosome, position, ref_allele, alt_allele, variant_type,
                 quality, depth, allele_frequency, gene, impact,
                 clinical_significance, dbsnp_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    var.chromosome,
                    var.position,
                    var.ref_allele,
                    var.alt_allele,
                    var.variant_type,
                    var.quality,
                    var.depth,
                    var.allele_frequency,
                    var.gene,
                    var.impact,
                    var.clinical_significance,
                    var.dbsnp_id,
                ),
            )

        # Create samples table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT UNIQUE NOT NULL,
                patient_id TEXT,
                collection_date DATE,
                tissue_type TEXT,
                sequencing_platform TEXT,
                coverage REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Insert sample data
        for i in range(10):
            cursor.execute(
                """
                INSERT OR IGNORE INTO samples
                (sample_id, patient_id, collection_date, tissue_type,
                 sequencing_platform, coverage)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    f"SAMPLE_{i:04d}",
                    f"PATIENT_{i:03d}",
                    (datetime.now() - timedelta(days=random.randint(1, 365))).date(),
                    random.choice(["Blood", "Tumor", "Saliva", "Tissue"]),
                    random.choice(["Illumina", "PacBio", "Nanopore"]),
                    random.uniform(20, 100),
                ),
            )

        # Create genotypes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS genotypes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id INTEGER,
                variant_id INTEGER,
                genotype TEXT,
                quality INTEGER,
                read_depth INTEGER,
                FOREIGN KEY (sample_id) REFERENCES samples(id),
                FOREIGN KEY (variant_id) REFERENCES variants(id),
                UNIQUE(sample_id, variant_id)
            )
        """
        )

        conn.commit()

        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM variants")
        variant_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM samples")
        sample_count = cursor.fetchone()[0]

        conn.close()

        logger.info(f"Created database with {variant_count} variants and {sample_count} samples")
        return db_path

    def generate_hypervectors(self, variants: List[GenomicVariant], dimension: int = 10000):
        """Generate test hypervectors from variants."""
        hv_dir = self.output_dir / "hypervectors"
        hv_dir.mkdir(exist_ok=True)

        logger.info(f"Generating hypervectors (dimension={dimension})...")

        hypervectors = []

        for i, var in enumerate(variants[:100]):  # First 100 variants
            # Create a deterministic hypervector based on variant properties
            # Use hash for deterministic randomness
            var_str = f"{var.chromosome}_{var.position}_{var.ref_allele}_{var.alt_allele}"
            var_hash = int(hashlib.md5(var_str.encode()).hexdigest(), 16)

            # Set random seed based on hash
            np.random.seed(var_hash % (2**32))

            # Generate hypervector
            # Binary hypervector with ~50% sparsity
            hv = np.random.choice([0, 1], size=dimension, p=[0.5, 0.5])

            # Store metadata
            hv_data = {
                "variant_id": f"{var.chromosome}:{var.position}:{var.ref_allele}>{var.alt_allele}",
                "dimension": dimension,
                "sparsity": float(np.mean(hv == 0)),
                "hamming_weight": int(np.sum(hv)),
                "hypervector": hv.tolist() if i < 10 else None,  # Only store first 10 full vectors
                "checksum": hashlib.sha256(hv.tobytes()).hexdigest(),
            }

            hypervectors.append(hv_data)

            # Save individual hypervector
            if i < 10:
                hv_file = hv_dir / f"hypervector_{i:04d}.json"
                with open(hv_file, "w") as f:
                    json.dump(hv_data, f, indent=2)

        # Save summary
        summary_file = hv_dir / "hypervectors_summary.json"
        with open(summary_file, "w") as f:
            summary = {
                "total_vectors": len(hypervectors),
                "dimension": dimension,
                "average_sparsity": np.mean([hv["sparsity"] for hv in hypervectors]),
                "average_hamming_weight": np.mean([hv["hamming_weight"] for hv in hypervectors]),
                "generated_at": datetime.now().isoformat(),
            }
            json.dump(summary, f, indent=2)

        logger.info(f"Generated {len(hypervectors)} hypervectors")
        return hypervectors

    def generate_zk_proofs(self, count: int = 10):
        """Generate sample ZK proof data."""
        zk_dir = self.output_dir / "zk_proofs"
        zk_dir.mkdir(exist_ok=True)

        logger.info(f"Generating {count} sample ZK proofs...")

        proofs = []

        for i in range(count):
            # Generate mock proof data
            proof = {
                "proof_id": f"proof_{i:04d}",
                "circuit_type": random.choice(["sum64", "median_verification", "range_proof"]),
                "public_inputs": {
                    "commitment": hashlib.sha256(f"input_{i}".encode()).hexdigest(),
                    "result": random.randint(1, 1000),
                },
                "proof": {
                    "pi_a": [hex(random.getrandbits(256)), hex(random.getrandbits(256))],
                    "pi_b": [
                        [hex(random.getrandbits(256)), hex(random.getrandbits(256))],
                        [hex(random.getrandbits(256)), hex(random.getrandbits(256))],
                    ],
                    "pi_c": [hex(random.getrandbits(256)), hex(random.getrandbits(256))],
                    "protocol": "groth16",
                    "curve": "bn128",
                },
                "metadata": {
                    "prover": f"client_{i % 3}",
                    "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                    "computation_time_ms": random.uniform(100, 1000),
                    "verified": random.random() > 0.1,  # 90% valid
                },
            }

            proofs.append(proof)

            # Save individual proof
            proof_file = zk_dir / f"proof_{i:04d}.json"
            with open(proof_file, "w") as f:
                json.dump(proof, f, indent=2)

        # Save summary
        summary_file = zk_dir / "proofs_summary.json"
        with open(summary_file, "w") as f:
            summary = {
                "total_proofs": count,
                "circuit_types": list(set(p["circuit_type"] for p in proofs)),
                "valid_proofs": sum(1 for p in proofs if p["metadata"]["verified"]),
                "average_computation_time_ms": np.mean(
                    [p["metadata"]["computation_time_ms"] for p in proofs]
                ),
                "generated_at": datetime.now().isoformat(),
            }
            json.dump(summary, f, indent=2)

        logger.info(f"Generated {count} ZK proofs")
        return proofs

    def generate_clinical_reports(self, variants: List[GenomicVariant], count: int = 5):
        """Generate mock clinical reports."""
        reports_dir = self.output_dir / "clinical_reports"
        reports_dir.mkdir(exist_ok=True)

        logger.info(f"Generating {count} clinical reports...")

        for i in range(count):
            report = {
                "report_id": f"REPORT_{i:04d}",
                "patient_id": f"PATIENT_{i:03d}",
                "sample_id": f"SAMPLE_{i:04d}",
                "report_date": datetime.now().isoformat(),
                "variants_analyzed": len(variants),
                "findings": {
                    "pathogenic": [],
                    "likely_pathogenic": [],
                    "uncertain_significance": [],
                    "likely_benign": [],
                    "benign": [],
                },
                "recommendations": [],
                "pharmacogenomics": [],
            }

            # Add some clinical findings
            clinical_vars = [v for v in variants if v.clinical_significance]
            for var in clinical_vars[:5]:
                finding = {
                    "variant": f"{var.chromosome}:{var.position}:{var.ref_allele}>{var.alt_allele}",
                    "gene": var.gene,
                    "significance": var.clinical_significance,
                    "allele_frequency": var.allele_frequency,
                    "zygosity": "Heterozygous" if var.allele_frequency < 0.5 else "Homozygous",
                }

                if var.impact == "HIGH":
                    report["findings"]["pathogenic"].append(finding)
                elif var.impact == "MODERATE":
                    report["findings"]["likely_pathogenic"].append(finding)
                else:
                    report["findings"]["uncertain_significance"].append(finding)

            # Add recommendations
            if report["findings"]["pathogenic"]:
                report["recommendations"].append("Genetic counseling recommended")
                report["recommendations"].append("Consider family screening")

            # Add pharmacogenomics
            pgx_genes = ["CYP2D6", "CYP2C19", "VKORC1", "SLCO1B1"]
            for gene in random.sample(pgx_genes, 2):
                report["pharmacogenomics"].append(
                    {
                        "gene": gene,
                        "phenotype": random.choice(["Normal", "Intermediate", "Poor", "Rapid"]),
                        "medications_affected": random.randint(1, 10),
                    }
                )

            # Save report
            report_file = reports_dir / f"report_{i:04d}.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

        logger.info(f"Generated {count} clinical reports")

    def generate_all(self):
        """Generate all demo data."""
        logger.info("=" * 60)
        logger.info("GENERATING COMPLETE DEMO DATASET")
        logger.info("=" * 60)

        # Generate variants
        variants = self.generate_variants(1000)

        # Write VCF
        vcf_path = self.write_vcf(variants, "demo_variants.vcf")

        # Generate mock FAST5
        fast5_reads = self.generate_mock_fast5(100)

        # Create database
        db_path = self.create_sqlite_db(variants)

        # Generate hypervectors
        hypervectors = self.generate_hypervectors(variants, dimension=10000)

        # Generate ZK proofs
        zk_proofs = self.generate_zk_proofs(10)

        # Generate clinical reports
        self.generate_clinical_reports(variants, 5)

        # Create master manifest
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "seed": self.seed,
            "output_directory": str(self.output_dir),
            "files_generated": {
                "vcf": str(vcf_path),
                "database": str(db_path),
                "fast5_directory": str(self.output_dir / "fast5"),
                "hypervectors_directory": str(self.output_dir / "hypervectors"),
                "zk_proofs_directory": str(self.output_dir / "zk_proofs"),
                "clinical_reports_directory": str(self.output_dir / "clinical_reports"),
            },
            "statistics": {
                "total_variants": len(variants),
                "snps": sum(1 for v in variants if v.variant_type == "SNP"),
                "indels": sum(1 for v in variants if v.variant_type == "INDEL"),
                "cnvs": sum(1 for v in variants if v.variant_type == "CNV"),
                "clinical_variants": sum(1 for v in variants if v.clinical_significance),
                "fast5_reads": len(fast5_reads),
                "hypervectors": len(hypervectors),
                "zk_proofs": len(zk_proofs),
            },
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("=" * 60)
        logger.info("DEMO DATA GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Manifest: {manifest_path}")

        return manifest


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic genomic demo data for GenomeVault"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default="demo_data", help="Output directory (default: demo_data)"
    )
    parser.add_argument(
        "--variants", type=int, default=1000, help="Number of variants to generate (default: 1000)"
    )
    parser.add_argument(
        "--fast5-reads", type=int, default=100, help="Number of mock FAST5 reads (default: 100)"
    )
    parser.add_argument(
        "--zk-proofs", type=int, default=10, help="Number of ZK proofs to generate (default: 10)"
    )

    args = parser.parse_args()

    print("\n🧬 GenomeVault Demo Data Generator")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print(f"Variants: {args.variants}")
    print(f"FAST5 reads: {args.fast5_reads}")
    print(f"ZK proofs: {args.zk_proofs}")
    print("=" * 60)

    # Create generator
    generator = DemoDataGenerator(seed=args.seed, output_dir=args.output)

    # Generate all data
    manifest = generator.generate_all()

    # Print summary
    print("\n✅ Demo data generation complete!")
    print("\nGenerated files:")
    for category, path in manifest["files_generated"].items():
        print(f"  {category}: {path}")

    print("\nStatistics:")
    for stat, value in manifest["statistics"].items():
        print(f"  {stat}: {value}")

    print(f"\nAll data saved to: {args.output}/")
    print(f"Manifest: {args.output}/manifest.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
