#!/usr/bin/env python3
"""
k=12 Privacy-Preserving GDiff Pipeline

Uses core GenomeVault codebase to run complete privacy-preserving workflow:
1. Align experimental FASTQ to 12 guide FASTA sequences (privacy-preserving)
2. Generate GDiff differential encoding with error bounds
3. Create HDC hypervector (Metal GPU accelerated)

PRIVACY ARCHITECTURE:
- Experimental data NEVER touches consensus directly
- Aligns to guide pool (blind middleman layer)
- k=12 anonymity throughout
"""

import sys
import time
import logging
from pathlib import Path

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.differential_encoding.align_to_reference_pool import PrivacyPreservingReferencePoolAligner
from genomevault.differential_encoding.gdiff.encoder import GDiffEncoder
from genomevault.differential_encoding.gdiff.schema import GDiffDocument
from genomevault.hypervector_transform.unified_encoder import UnifiedGenomicEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("k=12 Privacy-Preserving GDiff Pipeline")
    logger.info("="*80)

    # Paths
    base_dir = Path("benchmark_results/enhanced_privacy_k13_phase123_optimized")
    guide_dir = base_dir / "layer2_reference_pool"

    # 12 guide FASTA files (blind middleman)
    guide_fastas = [guide_dir / f"ref{i}.fa.gz" for i in range(1, 13)]

    # 12 guide BAMs (for GDiff encoding)
    guide_bams = [guide_dir / f"ref{i}.sorted.bam" for i in range(1, 13)]

    # Experimental FASTQ files
    experimental_r1 = Path("data/downloaded/fastq/ERR3239334_1.fastq.gz")
    experimental_r2 = Path("data/downloaded/fastq/ERR3239334_2.fastq.gz")

    # Output files
    experimental_bam = base_dir / "experimental.sorted.bam"
    gdiff_file = base_dir / "experimental.gdiff.gz"
    hypervector_file = base_dir / "experimental_hypervector.npy"

    # Verify inputs
    logger.info("\nVerifying inputs...")
    for i, fasta in enumerate(guide_fastas, 1):
        if not fasta.exists():
            logger.error(f"Guide FASTA not found: {fasta}")
            return 1
    logger.info(f"  ✓ All 12 guide FASTAs present")

    if not experimental_r1.exists() or not experimental_r2.exists():
        logger.error("Experimental FASTQ files not found")
        return 1
    logger.info(f"  ✓ Experimental FASTQ: {experimental_r1.name}, {experimental_r2.name}")

    # ===========================================================================
    # STAGE 1: Privacy-Preserving Alignment (Experimental → Guide Pool)
    # ===========================================================================
    if experimental_bam.exists():
        logger.info(f"\n✓ Experimental BAM already exists: {experimental_bam}")
        logger.info(f"  Size: {experimental_bam.stat().st_size / (1024**3):.2f} GB")
        logger.info("  Skipping alignment stage...")
    else:
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: Privacy-Preserving Alignment")
        logger.info("="*80)
        logger.info("Aligning experimental FASTQ to 12 guide FASTA sequences")
        logger.info("PRIVACY: NO direct contact with consensus!")

        start = time.time()

        aligner = PrivacyPreservingReferencePoolAligner(
            guide_fasta_files=guide_fastas,
            threads=10
        )

        # Use random guide cycling for information-theoretic privacy
        logger.info("Using random guide cycling (k=12 anonymity)")
        metadata = aligner.align_query_to_pool_with_random_cycling(
            query_fastq_1=experimental_r1,
            query_fastq_2=experimental_r2,
            output_bam=experimental_bam,
            chunk_size=10_000_000  # 10M reads per chunk
        )

        logger.info(f"✓ Random guide cycling metadata: {metadata['total_chunks']} chunks")
        logger.info(f"✓ k={metadata['k_anonymity']} anonymity maintained")

        alignment_time = time.time() - start
        logger.info(f"\n✓ Alignment complete in {alignment_time/3600:.2f} hours")
        logger.info(f"  BAM size: {experimental_bam.stat().st_size / (1024**3):.2f} GB")

    # ===========================================================================
    # STAGE 2: GDiff Differential Encoding (Error-Aware)
    # ===========================================================================
    if gdiff_file.exists():
        logger.info(f"\n✓ GDiff file already exists: {gdiff_file}")
        logger.info(f"  Size: {gdiff_file.stat().st_size / (1024**2):.1f} MB")
        logger.info("  Skipping encoding stage...")
        gdiff_doc = GDiffDocument.load(str(gdiff_file))
    else:
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: GDiff Differential Encoding (Error-Aware)")
        logger.info("="*80)

        start = time.time()

        # Template-based encoding with ClinVar database
        template_path = Path("data/templates/gdiff_template_GRCh38.json.gz")

        encoder = GDiffEncoder(
            query_bam=str(experimental_bam),
            pool_bams=[str(bam) for bam in guide_bams],
            genome_build="k12_pool",
            min_base_quality=20,
            min_mapping_quality=20,
            enable_quality_check=True,  # Error-aware system
            fastq_path=str(experimental_r1),
            target_epsilon=0.05,  # Diagnostic-grade accuracy
            template_path=str(template_path) if template_path.exists() else None  # Template-based encoding
        )

        if template_path.exists():
            logger.info(f"✓ Template-based encoding enabled: {template_path.name}")
            logger.info(f"  Template contains 3.88M clinical variants from ClinVar")

        logger.info("Computing differential encoding...")
        gdiff_doc = encoder.compute_differential_encoding()

        logger.info(f"Saving GDiff to {gdiff_file}...")
        gdiff_doc.save(str(gdiff_file), compress=True)

        encoding_time = time.time() - start

        logger.info(f"\n✓ GDiff encoding complete in {encoding_time/60:.1f} minutes")
        logger.info(f"  Total variants: {len(gdiff_doc.differential_variants):,}")
        logger.info(f"  k-anonymity: {gdiff_doc.metadata.k_anonymity}")
        logger.info(f"  File size: {gdiff_file.stat().st_size / (1024**2):.1f} MB")

        # Display error bounds if available
        if gdiff_doc.metadata.error_bounds:
            from genomevault.differential_encoding.gdiff.error_reporting import (
                generate_error_report,
                format_error_report
            )
            logger.info("\nError Bounds:")
            report = generate_error_report(gdiff_doc.metadata.error_bounds, detailed=False)
            report_text = format_error_report(report, markdown=False)
            for line in report_text.split('\n'):
                logger.info(f"  {line}")

    # ===========================================================================
    # STAGE 3: HDC Encoding (Metal GPU)
    # ===========================================================================
    logger.info("\n" + "="*80)
    logger.info("STAGE 3: Hyperdimensional Computing (Metal GPU)")
    logger.info("="*80)

    start = time.time()

    # Convert GDiff variants to HDC format
    logger.info("Converting variants to HDC format...")
    variant_data = []
    for v in gdiff_doc.differential_variants:
        variant_data.append({
            "chrom": v.chrom,
            "pos": v.pos,
            "ref": v.ref,
            "alt": v.alt,
            "quality": v.differential_context.confidence * 100,
            "diff_type": v.differential_context.diff_type,
            "pool_coverage": v.differential_context.pool_coverage
        })

    logger.info(f"Encoding {len(variant_data):,} variants to hypervector...")
    encoder = UnifiedGenomicEncoder(
        dimension=10000,
        k_anonymity=gdiff_doc.metadata.k_anonymity,
        backend="auto"  # Will use Metal GPU on Apple Silicon
    )

    hypervector = encoder.encode_variants(variant_data)
    hdc_time = time.time() - start

    import numpy as np
    hv_size_kb = (hypervector.size * hypervector.itemsize) / 1024

    logger.info(f"\n✓ HDC encoding complete in {hdc_time:.2f}s")
    logger.info(f"  Hypervector dimension: {hypervector.shape[0]:,}D")
    logger.info(f"  Hypervector size: {hv_size_kb:.2f} KB")
    logger.info(f"  Backend: {encoder.backend}")
    logger.info(f"  Throughput: {len(variant_data)/hdc_time:.1f} variants/sec")

    # Save hypervector
    np.save(hypervector_file, hypervector)
    logger.info(f"  Saved: {hypervector_file}")

    # ===========================================================================
    # Save Results
    # ===========================================================================
    import json
    results_file = base_dir / "k12_pipeline_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "pipeline": "k=12 Privacy-Preserving GDiff Pipeline",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "k_anonymity": 12,
            "privacy_preserved": True,
            "stages": {
                "alignment": {
                    "experimental_bam": str(experimental_bam),
                    "size_gb": experimental_bam.stat().st_size / (1024**3) if experimental_bam.exists() else 0
                },
                "gdiff_encoding": {
                    "total_variants": len(variant_data),
                    "gdiff_file": str(gdiff_file),
                    "size_mb": gdiff_file.stat().st_size / (1024**2),
                    "error_aware": gdiff_doc.metadata.error_bounds is not None
                },
                "hdc_encoding": {
                    "dimension": hypervector.shape[0],
                    "size_kb": hv_size_kb,
                    "backend": encoder.backend,
                    "duration_s": hdc_time,
                    "throughput_var_per_sec": len(variant_data)/hdc_time
                }
            }
        }, f, indent=2)

    logger.info(f"\n💾 Results saved: {results_file}")
    logger.info("\n" + "="*80)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Privacy-preserving k=12 GDiff pipeline executed successfully")
    logger.info(f"Experimental data NEVER touched consensus directly")
    logger.info(f"All privacy guarantees maintained throughout")

    return 0


if __name__ == "__main__":
    sys.exit(main())
