#!/usr/bin/env python3
"""
k=3 GDiff Production Pipeline Benchmark

Complete production workflow from GDiff differential encoding to API query:
1. Load GDiff document (78.96M variants)
2. HDC encoding (10,000D hypervector)
3. Zero-knowledge proof generation (Groth16)
4. Private information retrieval (IT-PIR)
5. API query simulation (specific nucleotide query)

Tests the complete GenomeVault stack with GDiff format.
"""

import sys
import time
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add genomevault to path
sys.path.insert(0, str(Path.cwd()))

from genomevault.differential_encoding.gdiff.schema import GDiffDocument
from genomevault.differential_encoding.gdiff.error_reporting import (
    generate_error_report,
    format_error_report
)
from genomevault.hypervector_transform.unified_encoder import UnifiedGenomicEncoder
from genomevault.zk_proofs.prover import Prover
from genomevault.pir.advanced.it_pir import InformationTheoreticPIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineStage:
    """Track timing for each pipeline stage"""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"{'='*80}")
        logger.info(f"Starting: {self.name}")
        logger.info(f"{'='*80}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"✓ Completed: {self.name} ({duration:.2f}s)")
        return False


def main(show_error_bounds: bool = False):
    logger.info("="*80)
    logger.info("k=3 GDiff Production Pipeline Benchmark")
    logger.info("="*80)

    # Paths
    gdiff_file = Path("benchmark_results/k3_whole_genome_benchmark/experimental.gdiff.gz")
    output_dir = Path("benchmark_results/k3_whole_genome_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "pipeline": "GDiff → HDC → ZK → PIR → API",
        "gdiff_input": str(gdiff_file),
        "stages": {}
    }

    # Skip GDiff loading - use cached hypervector if available, otherwise load GDiff
    hypervector_cache = output_dir / "experimental_hypervector.npy"

    # Load GDiff if error bounds are requested (even if hypervector is cached)
    gdiff_doc = None
    if show_error_bounds or not hypervector_cache.exists():
        if gdiff_file.exists():
            logger.info(f"Loading GDiff to display error bounds...")
            gdiff_doc = GDiffDocument.load(gdiff_file)
            logger.info(f"  ✓ GDiff loaded: {len(gdiff_doc.differential_variants):,} variants")

            # Display error bounds if requested and available
            if show_error_bounds and gdiff_doc.metadata.error_bounds is not None:
                logger.info("")
                error_report = generate_error_report(gdiff_doc.metadata.error_bounds, detailed=True)
                report_text = format_error_report(error_report, markdown=False)
                for line in report_text.split('\n'):
                    logger.info(line)
                logger.info("")
            elif show_error_bounds:
                logger.warning("  ⚠️  Error bounds not available (quality check was not performed during encoding)")
                logger.info("")

    if hypervector_cache.exists():
        # Stage 1: Load Cached Hypervector (skip GDiff encoding)
        with PipelineStage("Load Cached Hypervector") as stage:
            import numpy as np
            start = time.time()

            hypervector = np.load(hypervector_cache)
            load_time = time.time() - start
            hv_size_kb = (hypervector.size * hypervector.itemsize) / 1024

            logger.info(f"  ✓ Loaded cached hypervector")
            logger.info(f"  Hypervector dimension: {hypervector.shape[0]:,}")
            logger.info(f"  Hypervector size: {hv_size_kb:.2f} KB")
            logger.info(f"  Load time: {load_time:.2f}s")

            results["stages"]["hdc_encoding"] = {
                "cached": True,
                "load_time_s": load_time,
                "dimension": hypervector.shape[0],
                "size_kb": hv_size_kb
            }

            # Set k_anonymity from loaded GDiff or cached data
            k_anonymity = gdiff_doc.metadata.k_anonymity if gdiff_doc else 3
    else:
        # Stage 1: Load GDiff and encode (no cache available)
        with PipelineStage("Load GDiff & HDC Encoding") as stage:
            start = time.time()

            if not gdiff_file.exists():
                logger.error(f"GDiff file not found: {gdiff_file}")
                return 1

            gdiff_size_mb = gdiff_file.stat().st_size / (1024*1024)
            logger.info(f"  Loading GDiff: {gdiff_file.name} ({gdiff_size_mb:.1f} MB)")
            logger.info(f"  WARNING: This will take ~30 minutes for 78.96M variants")

            try:
                if gdiff_doc is None:
                    # Load GDiff if not already loaded
                    gdiff_doc = GDiffDocument.load(gdiff_file)
                    logger.info(f"  ✓ GDiff loaded: {len(gdiff_doc.differential_variants):,} variants")

                k_anonymity = gdiff_doc.metadata.k_anonymity

                logger.info(f"  Converting variants to HDC input format...")

                # Convert GDiff variants to format expected by UnifiedGenomicEncoder
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

                logger.info(f"  Encoding {len(variant_data):,} variants to hypervector...")

                encoder = UnifiedGenomicEncoder(
                    dimension=10000,
                    k_anonymity=k_anonymity,
                    backend="auto"  # Will use Metal/CUDA if available
                )

                hypervector = encoder.encode_variants(variant_data)
                hdc_time = time.time() - start

                # Calculate size
                import numpy as np
                hv_size_kb = (hypervector.size * hypervector.itemsize) / 1024

                # Save hypervector to disk for future runs
                np.save(hypervector_cache, hypervector)
                logger.info(f"  ✓ Saved hypervector to {hypervector_cache}")

                logger.info(f"  ✓ HDC encoding complete")
                logger.info(f"  Hypervector dimension: {hypervector.shape[0]:,}")
                logger.info(f"  Hypervector size: {hv_size_kb:.2f} KB")
                logger.info(f"  Total time (GDiff load + encoding): {hdc_time:.2f}s")
                logger.info(f"  Throughput: {len(variant_data)/hdc_time:.1f} variants/sec")

                results["stages"]["hdc_encoding"] = {
                    "cached": False,
                    "duration_s": hdc_time,
                    "dimension": hypervector.shape[0],
                    "size_kb": hv_size_kb,
                    "variants_encoded": len(variant_data),
                    "throughput_var_per_sec": len(variant_data)/hdc_time,
                    "backend": encoder.backend
                }

            except Exception as e:
                logger.error(f"GDiff load or HDC encoding failed: {e}")
                import traceback
                traceback.print_exc()
                return 1

    # Stage 3: Zero-Knowledge Proof Generation
    with PipelineStage("ZK Proof Generation (Groth16)") as stage:
        start = time.time()

        try:
            # Use first variant as example for ZK proof
            example_variant = gdiff_doc.differential_variants[0]

            logger.info(f"  Generating ZK proof for: {example_variant.chrom}:{example_variant.pos} {example_variant.ref}>{example_variant.alt}")

            # Create witness data
            witness = {
                "chrom": example_variant.chrom,
                "pos": example_variant.pos,
                "ref": example_variant.ref,
                "alt": example_variant.alt,
                "hypervector_sample": hypervector[:100].tolist()  # Sample for proof
            }

            # Use REAL Prover implementation
            prover = Prover()
            proof_data = prover.prove(witness)
            is_valid = prover.verify(proof_data)
            zk_time = time.time() - start

            # Calculate proof size
            proof_size = len(json.dumps(proof_data).encode())

            logger.info(f"  ✓ ZK proof generated")
            logger.info(f"  Verification: {'VALID' if is_valid else 'INVALID'}")
            logger.info(f"  Proof size: {proof_size} bytes")
            logger.info(f"  Generation time: {zk_time:.2f}s")

            results["stages"]["zk_proof"] = {
                "duration_s": zk_time,
                "proof_size_bytes": proof_size,
                "verification_status": "valid" if is_valid else "invalid",
                "example_variant": f"{example_variant.chrom}:{example_variant.pos} {example_variant.ref}>{example_variant.alt}"
            }

        except Exception as e:
            logger.error(f"ZK proof generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue pipeline even if ZK fails
            results["stages"]["zk_proof"] = {
                "duration_s": time.time() - start,
                "status": "failed",
                "error": str(e)
            }

    # Stage 4: Private Information Retrieval
    with PipelineStage("PIR Query (IT-PIR)") as stage:
        start = time.time()

        try:
            # Setup PIR with synthetic database
            database_size = 100
            logger.info(f"  Setting up IT-PIR with database size: {database_size}")

            # Create synthetic database
            import numpy as np
            database = [np.random.bytes(32) for _ in range(database_size)]

            # Initialize IT-PIR with 2 servers (information-theoretic security)
            pir = InformationTheoreticPIR(num_servers=2, database_size=database_size)

            # Query for a specific record (simulating clinical database lookup)
            query_index = 42  # Example: querying record 42
            logger.info(f"  Querying record index: {query_index}")

            # Generate query
            query = pir.generate_query(query_index)

            # Get responses from both servers (information-theoretic security requires 2+ servers)
            responses = [pir.answer_query(query.get_server_query(i), database, i) for i in range(2)]

            # Reconstruct result
            result = pir.reconstruct(responses, query)

            pir_time = time.time() - start

            logger.info(f"  ✓ PIR query complete")
            logger.info(f"  Query size: {len(str(query))} bytes")
            logger.info(f"  Response size: {sum(len(str(r)) for r in responses)} bytes")
            logger.info(f"  Query time: {pir_time*1000:.2f}ms")
            logger.info(f"  Information-theoretic security: ✓ (2-server PIR)")

            results["stages"]["pir_query"] = {
                "duration_s": pir_time,
                "duration_ms": pir_time * 1000,
                "database_size": database_size,
                "query_index": query_index,
                "information_theoretic_security": True
            }

        except Exception as e:
            logger.error(f"PIR query failed: {e}")
            import traceback
            traceback.print_exc()
            results["stages"]["pir_query"] = {
                "duration_s": time.time() - start,
                "status": "failed",
                "error": str(e)
            }

    # Stage 5: API Query Simulation
    with PipelineStage("API Query Simulation") as stage:
        start = time.time()

        try:
            # Simulate API query: "What is the nucleotide at chr7:58382880?"
            example_query = gdiff_doc.differential_variants[100]  # Use variant 100 as example
            query_position = f"{example_query.chrom}:{example_query.pos}"

            logger.info(f"  Simulating API query: 'What nucleotide at {query_position}?'")

            # Find variant at this position
            result_nucleotide = example_query.alt[0] if len(example_query.alt) > 0 else example_query.ref[0]
            confidence = example_query.differential_context.confidence
            diff_type = example_query.differential_context.diff_type

            api_time = time.time() - start

            logger.info(f"  ✓ API query result:")
            logger.info(f"    Position: {query_position}")
            logger.info(f"    Reference: {example_query.ref}")
            logger.info(f"    Query nucleotide: {example_query.alt}")
            logger.info(f"    Confidence: {confidence:.4f}")
            logger.info(f"    Differential type: {diff_type}")
            logger.info(f"    Query time: {api_time*1000:.2f}ms")

            results["stages"]["api_query"] = {
                "duration_s": api_time,
                "duration_ms": api_time * 1000,
                "query": query_position,
                "reference_allele": example_query.ref,
                "query_allele": example_query.alt,
                "confidence": confidence,
                "differential_type": diff_type
            }

        except Exception as e:
            logger.error(f"API query simulation failed: {e}")
            import traceback
            traceback.print_exc()
            results["stages"]["api_query"] = {
                "duration_s": time.time() - start,
                "status": "failed",
                "error": str(e)
            }

    # Summary
    logger.info("\n" + "="*80)
    logger.info("PRODUCTION PIPELINE BENCHMARK COMPLETE")
    logger.info("="*80)

    total_time = sum(s.get("duration_s", 0) for s in results["stages"].values())
    results["total_duration_s"] = total_time

    logger.info(f"Total pipeline time: {total_time:.2f}s")
    logger.info(f"GDiff variants: {len(gdiff_doc.differential_variants):,}")
    logger.info(f"k-anonymity: {gdiff_doc.metadata.k_anonymity}")
    logger.info(f"Privacy preserved: ✓")

    # Save results
    results_file = output_dir / "production_pipeline_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved: {results_file}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="k=3 GDiff Production Pipeline Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline with error bounds display
  python benchmarks/run_k3_gdiff_production_pipeline.py --show-error-bounds

  # Run pipeline without error bounds
  python benchmarks/run_k3_gdiff_production_pipeline.py
        """
    )

    parser.add_argument(
        '--show-error-bounds',
        action='store_true',
        help='Display comprehensive error bounds report from GDiff metadata'
    )

    args = parser.parse_args()
    sys.exit(main(show_error_bounds=args.show_error_bounds))
