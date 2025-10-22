#!/usr/bin/env python3
"""
Example: Using the GenomeVault Sequence Alignment System

This script demonstrates practical usage of the alignment system for:
1. Identifying reference genomes for unknown samples
2. Quality control and validation
3. Integration with differential encoding pipeline
4. Handling ambiguous cases

Usage:
    python alignment_example.py --vcf sample.vcf.gz --references ./references/
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from genomevault.differential_encoding.reference_management import (
    SecureReferenceGenomeManager,
    GenomeSection,
    Variant,
)
from genomevault.differential_encoding.sequence_alignment import (
    create_default_aligner,
    AlignmentStrategy,
    ConsensusResult,
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


def parse_vcf_file(vcf_path: Path) -> Dict[str, List[Variant]]:
    """
    Parse VCF file into variants organized by chromosome.
    
    Args:
        vcf_path: Path to VCF file (.vcf or .vcf.gz)
        
    Returns:
        Dictionary mapping chromosome to list of variants
    """
    import gzip
    
    variants_by_chr: Dict[str, List[Variant]] = {}
    
    # Determine if gzipped
    open_func = gzip.open if vcf_path.suffix == ".gz" else open
    mode = "rt" if vcf_path.suffix == ".gz" else "r"
    
    logger.info(f"Parsing VCF file: {vcf_path}")
    
    with open_func(vcf_path, mode) as f:
        for line in f:
            line = line.strip()
            
            # Skip headers and empty lines
            if line.startswith("#") or not line:
                continue
            
            # Parse variant line
            fields = line.split("\t")
            
            if len(fields) < 8:
                continue
            
            try:
                chromosome = fields[0]
                position = int(fields[1])
                ref = fields[3]
                alt = fields[4]
                qual_str = fields[5]
                filter_str = fields[6]
                
                # Parse quality
                if qual_str == ".":
                    quality = 1.0
                else:
                    quality = float(qual_str)
                    if quality > 100:
                        quality = min(quality / 100.0, 1.0)
                
                # Parse genotype if available
                genotype = "0/1"
                if len(fields) >= 10:
                    format_fields = fields[8].split(":")
                    sample_fields = fields[9].split(":")
                    if "GT" in format_fields:
                        gt_index = format_fields.index("GT")
                        if gt_index < len(sample_fields):
                            genotype = sample_fields[gt_index]
                
                # Create variant
                variant = Variant(
                    chromosome=chromosome,
                    position=position,
                    ref=ref,
                    alt=alt,
                    genotype=genotype,
                    quality=quality,
                    filter=filter_str,
                )
                
                # Add to dictionary
                if chromosome not in variants_by_chr:
                    variants_by_chr[chromosome] = []
                variants_by_chr[chromosome].append(variant)
                
            except Exception as e:
                logger.warning(f"Failed to parse variant: {e}")
                continue
    
    # Sort variants by position
    for chromosome in variants_by_chr:
        variants_by_chr[chromosome].sort(key=lambda v: v.position)
    
    total_variants = sum(len(v) for v in variants_by_chr.values())
    logger.info(
        f"Parsed {total_variants} variants across "
        f"{len(variants_by_chr)} chromosomes"
    )
    
    return variants_by_chr


def align_sample(
    vcf_path: Path,
    reference_dir: Path,
    strategy: AlignmentStrategy = AlignmentStrategy.HYBRID,
    num_references: int = 3,
    output_json: bool = True,
) -> Dict:
    """
    Align a sample VCF file against reference genomes.
    
    Args:
        vcf_path: Path to sample VCF file
        reference_dir: Directory containing reference VCF files
        strategy: Alignment strategy to use
        num_references: Number of references for consensus
        output_json: Whether to output JSON results
        
    Returns:
        Dictionary with alignment results
    """
    logger.info("="*60)
    logger.info("GenomeVault Sequence Alignment")
    logger.info("="*60)
    
    # 1. Load reference genomes
    logger.info(f"Loading reference genomes from: {reference_dir}")
    ref_manager = SecureReferenceGenomeManager(reference_dir)
    
    if ref_manager.reference_count == 0:
        logger.error("No reference genomes found!")
        return {}
    
    logger.info(
        f"Loaded {ref_manager.reference_count} reference genomes: "
        f"{', '.join(ref_manager.genome_ids)}"
    )
    
    # 2. Create aligner
    logger.info(f"Creating aligner with strategy: {strategy.value}")
    aligner = create_default_aligner(
        reference_manager=ref_manager,
        strategy=strategy,
        num_references=num_references,
    )
    
    # 3. Parse sample VCF
    variants_by_chr = parse_vcf_file(vcf_path)
    
    if not variants_by_chr:
        logger.error("No variants found in VCF file!")
        return {}
    
    # 4. Align each chromosome
    logger.info(f"Aligning {len(variants_by_chr)} chromosomes...")
    
    results = {}
    alignment_scores = {}
    total_start = time.time()
    
    for chromosome, variants in variants_by_chr.items():
        logger.info(f"\nProcessing {chromosome} ({len(variants)} variants)...")
        
        # Create genome section
        section = GenomeSection(
            chromosome=chromosome,
            start_position=min(v.position for v in variants),
            end_position=max(v.position for v in variants),
            variants=variants,
        )
        
        # Align
        start = time.time()
        result = aligner.align(section)
        elapsed = time.time() - start
        
        # Log results
        logger.info(
            f"  Primary match: {result.primary_reference} "
            f"(score: {result.alignment_scores[result.primary_reference].overall_score:.3f}, "
            f"confidence: {result.confidence:.2%})"
        )
        
        if result.secondary_references:
            logger.info(f"  Secondary matches: {', '.join(result.secondary_references)}")
        
        logger.info(f"  Alignment time: {elapsed:.2f}s")
        
        if result.ambiguous:
            logger.warning(f"  ⚠️  AMBIGUOUS alignment for {chromosome}")
        
        # Store results
        results[chromosome] = {
            "primary_reference": result.primary_reference,
            "secondary_references": result.secondary_references,
            "consensus_score": result.consensus_score,
            "confidence": result.confidence,
            "ambiguous": result.ambiguous,
            "alignment_time": elapsed,
        }
        
        # Store detailed scores
        alignment_scores[chromosome] = {
            ref_id: {
                "overall_score": score.overall_score,
                "variant_match_rate": score.variant_match_rate,
                "kmer_match_rate": score.kmer_match_rate,
                "snp_matches": score.snp_matches,
                "indel_matches": score.indel_matches,
                "new_variants": score.new_variants,
            }
            for ref_id, score in result.alignment_scores.items()
        }
    
    total_elapsed = time.time() - total_start
    
    # 5. Determine overall best reference by majority vote
    logger.info("\n" + "="*60)
    logger.info("Computing overall consensus...")
    
    # Convert results to ConsensusResult objects for voting
    consensus_results = {}
    for chromosome, result_dict in results.items():
        consensus_results[chromosome] = ConsensusResult(
            primary_reference=result_dict["primary_reference"],
            secondary_references=result_dict["secondary_references"],
            consensus_score=result_dict["consensus_score"],
            confidence=result_dict["confidence"],
            ambiguous=result_dict["ambiguous"],
        )
    
    overall_reference = aligner.majority_vote(consensus_results)
    
    logger.info(f"Overall best reference: {overall_reference}")
    logger.info(f"Total alignment time: {total_elapsed:.2f}s")
    
    # 6. Compile summary
    summary = {
        "sample": str(vcf_path),
        "overall_reference": overall_reference,
        "total_variants": sum(len(v) for v in variants_by_chr.values()),
        "total_chromosomes": len(variants_by_chr),
        "total_time": total_elapsed,
        "strategy": strategy.value,
        "num_references_used": num_references,
        "per_chromosome_results": results,
        "detailed_scores": alignment_scores,
    }
    
    # 7. Output summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Sample: {vcf_path.name}")
    logger.info(f"Overall Reference: {overall_reference}")
    logger.info(f"Total Variants: {summary['total_variants']:,}")
    logger.info(f"Chromosomes Processed: {summary['total_chromosomes']}")
    logger.info(f"Total Time: {total_elapsed:.2f}s")
    
    # Check for ambiguous chromosomes
    ambiguous_chrs = [
        chr for chr, res in results.items()
        if res["ambiguous"]
    ]
    
    if ambiguous_chrs:
        logger.warning(f"\n⚠️  Ambiguous alignments for: {', '.join(ambiguous_chrs)}")
        logger.warning("Consider manual review or using multiple references")
    
    # Per-chromosome breakdown
    logger.info("\nPer-Chromosome Results:")
    for chromosome, result in results.items():
        status = "⚠️ " if result["ambiguous"] else "✓ "
        logger.info(
            f"  {status}{chromosome}: {result['primary_reference']} "
            f"(confidence: {result['confidence']:.2%})"
        )
    
    # 8. Save JSON if requested
    if output_json:
        output_path = vcf_path.parent / f"{vcf_path.stem}_alignment_results.json"
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")
    
    return summary


def validate_with_known_reference(
    vcf_path: Path,
    reference_dir: Path,
    known_reference: str,
) -> bool:
    """
    Validate alignment system with a sample of known origin.
    
    Args:
        vcf_path: Path to sample VCF
        reference_dir: Directory with references
        known_reference: Expected reference ID
        
    Returns:
        True if alignment is correct
    """
    logger.info("\n" + "="*60)
    logger.info("VALIDATION MODE")
    logger.info("="*60)
    logger.info(f"Expected reference: {known_reference}")
    
    # Run alignment
    results = align_sample(
        vcf_path=vcf_path,
        reference_dir=reference_dir,
        strategy=AlignmentStrategy.HYBRID,
        num_references=3,
        output_json=False,
    )
    
    if not results:
        logger.error("Alignment failed!")
        return False
    
    # Check result
    predicted = results["overall_reference"]
    correct = predicted == known_reference
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULT")
    logger.info("="*60)
    logger.info(f"Expected: {known_reference}")
    logger.info(f"Predicted: {predicted}")
    logger.info(f"Result: {'✓ PASS' if correct else '✗ FAIL'}")
    
    if not correct:
        logger.error("Alignment prediction does not match known reference!")
    
    return correct


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GenomeVault Sequence Alignment Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--vcf",
        type=Path,
        required=True,
        help="Path to sample VCF file (.vcf or .vcf.gz)",
    )
    
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="Directory containing reference VCF files",
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid",
        choices=["kmer_only", "variant_scoring", "hybrid", "consensus"],
        help="Alignment strategy (default: hybrid)",
    )
    
    parser.add_argument(
        "--num-references",
        type=int,
        default=3,
        help="Number of references for consensus (default: 3)",
    )
    
    parser.add_argument(
        "--validate",
        type=str,
        help="Known reference ID for validation mode",
    )
    
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Don't output JSON results file",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.vcf.exists():
        logger.error(f"VCF file not found: {args.vcf}")
        sys.exit(1)
    
    if not args.references.exists():
        logger.error(f"Reference directory not found: {args.references}")
        sys.exit(1)
    
    # Map strategy string to enum
    strategy_map = {
        "kmer_only": AlignmentStrategy.KMER_ONLY,
        "variant_scoring": AlignmentStrategy.VARIANT_SCORING,
        "hybrid": AlignmentStrategy.HYBRID,
        "consensus": AlignmentStrategy.CONSENSUS,
    }
    strategy = strategy_map[args.strategy]
    
    # Run validation or standard alignment
    if args.validate:
        success = validate_with_known_reference(
            vcf_path=args.vcf,
            reference_dir=args.references,
            known_reference=args.validate,
        )
        sys.exit(0 if success else 1)
    else:
        results = align_sample(
            vcf_path=args.vcf,
            reference_dir=args.references,
            strategy=strategy,
            num_references=args.num_references,
            output_json=not args.no_json,
        )
        
        if not results:
            sys.exit(1)


if __name__ == "__main__":
    main()
