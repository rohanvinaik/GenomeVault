"""
Sequence Alignment System for GenomeVault

This module implements a lightweight, low-compute alignment system for identifying
which reference genome(s) best match an input sequence. Designed specifically for
variant-based genomic data (VCF format) rather than raw sequencing reads.

Architecture:
1. K-mer based pre-screening for fast candidate selection
2. Variant-level alignment scoring
3. Multi-reference consensus mechanism with fuzzy matching
4. Integration with existing differential encoding pipeline

Key differences from traditional read alignment (BWA, Minimap2):
- Works with variant calls (VCF), not raw reads (FASTQ)
- Optimized for whole genome or large segment alignment
- Tolerates population-level variation (SNPs, indels)
- Low computational requirements
- Multi-reference consensus for ambiguous cases

References:
- Minimap2: Li, H. (2018). Bioinformatics, 34(18), 3094-3100
- K-mer methods: LAVA, KmerKeys, SKA2
- Consensus approaches: Multi-mapper strategies from RNA-seq
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from genomevault.differential_encoding.reference_management import (
    ReferenceGenome,
    SecureReferenceGenomeManager,
    Variant,
    GenomeSection,
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class AlignmentStrategy(Enum):
    """Alignment strategy selection."""
    KMER_ONLY = "kmer_only"  # Fast k-mer based identification
    VARIANT_SCORING = "variant_scoring"  # Detailed variant-level alignment
    HYBRID = "hybrid"  # K-mer pre-screening + variant scoring (recommended)
    CONSENSUS = "consensus"  # Multi-reference consensus with voting


@dataclass
class AlignmentScore:
    """
    Alignment score for a reference genome.
    
    Attributes:
        reference_id: Reference genome identifier
        kmer_match_rate: Proportion of k-mers matching (0.0-1.0)
        variant_match_rate: Proportion of variants matching (0.0-1.0)
        snp_matches: Number of exact SNP matches
        snp_mismatches: Number of SNP mismatches
        indel_matches: Number of indel matches
        indel_mismatches: Number of indel mismatches
        new_variants: Number of variants in query not in reference
        missing_variants: Number of variants in reference not in query
        overall_score: Combined alignment score (0.0-1.0)
        confidence: Confidence in alignment (0.0-1.0)
    """
    reference_id: str
    kmer_match_rate: float = 0.0
    variant_match_rate: float = 0.0
    snp_matches: int = 0
    snp_mismatches: int = 0
    indel_matches: int = 0
    indel_mismatches: int = 0
    new_variants: int = 0
    missing_variants: int = 0
    overall_score: float = 0.0
    confidence: float = 0.0
    
    def __lt__(self, other: AlignmentScore) -> bool:
        """Enable sorting by overall score."""
        return self.overall_score < other.overall_score


@dataclass
class ConsensusResult:
    """
    Multi-reference consensus alignment result.
    
    Attributes:
        primary_reference: Best matching reference ID
        secondary_references: Additional good matches (for validation)
        consensus_score: Agreement level across references (0.0-1.0)
        alignment_scores: Scores for all evaluated references
        voting_results: Detailed voting breakdown by genomic region
        confidence: Overall confidence in primary assignment (0.0-1.0)
        ambiguous: Whether assignment is ambiguous (requires review)
    """
    primary_reference: str
    secondary_references: List[str] = field(default_factory=list)
    consensus_score: float = 0.0
    alignment_scores: Dict[str, AlignmentScore] = field(default_factory=dict)
    voting_results: Dict[str, Dict[str, int]] = field(default_factory=dict)
    confidence: float = 0.0
    ambiguous: bool = False


class KmerIndex:
    """
    Fast k-mer index for reference genome pre-screening.
    
    Uses k-mer decomposition for rapid similarity estimation without
    full alignment. Optimized for variant-based data.
    
    Implementation inspired by:
    - KmerKeys (hash table based)
    - LAVA (lightweight variant assignment)
    - Minimap2 (minimizer approach)
    """
    
    def __init__(self, k: int = 31):
        """
        Initialize k-mer index.
        
        Args:
            k: K-mer length (default 31, optimal for human genome uniqueness)
        """
        self.k = k
        # Maps k-mer hash -> set of (reference_id, chromosome, position)
        self.kmer_map: Dict[int, Set[Tuple[str, str, int]]] = defaultdict(set)
        self.reference_ids: Set[str] = set()
        
        logger.debug(f"Initialized KmerIndex with k={k}")
    
    def _hash_kmer(self, kmer: str) -> int:
        """
        Hash k-mer to integer for fast lookup.
        
        Uses first 8 bytes of SHA-256 for good distribution.
        
        Args:
            kmer: K-mer sequence
            
        Returns:
            Integer hash
        """
        return int.from_bytes(
            hashlib.sha256(kmer.encode()).digest()[:8],
            byteorder='big'
        )
    
    def _extract_kmers(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str
    ) -> Set[str]:
        """
        Extract k-mers from variant context.
        
        For a variant, we extract k-mers from:
        1. The reference allele context
        2. The alternate allele context
        
        Args:
            chromosome: Chromosome identifier
            position: Variant position
            ref: Reference allele
            alt: Alternate allele
            
        Returns:
            Set of k-mers
        """
        kmers = set()
        
        # For simplicity, use the allele sequences directly
        # In production, could include flanking context
        if len(ref) >= self.k:
            for i in range(len(ref) - self.k + 1):
                kmers.add(ref[i:i+self.k])
        
        if len(alt) >= self.k:
            for i in range(len(alt) - self.k + 1):
                kmers.add(alt[i:i+self.k])
        
        return kmers
    
    def index_reference(self, reference: ReferenceGenome) -> None:
        """
        Index a reference genome's variants.
        
        Args:
            reference: Reference genome to index
        """
        logger.info(f"Indexing reference genome: {reference.genome_id}")
        
        self.reference_ids.add(reference.genome_id)
        variant_count = 0
        
        for chromosome, variants in reference.variants.items():
            for variant in variants:
                # Extract k-mers from variant
                kmers = self._extract_kmers(
                    chromosome,
                    variant.position,
                    variant.ref,
                    variant.alt
                )
                
                # Add to index
                for kmer in kmers:
                    kmer_hash = self._hash_kmer(kmer)
                    self.kmer_map[kmer_hash].add(
                        (reference.genome_id, chromosome, variant.position)
                    )
                
                variant_count += 1
        
        logger.info(
            f"Indexed {variant_count} variants from {reference.genome_id}, "
            f"total k-mers: {len(self.kmer_map)}"
        )
    
    def query_variants(
        self,
        variants: List[Variant],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Query variants against k-mer index.
        
        Returns match rates for each reference genome.
        
        Args:
            variants: List of query variants
            top_k: Number of top matches to return
            
        Returns:
            Dictionary mapping reference_id to match rate (0.0-1.0)
        """
        if not variants:
            return {}
        
        # Count k-mer matches per reference
        reference_matches: Counter = Counter()
        total_kmers = 0
        
        for variant in variants:
            kmers = self._extract_kmers(
                variant.chromosome,
                variant.position,
                variant.ref,
                variant.alt
            )
            
            for kmer in kmers:
                kmer_hash = self._hash_kmer(kmer)
                total_kmers += 1
                
                if kmer_hash in self.kmer_map:
                    # Increment count for each reference with this k-mer
                    for ref_id, _, _ in self.kmer_map[kmer_hash]:
                        reference_matches[ref_id] += 1
        
        # Compute match rates
        if total_kmers == 0:
            return {}
        
        match_rates = {
            ref_id: count / total_kmers
            for ref_id, count in reference_matches.most_common(top_k)
        }
        
        logger.debug(
            f"K-mer query: {len(variants)} variants, {total_kmers} k-mers, "
            f"top match: {list(match_rates.keys())[0] if match_rates else 'none'}"
        )
        
        return match_rates


class VariantAligner:
    """
    Variant-level alignment scorer.
    
    Computes detailed alignment scores based on variant-level comparison,
    accounting for SNPs, indels, and genotype differences with fuzzy matching.
    """
    
    def __init__(
        self,
        snp_weight: float = 1.0,
        indel_weight: float = 0.8,
        genotype_weight: float = 0.3,
        position_tolerance: int = 10,
    ):
        """
        Initialize variant aligner.
        
        Args:
            snp_weight: Weight for SNP matches (default 1.0)
            indel_weight: Weight for indel matches (default 0.8, slightly lower)
            genotype_weight: Weight for genotype differences (default 0.3)
            position_tolerance: Position tolerance for fuzzy matching (bp)
        """
        self.snp_weight = snp_weight
        self.indel_weight = indel_weight
        self.genotype_weight = genotype_weight
        self.position_tolerance = position_tolerance
        
        logger.debug(
            f"Initialized VariantAligner: "
            f"snp_weight={snp_weight}, indel_weight={indel_weight}, "
            f"genotype_weight={genotype_weight}, tolerance={position_tolerance}bp"
        )
    
    def _is_snp(self, variant: Variant) -> bool:
        """Check if variant is a SNP (single nucleotide)."""
        return len(variant.ref) == 1 and len(variant.alt) == 1
    
    def _is_indel(self, variant: Variant) -> bool:
        """Check if variant is an indel."""
        return len(variant.ref) != len(variant.alt)
    
    def _fuzzy_match_position(
        self,
        pos1: int,
        pos2: int,
        tolerance: int
    ) -> bool:
        """
        Check if two positions match within tolerance.
        
        Args:
            pos1: First position
            pos2: Second position
            tolerance: Maximum allowed difference (bp)
            
        Returns:
            True if positions match within tolerance
        """
        return abs(pos1 - pos2) <= tolerance
    
    def _match_variants(
        self,
        query_variant: Variant,
        reference_variants: List[Variant],
        tolerance: int
    ) -> Optional[Variant]:
        """
        Find matching variant in reference.
        
        Looks for variants at same/nearby position with same alleles.
        Allows fuzzy matching for indels (position tolerance).
        
        Args:
            query_variant: Query variant
            reference_variants: List of reference variants to search
            tolerance: Position tolerance for matching
            
        Returns:
            Matching reference variant or None
        """
        for ref_var in reference_variants:
            # Check if positions match (with tolerance)
            if not self._fuzzy_match_position(
                query_variant.position,
                ref_var.position,
                tolerance
            ):
                continue
            
            # Check if alleles match
            if (query_variant.ref == ref_var.ref and
                query_variant.alt == ref_var.alt):
                return ref_var
            
            # For indels, allow minor position shifts
            if self._is_indel(query_variant) and self._is_indel(ref_var):
                # Check if it's likely the same indel (same length change)
                query_len_change = len(query_variant.alt) - len(query_variant.ref)
                ref_len_change = len(ref_var.alt) - len(ref_var.ref)
                
                if query_len_change == ref_len_change:
                    return ref_var
        
        return None
    
    def align_section(
        self,
        query_section: GenomeSection,
        reference_section: GenomeSection,
        reference_id: str
    ) -> AlignmentScore:
        """
        Compute alignment score for a genomic section.
        
        Args:
            query_section: Query genome section
            reference_section: Reference genome section
            reference_id: Reference genome identifier
            
        Returns:
            AlignmentScore with detailed metrics
        """
        score = AlignmentScore(reference_id=reference_id)
        
        # Separate variants by type
        query_snps = [v for v in query_section.variants if self._is_snp(v)]
        query_indels = [v for v in query_section.variants if self._is_indel(v)]
        ref_variants = reference_section.variants
        
        # Score SNPs
        for query_var in query_snps:
            match = self._match_variants(
                query_var,
                ref_variants,
                tolerance=0  # Exact position for SNPs
            )
            
            if match:
                score.snp_matches += 1
                # Check genotype difference
                if query_var.genotype != match.genotype:
                    score.snp_mismatches += 1  # Count as partial mismatch
            else:
                score.new_variants += 1
        
        # Score indels (with position tolerance)
        for query_var in query_indels:
            match = self._match_variants(
                query_var,
                ref_variants,
                tolerance=self.position_tolerance
            )
            
            if match:
                score.indel_matches += 1
                if query_var.genotype != match.genotype:
                    score.indel_mismatches += 1
            else:
                score.new_variants += 1
        
        # Count missing variants (in reference but not in query)
        query_positions = {v.position for v in query_section.variants}
        for ref_var in ref_variants:
            if ref_var.position not in query_positions:
                score.missing_variants += 1
        
        # Compute rates
        total_query_variants = len(query_section.variants)
        if total_query_variants > 0:
            matches = score.snp_matches + score.indel_matches
            score.variant_match_rate = matches / total_query_variants
        
        # Compute overall score (weighted)
        total_snps = score.snp_matches + score.new_variants
        total_indels = score.indel_matches + score.new_variants
        
        snp_score = 0.0
        if total_snps > 0:
            snp_score = (score.snp_matches / total_snps) * self.snp_weight
        
        indel_score = 0.0
        if total_indels > 0:
            indel_score = (score.indel_matches / total_indels) * self.indel_weight
        
        # Penalty for genotype mismatches
        genotype_penalty = (
            (score.snp_mismatches + score.indel_mismatches) *
            self.genotype_weight /
            max(total_query_variants, 1)
        )
        
        # Combined score
        if total_snps > 0 or total_indels > 0:
            score.overall_score = (
                (snp_score + indel_score) / 2.0 - genotype_penalty
            )
            score.overall_score = max(0.0, min(1.0, score.overall_score))
        
        # Compute confidence based on number of variants
        # More variants = higher confidence in score
        score.confidence = min(
            1.0,
            total_query_variants / 100.0  # Saturates at 100 variants
        )
        
        return score


class MultiReferenceAligner:
    """
    Multi-reference consensus alignment system.
    
    Aligns query sequences against multiple reference genomes and uses
    consensus voting to identify the best match(es). Handles ambiguous
    cases where sequence is intermediate between references.
    
    Consensus Strategies:
    1. Simple majority vote (per chromosome/region)
    2. Weighted voting (by alignment score)
    3. Hierarchical consensus (chromosome -> region -> variant level)
    """
    
    def __init__(
        self,
        reference_manager: SecureReferenceGenomeManager,
        kmer_index: Optional[KmerIndex] = None,
        variant_aligner: Optional[VariantAligner] = None,
        strategy: AlignmentStrategy = AlignmentStrategy.HYBRID,
        num_references: int = 3,
        consensus_threshold: float = 0.6,
    ):
        """
        Initialize multi-reference aligner.
        
        Args:
            reference_manager: Reference genome manager
            kmer_index: Optional pre-built k-mer index
            variant_aligner: Optional variant aligner
            strategy: Alignment strategy
            num_references: Number of references for consensus (default 3)
            consensus_threshold: Minimum agreement for consensus (0.6 = 60%)
        """
        self.reference_manager = reference_manager
        self.kmer_index = kmer_index or KmerIndex()
        self.variant_aligner = variant_aligner or VariantAligner()
        self.strategy = strategy
        self.num_references = num_references
        self.consensus_threshold = consensus_threshold
        
        # Build k-mer index if not provided
        if kmer_index is None:
            self._build_kmer_index()
        
        logger.info(
            f"Initialized MultiReferenceAligner: "
            f"strategy={strategy.value}, "
            f"num_references={num_references}, "
            f"threshold={consensus_threshold}"
        )
    
    def _build_kmer_index(self) -> None:
        """Build k-mer index for all references."""
        logger.info("Building k-mer index for all references...")
        
        for genome_id in self.reference_manager.genome_ids:
            reference = self.reference_manager.pool.get_reference(genome_id)
            self.kmer_index.index_reference(reference)
        
        logger.info("K-mer index built successfully")
    
    def _select_candidate_references(
        self,
        query_variants: List[Variant],
        top_k: int
    ) -> List[str]:
        """
        Select candidate references using k-mer pre-screening.
        
        Args:
            query_variants: Query variants
            top_k: Number of candidates to return
            
        Returns:
            List of reference IDs (best matches first)
        """
        match_rates = self.kmer_index.query_variants(query_variants, top_k)
        return list(match_rates.keys())
    
    def align(
        self,
        query_section: GenomeSection,
        chromosome: Optional[str] = None,
        fast_mode: bool = False,
    ) -> ConsensusResult:
        """
        Align query section against multiple references with consensus.
        
        Args:
            query_section: Query genome section to align
            chromosome: Optional chromosome constraint
            fast_mode: If True, use k-mer only (faster but less accurate)
            
        Returns:
            ConsensusResult with primary reference and consensus metrics
        """
        logger.info(
            f"Aligning query section: {query_section.chromosome}:"
            f"{query_section.start_position}-{query_section.end_position}"
        )
        
        # Step 1: Select candidate references
        if self.strategy in [AlignmentStrategy.HYBRID, AlignmentStrategy.CONSENSUS]:
            candidates = self._select_candidate_references(
                query_section.variants,
                top_k=self.num_references * 2  # Select more for filtering
            )
        else:
            candidates = self.reference_manager.genome_ids
        
        if not candidates:
            logger.warning("No candidate references found")
            candidates = self.reference_manager.genome_ids[:self.num_references]
        
        # Limit to num_references
        candidates = candidates[:self.num_references]
        
        logger.debug(f"Selected {len(candidates)} candidate references")
        
        # Step 2: Compute detailed alignment scores
        alignment_scores: Dict[str, AlignmentScore] = {}
        
        for ref_id in candidates:
            reference = self.reference_manager.pool.get_reference(ref_id)
            
            # Get matching reference section
            ref_section = reference.get_section(
                query_section.chromosome,
                query_section.start_position,
                query_section.end_position
            )
            
            # Compute alignment score
            if fast_mode or self.strategy == AlignmentStrategy.KMER_ONLY:
                # Use k-mer match rate only
                score = AlignmentScore(reference_id=ref_id)
                match_rate = self.kmer_index.query_variants(
                    query_section.variants,
                    top_k=1
                ).get(ref_id, 0.0)
                score.kmer_match_rate = match_rate
                score.overall_score = match_rate
                score.confidence = 0.5  # Lower confidence for k-mer only
            else:
                # Full variant-level alignment
                score = self.variant_aligner.align_section(
                    query_section,
                    ref_section,
                    ref_id
                )
                # Incorporate k-mer score
                kmer_rate = self.kmer_index.query_variants(
                    query_section.variants,
                    top_k=1
                ).get(ref_id, 0.0)
                score.kmer_match_rate = kmer_rate
                # Combine scores (70% variant, 30% k-mer)
                score.overall_score = (
                    0.7 * score.overall_score + 0.3 * kmer_rate
                )
            
            alignment_scores[ref_id] = score
        
        # Step 3: Consensus voting
        # Sort by overall score
        sorted_scores = sorted(
            alignment_scores.values(),
            key=lambda s: s.overall_score,
            reverse=True
        )
        
        if not sorted_scores:
            logger.warning("No alignment scores computed")
            return ConsensusResult(
                primary_reference="unknown",
                confidence=0.0,
                ambiguous=True
            )
        
        # Primary reference is best match
        primary = sorted_scores[0]
        
        # Secondary references are other good matches
        secondary = [
            s.reference_id for s in sorted_scores[1:]
            if s.overall_score >= self.consensus_threshold
        ]
        
        # Compute consensus score (agreement among top references)
        if len(sorted_scores) >= 2:
            # Measure gap between best and second-best
            score_gap = primary.overall_score - sorted_scores[1].overall_score
            # Higher gap = stronger consensus
            consensus_score = min(1.0, score_gap * 2.0 + 0.5)
        else:
            consensus_score = 1.0  # Only one reference = perfect "consensus"
        
        # Check for ambiguity
        ambiguous = (
            consensus_score < self.consensus_threshold or
            primary.confidence < 0.5 or
            len(secondary) >= 2  # Multiple similarly good matches
        )
        
        logger.info(
            f"Alignment complete: primary={primary.reference_id}, "
            f"score={primary.overall_score:.3f}, "
            f"consensus={consensus_score:.3f}, "
            f"ambiguous={ambiguous}"
        )
        
        return ConsensusResult(
            primary_reference=primary.reference_id,
            secondary_references=secondary,
            consensus_score=consensus_score,
            alignment_scores=alignment_scores,
            confidence=primary.confidence,
            ambiguous=ambiguous
        )
    
    def align_genome(
        self,
        query_genome_sections: List[GenomeSection],
        chunk_size: int = 1000000,  # 1 Mb chunks
    ) -> Dict[str, ConsensusResult]:
        """
        Align entire genome in chunks with per-chunk consensus.
        
        Args:
            query_genome_sections: List of genome sections to align
            chunk_size: Size of chunks for alignment (bp)
            
        Returns:
            Dictionary mapping chunk identifier to ConsensusResult
        """
        results = {}
        
        for section in query_genome_sections:
            # Generate chunk identifier
            chunk_id = (
                f"{section.chromosome}:"
                f"{section.start_position}-{section.end_position}"
            )
            
            # Align chunk
            result = self.align(section)
            results[chunk_id] = result
            
            logger.debug(
                f"Aligned chunk {chunk_id}: "
                f"primary={result.primary_reference}, "
                f"score={result.alignment_scores[result.primary_reference].overall_score:.3f}"
            )
        
        return results
    
    def majority_vote(
        self,
        chunk_results: Dict[str, ConsensusResult]
    ) -> str:
        """
        Determine overall genome reference by majority vote across chunks.
        
        Args:
            chunk_results: Dictionary of per-chunk consensus results
            
        Returns:
            Reference ID with most votes
        """
        votes: Counter = Counter()
        
        for chunk_id, result in chunk_results.items():
            # Weight vote by confidence
            votes[result.primary_reference] += result.confidence
        
        if not votes:
            logger.warning("No votes to count")
            return "unknown"
        
        winner = votes.most_common(1)[0][0]
        total_votes = sum(votes.values())
        winner_votes = votes[winner]
        
        logger.info(
            f"Majority vote: {winner} "
            f"({winner_votes/total_votes*100:.1f}% of votes)"
        )
        
        return winner


def create_default_aligner(
    reference_manager: SecureReferenceGenomeManager,
    strategy: AlignmentStrategy = AlignmentStrategy.HYBRID,
    **kwargs
) -> MultiReferenceAligner:
    """
    Create a MultiReferenceAligner with sensible defaults.
    
    Args:
        reference_manager: Reference genome manager
        strategy: Alignment strategy
        **kwargs: Additional arguments for MultiReferenceAligner
        
    Returns:
        Configured MultiReferenceAligner
    """
    return MultiReferenceAligner(
        reference_manager=reference_manager,
        strategy=strategy,
        **kwargs
    )
