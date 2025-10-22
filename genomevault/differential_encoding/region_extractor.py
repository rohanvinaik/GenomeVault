"""
Multi-Reference Region Extraction for Differential Encoding

Extracts the same genomic region from all references in the pool for k-anonymity.

This ensures that differential encoding uses the SAME region from ALL references,
maintaining privacy by not revealing which reference was ultimately used.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import pysam

from genomevault.differential_encoding.reference_management import (
    SecureReferenceGenomeManager,
    GenomeSection,
)
from genomevault.differential_encoding.fastq_processor import GenomicRegion

logger = logging.getLogger(__name__)


@dataclass
class MultiReferenceRegion:
    """
    Same genomic region extracted from multiple references.

    Attributes:
        chromosome: Chromosome identifier
        start: Start position
        end: End position
        reference_sections: Dict mapping reference_id → GenomeSection
        reference_sequences: Dict mapping reference_id → sequence string
    """
    chromosome: str
    start: int
    end: int
    reference_sections: Dict[str, GenomeSection]
    reference_sequences: Dict[str, str]

    @property
    def num_references(self) -> int:
        """Number of references extracted."""
        return len(self.reference_sections)

    def get_reference_ids(self) -> List[str]:
        """Get list of reference IDs."""
        return list(self.reference_sections.keys())


class MultiReferenceExtractor:
    """
    Extracts the same genomic region from all references in the pool.

    This is critical for privacy-preserving differential encoding:
    - Same region extracted from ALL references
    - Differential encoding randomly selects one
    - Attacker cannot determine which was used (k-anonymity)
    """

    def __init__(self, reference_manager: SecureReferenceGenomeManager):
        """
        Initialize extractor with reference pool.

        Args:
            reference_manager: Manager with loaded reference genomes
        """
        self.reference_manager = reference_manager
        logger.info(
            f"Initialized MultiReferenceExtractor with "
            f"{len(reference_manager.genome_ids)} references"
        )

    def extract_region(
        self,
        region: GenomicRegion,
        reference_ids: List[str] = None,
    ) -> MultiReferenceRegion:
        """
        Extract the same region from all (or specified) references.

        Args:
            region: Genomic region to extract
            reference_ids: Optional list of specific references to extract from
                          (defaults to all references in pool)

        Returns:
            MultiReferenceRegion with extracted sections from all references
        """
        if reference_ids is None:
            reference_ids = self.reference_manager.genome_ids

        logger.info(
            f"Extracting region {region.chromosome}:{region.start}-{region.end} "
            f"from {len(reference_ids)} references"
        )

        reference_sections = {}
        reference_sequences = {}

        for ref_id in reference_ids:
            try:
                # Get reference genome
                reference = self.reference_manager.pool.get_reference(ref_id)

                # Extract section
                section = reference.get_section(
                    chromosome=region.chromosome,
                    start=region.start,
                    end=region.end,
                )

                # Get sequence (if available)
                sequence = self._extract_sequence(
                    ref_id=ref_id,
                    chromosome=region.chromosome,
                    start=region.start,
                    end=region.end,
                )

                reference_sections[ref_id] = section
                reference_sequences[ref_id] = sequence

                logger.debug(
                    f"Extracted from {ref_id}: {len(section.variants)} variants, "
                    f"{len(sequence)} bp sequence"
                )

            except Exception as e:
                logger.warning(f"Failed to extract from {ref_id}: {e}")
                continue

        if not reference_sections:
            raise ValueError(f"Failed to extract region from any reference")

        result = MultiReferenceRegion(
            chromosome=region.chromosome,
            start=region.start,
            end=region.end,
            reference_sections=reference_sections,
            reference_sequences=reference_sequences,
        )

        logger.info(
            f"Successfully extracted from {result.num_references}/{len(reference_ids)} references"
        )

        return result

    def _extract_sequence(
        self,
        ref_id: str,
        chromosome: str,
        start: int,
        end: int,
    ) -> str:
        """
        Extract reference sequence from FASTA file.

        Args:
            ref_id: Reference genome ID
            chromosome: Chromosome name
            start: Start position (0-based)
            end: End position (exclusive)

        Returns:
            Extracted sequence string
        """
        try:
            # Get FASTA file path for this reference
            fasta_path = self._get_reference_fasta(ref_id)

            if not fasta_path or not fasta_path.exists():
                logger.warning(f"FASTA not found for {ref_id}")
                return ""

            # Use pysam to extract sequence
            with pysam.FastaFile(str(fasta_path)) as fasta:
                # Normalize chromosome name (chr22 vs 22)
                chrom_name = chromosome
                if chrom_name not in fasta.references:
                    # Try without 'chr' prefix
                    chrom_name = chromosome.replace('chr', '')
                    if chrom_name not in fasta.references:
                        logger.warning(
                            f"Chromosome {chromosome} not found in {ref_id}"
                        )
                        return ""

                sequence = fasta.fetch(chrom_name, start, end)
                return sequence

        except Exception as e:
            logger.warning(f"Failed to extract sequence from {ref_id}: {e}")
            return ""

    def _get_reference_fasta(self, ref_id: str) -> Path:
        """
        Get FASTA file path for a reference genome.

        Looks in standard locations based on reference ID.
        """
        # Try common locations
        potential_paths = [
            Path(f"benchmark_results/differential_encoding_samples/references/{ref_id}/reference.fa"),
            Path(f"benchmark_results/differential_encoding_samples/references/{ref_id}/genome.fa"),
            Path(f"data/references/{ref_id}.fa"),
            Path(f"data/references/{ref_id}/genome.fa"),
        ]

        for path in potential_paths:
            if path.exists():
                return path

        logger.debug(f"No FASTA found for {ref_id} in standard locations")
        return None

    def extract_multiple_regions(
        self,
        regions: List[GenomicRegion],
    ) -> List[MultiReferenceRegion]:
        """
        Extract multiple regions from all references.

        Args:
            regions: List of genomic regions to extract

        Returns:
            List of MultiReferenceRegion objects
        """
        results = []

        for i, region in enumerate(regions, 1):
            logger.info(f"Extracting region {i}/{len(regions)}")
            try:
                result = self.extract_region(region)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract region {i}: {e}")
                continue

        logger.info(f"Successfully extracted {len(results)}/{len(regions)} regions")
        return results
