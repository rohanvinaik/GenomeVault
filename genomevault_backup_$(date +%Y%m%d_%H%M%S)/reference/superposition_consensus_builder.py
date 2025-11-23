"""
Superposition Consensus Builder - Graph-Based Reference Genome

Extends Byzantine consensus with superposition support, representing multiple valid
alignment paths for variable genomic regions instead of forcing a single consensus.

Key Features:
1. Conserved regions (95-99% agreement) → single path (fast alignment)
2. Variable regions (structural variants, common indels) → multiple paths (population-aware)
3. Graph genome structure for efficient alignment
4. Population variant integration (gnomAD, 1000 Genomes)
5. Export to variation graph formats (VG, GFA)

Performance Target:
- 95-99% of genome uses single path (conserved)
- 1-5% uses multiple paths (variable)
- Total size: ~1.2GB for whole genome (1.2× single reference)

Usage:
    from genomevault.reference import build_superposition_consensus

    build_superposition_consensus(
        references=['hg38.fa.gz', 'hg19.fa.gz', 'chm13v2.0.fa.gz'],
        population_variants='gnomad.v3.1.2.vcf.gz',
        output='consensus_superposition/',
        conservation_threshold=0.95,
        threads=8
    )
"""

import gzip
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import hashlib
import numpy as np

from .byzantine_consensus_builder import (
    ByzantineConsensusBuilder,
    ConsensusBase
)

# Try to import BioPython, fall back to custom parser
try:
    from Bio import SeqIO
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

logger = logging.getLogger(__name__)


class PathSelectionStrategy(Enum):
    """Strategy for selecting paths in variable regions."""
    MOST_COMMON = "most_common"  # Select most common allele
    POPULATION_WEIGHTED = "population_weighted"  # Weight by population frequency
    ALL_PATHS = "all_paths"  # Include all valid paths


@dataclass
class PopulationVariant:
    """Represents a variant from population databases (gnomAD, 1000 Genomes)."""
    chromosome: str
    position: int
    ref_allele: str
    alt_alleles: List[str]
    allele_frequencies: List[float]
    variant_type: str  # SNV, INDEL, SV
    variant_id: str  # dbSNP ID or similar

    @property
    def is_common(self) -> bool:
        """Returns True if any allele frequency > 1%."""
        return any(af > 0.01 for af in self.allele_frequencies)

    @property
    def max_frequency(self) -> float:
        """Returns maximum allele frequency."""
        return max(self.allele_frequencies) if self.allele_frequencies else 0.0


@dataclass
class SuperpositionPath:
    """Represents one valid alignment path through a variable region."""
    allele_sequence: str
    population_frequency: float  # From gnomAD/1000 Genomes
    source_references: List[str]  # Which references support this path
    confidence: float
    path_id: str  # Unique identifier (hash of sequence)
    is_reference_path: bool = False  # True if this is the "reference" path

    def __post_init__(self):
        """Generate path ID if not provided."""
        if not self.path_id:
            self.path_id = hashlib.sha256(
                self.allele_sequence.encode()
            ).hexdigest()[:16]


@dataclass
class SuperpositionNode:
    """A genomic position that branches into multiple valid paths."""
    chromosome: str
    position: int
    end_position: int  # For regions, not single bases
    is_conserved: bool  # True if single path, False if multiple
    paths: List[SuperpositionPath] = field(default_factory=list)
    consensus_base: Optional[str] = None  # For conserved regions
    conservation_score: float = 1.0  # Agreement level across references

    @property
    def is_variable(self) -> bool:
        """Returns True if this node has multiple paths."""
        return not self.is_conserved and len(self.paths) > 1

    @property
    def region_length(self) -> int:
        """Returns length of this region."""
        return self.end_position - self.position

    def get_reference_path(self) -> Optional[SuperpositionPath]:
        """Returns the reference path (highest confidence)."""
        if not self.paths:
            return None
        # Sort by confidence, then by population frequency
        sorted_paths = sorted(
            self.paths,
            key=lambda p: (p.confidence, p.population_frequency),
            reverse=True
        )
        return sorted_paths[0]


@dataclass
class ConservedRegion:
    """Represents a conserved region with high agreement."""
    chromosome: str
    start: int
    end: int
    conservation_score: float
    sequence: str

    def __len__(self):
        return self.end - self.start


@dataclass
class VariableRegion:
    """Represents a variable region with multiple valid paths."""
    chromosome: str
    start: int
    end: int
    conservation_score: float
    paths: List[SuperpositionPath]
    population_variants: List[PopulationVariant]

    def __len__(self):
        return self.end - self.start


class SuperpositionConsensusBuilder(ByzantineConsensusBuilder):
    """
    Extends Byzantine consensus with superposition support.

    Features:
    - Identifies conserved regions (95-99% agreement) → single path
    - Identifies variable regions (structural variants, common indels) → multiple paths
    - Creates graph genome structure for efficient alignment
    - Indexes paths for best-match selection
    """

    def __init__(
        self,
        conservation_threshold: float = 0.95,
        population_variant_threshold: float = 0.01,  # 1% population frequency
        use_graph_structure: bool = True,
        path_selection: PathSelectionStrategy = PathSelectionStrategy.ALL_PATHS,
        window_size: int = 100,
        **kwargs
    ):
        """
        Initialize Superposition Consensus Builder.

        Args:
            conservation_threshold: Minimum agreement for conserved region (default: 0.95)
            population_variant_threshold: Minimum allele frequency to include (default: 0.01)
            use_graph_structure: Whether to build graph structure (default: True)
            path_selection: Strategy for selecting paths in variable regions
            window_size: Window size for conservation analysis (default: 100bp)
            **kwargs: Additional arguments for ByzantineConsensusBuilder
        """
        super().__init__(**kwargs)

        self.conservation_threshold = conservation_threshold
        self.pop_var_threshold = population_variant_threshold
        self.use_graph = use_graph_structure
        self.path_selection = path_selection
        self.window_size = window_size

        # Superposition-specific stats
        self.superposition_stats = {
            'conserved_regions': 0,
            'variable_regions': 0,
            'total_paths': 0,
            'conserved_bases': 0,
            'variable_bases': 0,
            'population_variants_loaded': 0,
            'population_variants_used': 0,
        }

        # Storage for superposition nodes
        self.superposition_nodes: Dict[str, List[SuperpositionNode]] = defaultdict(list)
        self.conserved_regions: Dict[str, List[ConservedRegion]] = defaultdict(list)
        self.variable_regions: Dict[str, List[VariableRegion]] = defaultdict(list)

        # Population variant database
        self.population_variants: Dict[str, List[PopulationVariant]] = defaultdict(list)

    def identify_conserved_regions(
        self,
        chrom: str,
        consensus_metadata: List[ConsensusBase],
        window_size: Optional[int] = None
    ) -> List[Tuple[int, int, bool]]:
        """
        Identify conserved vs variable regions using sliding window.

        Args:
            chrom: Chromosome name
            consensus_metadata: List of ConsensusBase objects for this chromosome
            window_size: Size of sliding window (default: self.window_size)

        Returns:
            List of (start, end, is_conserved) tuples
        """
        if window_size is None:
            window_size = self.window_size

        logger.info(f"Identifying conserved/variable regions for {chrom} (window={window_size}bp)...")

        regions = []
        current_start = 0
        current_is_conserved = None

        for i in range(0, len(consensus_metadata), window_size):
            window = consensus_metadata[i:i + window_size]

            # Calculate conservation score for this window
            high_conf_count = sum(1 for c in window if c.confidence >= self.conservation_threshold)
            conservation_score = high_conf_count / len(window) if window else 0.0

            is_conserved = conservation_score >= self.conservation_threshold

            # Start new region or continue current?
            if current_is_conserved is None:
                # First window
                current_start = i
                current_is_conserved = is_conserved
            elif current_is_conserved != is_conserved:
                # Region type changed - save previous region
                regions.append((current_start, i, current_is_conserved))
                current_start = i
                current_is_conserved = is_conserved

        # Save final region
        if current_is_conserved is not None:
            regions.append((current_start, len(consensus_metadata), current_is_conserved))

        # Update stats
        conserved_count = sum(1 for _, _, is_cons in regions if is_cons)
        variable_count = len(regions) - conserved_count

        conserved_bases = sum(end - start for start, end, is_cons in regions if is_cons)
        variable_bases = sum(end - start for start, end, is_cons in regions if not is_cons)

        self.superposition_stats['conserved_regions'] += conserved_count
        self.superposition_stats['variable_regions'] += variable_count
        self.superposition_stats['conserved_bases'] += conserved_bases
        self.superposition_stats['variable_bases'] += variable_bases

        logger.info(f"  Identified {conserved_count} conserved regions ({conserved_bases:,} bases)")
        logger.info(f"  Identified {variable_count} variable regions ({variable_bases:,} bases)")
        logger.info(f"  Conservation rate: {100*conserved_bases/(conserved_bases+variable_bases):.2f}%")

        return regions

    def load_population_variants(
        self,
        vcf_path: Path,
        min_frequency: Optional[float] = None,
        chromosomes: Optional[Set[str]] = None
    ) -> Dict[str, List[PopulationVariant]]:
        """
        Load common structural variants from population databases.

        Args:
            vcf_path: Path to gnomAD or 1000 Genomes VCF
            min_frequency: Minimum allele frequency to include (default: self.pop_var_threshold)
            chromosomes: Optional set of chromosomes to load (default: all)

        Returns:
            Dict mapping chromosome to list of variants
        """
        if min_frequency is None:
            min_frequency = self.pop_var_threshold

        logger.info(f"Loading population variants from {vcf_path}...")
        logger.info(f"  Minimum allele frequency: {min_frequency}")

        variants = defaultdict(list)
        total_variants = 0
        filtered_variants = 0

        # Open VCF file (handle .gz compression)
        if vcf_path.suffix == '.gz':
            handle = gzip.open(vcf_path, 'rt')
        else:
            handle = open(vcf_path, 'r')

        try:
            for line in handle:
                if line.startswith('#'):
                    continue  # Skip header

                fields = line.strip().split('\t')
                if len(fields) < 8:
                    continue

                chrom = fields[0]

                # Filter by chromosome if specified
                if chromosomes and chrom not in chromosomes:
                    continue

                pos = int(fields[1]) - 1  # Convert to 0-indexed
                ref = fields[3]
                alt = fields[4].split(',')

                # Parse INFO field for allele frequencies
                info = fields[7]
                info_dict = {}
                for item in info.split(';'):
                    if '=' in item:
                        key, value = item.split('=', 1)
                        info_dict[key] = value

                # Try to extract allele frequencies (format varies by database)
                allele_freqs = []
                if 'AF' in info_dict:
                    allele_freqs = [float(f) for f in info_dict['AF'].split(',')]
                elif 'AN' in info_dict and 'AC' in info_dict:
                    # Calculate frequency from allele counts
                    an = float(info_dict['AN'])
                    ac = [float(c) for c in info_dict['AC'].split(',')]
                    allele_freqs = [c / an if an > 0 else 0.0 for c in ac]

                # Filter by frequency
                if not allele_freqs or max(allele_freqs) < min_frequency:
                    filtered_variants += 1
                    continue

                # Determine variant type
                variant_type = 'SNV'
                for a in alt:
                    if len(a) != len(ref):
                        variant_type = 'INDEL'
                        break
                    if abs(len(a) - len(ref)) > 50:
                        variant_type = 'SV'
                        break

                # Get variant ID
                variant_id = fields[2] if len(fields) > 2 else f"{chrom}:{pos}"

                variant = PopulationVariant(
                    chromosome=chrom,
                    position=pos,
                    ref_allele=ref,
                    alt_alleles=alt,
                    allele_frequencies=allele_freqs,
                    variant_type=variant_type,
                    variant_id=variant_id
                )

                variants[chrom].append(variant)
                total_variants += 1

        finally:
            handle.close()

        self.superposition_stats['population_variants_loaded'] = total_variants

        logger.info(f"  Loaded {total_variants:,} common variants (filtered {filtered_variants:,})")
        logger.info(f"  Chromosomes: {len(variants)}")

        # Store in instance
        self.population_variants = variants

        return variants

    def build_superposition_paths(
        self,
        chrom: str,
        region_start: int,
        region_end: int,
        reference_sequences: Dict[str, str],
        consensus_metadata: List[ConsensusBase],
        population_variants: Optional[List[PopulationVariant]] = None
    ) -> List[SuperpositionPath]:
        """
        Create multiple alignment paths for a variable region.

        Strategy:
        1. Extract sequences from all references for this region
        2. Add population variants (indels, SVs) as alternative paths
        3. Compute confidence scores for each path
        4. Return all paths with frequency > threshold

        Args:
            chrom: Chromosome name
            region_start: Start position (0-indexed)
            region_end: End position (0-indexed)
            reference_sequences: Dict mapping reference name to full chromosome sequence
            consensus_metadata: List of ConsensusBase objects for this region
            population_variants: Optional list of population variants in this region

        Returns:
            List of SuperpositionPath objects
        """
        paths = []

        # Extract sequences from all references
        ref_sequences = {}
        for ref_name, full_seq in reference_sequences.items():
            if region_end <= len(full_seq):
                ref_sequences[ref_name] = full_seq[region_start:region_end]

        if not ref_sequences:
            logger.warning(f"No reference sequences for {chrom}:{region_start}-{region_end}")
            return paths

        # Group identical sequences and track sources
        sequence_sources = defaultdict(list)
        for ref_name, seq in ref_sequences.items():
            sequence_sources[seq].append(ref_name)

        # Create paths from reference sequences
        total_refs = len(ref_sequences)
        for seq, sources in sequence_sources.items():
            # Confidence based on number of supporting references
            confidence = len(sources) / total_refs

            # Population frequency: assume uniform if not in database
            pop_freq = 1.0 / len(sequence_sources)

            path = SuperpositionPath(
                allele_sequence=seq,
                population_frequency=pop_freq,
                source_references=sources,
                confidence=confidence,
                path_id="",  # Will be auto-generated
                is_reference_path=(confidence == max(
                    len(s) / total_refs for s in sequence_sources.values()
                ))
            )
            paths.append(path)

        # Add population variants as alternative paths
        if population_variants:
            for variant in population_variants:
                # Check if variant overlaps this region
                if not (region_start <= variant.position < region_end):
                    continue

                for i, alt_allele in enumerate(variant.alt_alleles):
                    # Get frequency for this allele
                    freq = variant.allele_frequencies[i] if i < len(variant.allele_frequencies) else 0.0

                    if freq < self.pop_var_threshold:
                        continue

                    # Build alternative sequence with this allele
                    # (simplified - full implementation would handle complex variants)
                    rel_pos = variant.position - region_start

                    # Use consensus sequence as base
                    consensus_seq = ''.join(c.base for c in consensus_metadata)

                    if rel_pos >= 0 and rel_pos < len(consensus_seq):
                        # Replace ref with alt
                        alt_seq = (
                            consensus_seq[:rel_pos] +
                            alt_allele +
                            consensus_seq[rel_pos + len(variant.ref_allele):]
                        )

                        path = SuperpositionPath(
                            allele_sequence=alt_seq,
                            population_frequency=freq,
                            source_references=[f"gnomAD:{variant.variant_id}"],
                            confidence=freq,  # Use population frequency as confidence
                            path_id=f"var_{variant.variant_id}_{i}",
                            is_reference_path=False
                        )
                        paths.append(path)
                        self.superposition_stats['population_variants_used'] += 1

        self.superposition_stats['total_paths'] += len(paths)

        return paths

    def build_superposition_consensus(
        self,
        references: List[Path],
        output_dir: Path,
        chromosomes: Optional[List[str]] = None,
        population_vcf: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Build superposition consensus from multiple references.

        Args:
            references: List of paths to reference FASTA files
            output_dir: Directory for output files
            chromosomes: Optional list of chromosomes to process
            population_vcf: Optional path to population variant VCF

        Returns:
            Dict mapping output type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*80)
        logger.info("SUPERPOSITION CONSENSUS BUILDER")
        logger.info("="*80)

        # Load all references using parent class method
        logger.info(f"Loading {len(references)} reference genomes...")
        ref_data = {}
        for ref_path in references:
            ref_name = ref_path.stem.replace('.fa', '').replace('.fasta', '')
            ref_data[ref_name] = self.load_reference(Path(ref_path), ref_name)

        # Load population variants if provided
        if population_vcf:
            self.load_population_variants(
                vcf_path=population_vcf,
                chromosomes=set(chromosomes) if chromosomes else None
            )

        # Determine chromosomes to process
        all_chroms = set()
        for ref_seqs in ref_data.values():
            all_chroms.update(ref_seqs.keys())

        if chromosomes:
            chroms_to_process = [c for c in chromosomes if c in all_chroms]
        else:
            chroms_to_process = sorted(all_chroms)

        logger.info(f"Processing {len(chroms_to_process)} chromosomes...")

        # Build consensus for each chromosome
        consensus_sequences = {}
        consensus_metadata_dict = {}

        for chrom in chroms_to_process:
            # Get sequences for this chromosome
            chrom_seqs = {}
            for ref_name, ref_seqs in ref_data.items():
                if chrom in ref_seqs:
                    chrom_seqs[ref_name] = ref_seqs[chrom]

            if len(chrom_seqs) < 2:
                logger.warning(f"Skipping {chrom}: only {len(chrom_seqs)} reference(s)")
                continue

            # Build basic consensus using parent class
            consensus_seq, metadata = self.build_consensus_chromosome(
                chrom, chrom_seqs, self.quality_weight
            )

            consensus_sequences[chrom] = consensus_seq
            consensus_metadata_dict[chrom] = metadata

            # Identify conserved/variable regions
            regions = self.identify_conserved_regions(chrom, metadata)

            # Build superposition nodes for variable regions
            chrom_pop_vars = self.population_variants.get(chrom, [])

            for start, end, is_conserved in regions:
                if is_conserved:
                    # Conserved region - single path
                    region_seq = consensus_seq[start:end]
                    conserved = ConservedRegion(
                        chromosome=chrom,
                        start=start,
                        end=end,
                        conservation_score=1.0,
                        sequence=region_seq
                    )
                    self.conserved_regions[chrom].append(conserved)

                    # Create single-path superposition node
                    path = SuperpositionPath(
                        allele_sequence=region_seq,
                        population_frequency=1.0,
                        source_references=list(chrom_seqs.keys()),
                        confidence=1.0,
                        path_id=f"{chrom}_{start}_{end}_conserved",
                        is_reference_path=True
                    )

                    node = SuperpositionNode(
                        chromosome=chrom,
                        position=start,
                        end_position=end,
                        is_conserved=True,
                        paths=[path],
                        consensus_base=region_seq[0] if region_seq else 'N',
                        conservation_score=1.0
                    )
                    self.superposition_nodes[chrom].append(node)

                else:
                    # Variable region - multiple paths
                    # Get population variants in this region
                    region_vars = [
                        v for v in chrom_pop_vars
                        if start <= v.position < end
                    ]

                    # Build multiple paths
                    paths = self.build_superposition_paths(
                        chrom=chrom,
                        region_start=start,
                        region_end=end,
                        reference_sequences=chrom_seqs,
                        consensus_metadata=metadata[start:end],
                        population_variants=region_vars
                    )

                    variable = VariableRegion(
                        chromosome=chrom,
                        start=start,
                        end=end,
                        conservation_score=0.5,  # Variable by definition
                        paths=paths,
                        population_variants=region_vars
                    )
                    self.variable_regions[chrom].append(variable)

                    # Create multi-path superposition node
                    node = SuperpositionNode(
                        chromosome=chrom,
                        position=start,
                        end_position=end,
                        is_conserved=False,
                        paths=paths,
                        consensus_base=None,
                        conservation_score=0.5
                    )
                    self.superposition_nodes[chrom].append(node)

        # Write output files
        output_files = self._write_superposition_outputs(
            output_dir,
            consensus_sequences,
            consensus_metadata_dict
        )

        # Print statistics
        self.print_superposition_stats()

        return output_files

    def _write_superposition_outputs(
        self,
        output_dir: Path,
        consensus_sequences: Dict[str, str],
        consensus_metadata: Dict[str, List[ConsensusBase]]
    ) -> Dict[str, Path]:
        """Write all superposition consensus outputs."""
        output_files = {}

        # 1. Linear consensus FASTA (conserved regions)
        linear_fasta = output_dir / "consensus_linear.fa"
        self.write_consensus_fasta(linear_fasta, consensus_sequences)
        output_files['linear_fasta'] = linear_fasta

        # 2. Superposition paths metadata (JSON)
        paths_json = output_dir / "superposition_paths.json"
        self._write_paths_json(paths_json)
        output_files['paths_json'] = paths_json

        # 3. Conserved regions (BED)
        conserved_bed = output_dir / "conserved_regions.bed"
        self._write_conserved_bed(conserved_bed)
        output_files['conserved_bed'] = conserved_bed

        # 4. Variable regions (BED)
        variable_bed = output_dir / "variable_regions.bed"
        self._write_variable_bed(variable_bed)
        output_files['variable_bed'] = variable_bed

        # 5. Path statistics (JSON)
        stats_json = output_dir / "path_statistics.json"
        self._write_statistics_json(stats_json)
        output_files['stats_json'] = stats_json

        # 6. Variation graph (if enabled)
        if self.use_graph:
            vg_file = output_dir / "consensus.vg"
            self.export_variation_graph(vg_file, format="vg")
            output_files['variation_graph'] = vg_file

        return output_files

    def _write_paths_json(self, output_path: Path):
        """Write superposition paths to JSON."""
        logger.info(f"Writing superposition paths to {output_path}")

        paths_data = {}
        for chrom, nodes in self.superposition_nodes.items():
            paths_data[chrom] = []
            for node in nodes:
                node_data = {
                    'position': node.position,
                    'end_position': node.end_position,
                    'is_conserved': node.is_conserved,
                    'conservation_score': node.conservation_score,
                    'paths': [
                        {
                            'path_id': p.path_id,
                            'sequence': p.allele_sequence,
                            'population_frequency': p.population_frequency,
                            'confidence': p.confidence,
                            'sources': p.source_references,
                            'is_reference': p.is_reference_path
                        }
                        for p in node.paths
                    ]
                }
                paths_data[chrom].append(node_data)

        with open(output_path, 'w') as f:
            json.dump(paths_data, f, indent=2)

        logger.info(f"  Wrote {sum(len(n) for n in paths_data.values())} nodes")

    def _write_conserved_bed(self, output_path: Path):
        """Write conserved regions to BED format."""
        logger.info(f"Writing conserved regions to {output_path}")

        with open(output_path, 'w') as f:
            f.write("# Chromosome\tStart\tEnd\tConservation\tLength\n")

            total = 0
            for chrom, regions in self.conserved_regions.items():
                for region in regions:
                    f.write(f"{chrom}\t{region.start}\t{region.end}\t"
                           f"{region.conservation_score:.4f}\t{len(region)}\n")
                    total += 1

        logger.info(f"  Wrote {total:,} conserved regions")

    def _write_variable_bed(self, output_path: Path):
        """Write variable regions to BED format."""
        logger.info(f"Writing variable regions to {output_path}")

        with open(output_path, 'w') as f:
            f.write("# Chromosome\tStart\tEnd\tNumPaths\tLength\n")

            total = 0
            for chrom, regions in self.variable_regions.items():
                for region in regions:
                    f.write(f"{chrom}\t{region.start}\t{region.end}\t"
                           f"{len(region.paths)}\t{len(region)}\n")
                    total += 1

        logger.info(f"  Wrote {total:,} variable regions")

    def _write_statistics_json(self, output_path: Path):
        """Write comprehensive statistics to JSON."""
        logger.info(f"Writing statistics to {output_path}")

        stats = {
            'superposition_stats': self.superposition_stats,
            'consensus_stats': self.stats,
            'summary': {
                'total_regions': (
                    self.superposition_stats['conserved_regions'] +
                    self.superposition_stats['variable_regions']
                ),
                'conservation_rate': (
                    100 * self.superposition_stats['conserved_bases'] /
                    (self.superposition_stats['conserved_bases'] +
                     self.superposition_stats['variable_bases'])
                    if (self.superposition_stats['conserved_bases'] +
                        self.superposition_stats['variable_bases']) > 0
                    else 0.0
                ),
                'avg_paths_per_variable_region': (
                    self.superposition_stats['total_paths'] /
                    self.superposition_stats['variable_regions']
                    if self.superposition_stats['variable_regions'] > 0
                    else 0.0
                )
            }
        }

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def export_variation_graph(
        self,
        output_path: Path,
        format: str = "vg"
    ):
        """
        Export superposition consensus as variation graph.

        Formats:
        - "vg": Variation graph toolkit format (JSON-based)
        - "gfa": Graphical Fragment Assembly format
        - "multi_fasta": Multiple FASTA with path annotations

        Args:
            output_path: Path for output file
            format: Output format (vg, gfa, multi_fasta)
        """
        logger.info(f"Exporting variation graph to {output_path} (format: {format})")

        if format == "vg":
            self._export_vg_format(output_path)
        elif format == "gfa":
            self._export_gfa_format(output_path)
        elif format == "multi_fasta":
            self._export_multi_fasta(output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_vg_format(self, output_path: Path):
        """Export to VG (Variation Graph) JSON format."""
        # Simplified VG format - real implementation would use vg toolkit
        vg_data = {
            'node': [],
            'edge': [],
            'path': []
        }

        node_id = 1
        for chrom, nodes in self.superposition_nodes.items():
            for node in nodes:
                # Add nodes for each path
                for path in node.paths:
                    vg_data['node'].append({
                        'id': node_id,
                        'sequence': path.allele_sequence
                    })

                    # Add path membership
                    vg_data['path'].append({
                        'name': path.path_id,
                        'mapping': [{
                            'position': {'node_id': node_id},
                            'rank': node.position
                        }]
                    })

                    node_id += 1

        with open(output_path, 'w') as f:
            json.dump(vg_data, f, indent=2)

        logger.info(f"  Exported {len(vg_data['node'])} nodes, {len(vg_data['path'])} paths")

    def _export_gfa_format(self, output_path: Path):
        """Export to GFA (Graphical Fragment Assembly) format."""
        with open(output_path, 'w') as f:
            f.write("H\tVN:Z:1.0\n")  # Header

            segment_id = 1
            for chrom, nodes in self.superposition_nodes.items():
                for node in nodes:
                    for path in node.paths:
                        # Segment line: S <id> <sequence>
                        f.write(f"S\t{segment_id}\t{path.allele_sequence}\n")

                        # Path line: P <path_name> <segments> <overlaps>
                        f.write(f"P\t{path.path_id}\t{segment_id}+\t*\n")

                        segment_id += 1

        logger.info(f"  Exported GFA with {segment_id-1} segments")

    def _export_multi_fasta(self, output_path: Path):
        """Export to multi-FASTA with path annotations."""
        with open(output_path, 'w') as f:
            for chrom, nodes in self.superposition_nodes.items():
                for node in nodes:
                    for path in node.paths:
                        # FASTA header with metadata
                        header = (
                            f">{path.path_id} "
                            f"chrom={chrom} "
                            f"pos={node.position}-{node.end_position} "
                            f"freq={path.population_frequency:.4f} "
                            f"conf={path.confidence:.4f}"
                        )
                        f.write(header + "\n")

                        # Write sequence in 60-char lines
                        seq = path.allele_sequence
                        for i in range(0, len(seq), 60):
                            f.write(seq[i:i+60] + "\n")

        logger.info(f"  Exported multi-FASTA")

    def print_superposition_stats(self):
        """Print superposition consensus statistics."""
        logger.info("=" * 80)
        logger.info("SUPERPOSITION CONSENSUS STATISTICS")
        logger.info("=" * 80)

        # Basic consensus stats
        self.print_stats()

        # Superposition-specific stats
        logger.info("")
        logger.info("Superposition Structure:")
        logger.info(f"  Conserved regions:  {self.superposition_stats['conserved_regions']:,} "
                   f"({self.superposition_stats['conserved_bases']:,} bases)")
        logger.info(f"  Variable regions:   {self.superposition_stats['variable_regions']:,} "
                   f"({self.superposition_stats['variable_bases']:,} bases)")

        total_bases = (
            self.superposition_stats['conserved_bases'] +
            self.superposition_stats['variable_bases']
        )
        if total_bases > 0:
            cons_pct = 100 * self.superposition_stats['conserved_bases'] / total_bases
            var_pct = 100 * self.superposition_stats['variable_bases'] / total_bases
            logger.info(f"  Conservation rate:  {cons_pct:.2f}% conserved, {var_pct:.2f}% variable")

        logger.info("")
        logger.info(f"  Total paths:        {self.superposition_stats['total_paths']:,}")
        if self.superposition_stats['variable_regions'] > 0:
            avg_paths = (
                self.superposition_stats['total_paths'] /
                self.superposition_stats['variable_regions']
            )
            logger.info(f"  Avg paths/variable: {avg_paths:.2f}")

        logger.info("")
        logger.info("Population Variants:")
        logger.info(f"  Loaded:             {self.superposition_stats['population_variants_loaded']:,}")
        logger.info(f"  Used in paths:      {self.superposition_stats['population_variants_used']:,}")

        logger.info("=" * 80)


def build_superposition_consensus(
    references: List[Path],
    output_dir: Path,
    population_vcf: Optional[Path] = None,
    conservation_threshold: float = 0.95,
    population_variant_threshold: float = 0.01,
    chromosomes: Optional[List[str]] = None,
    threads: int = 1
) -> Dict[str, Path]:
    """
    Build superposition consensus reference from multiple public references.

    Convenience function that creates a SuperpositionConsensusBuilder and
    runs the full pipeline.

    Args:
        references: List of paths to reference FASTA files (.fa or .fa.gz)
        output_dir: Directory for output files
        population_vcf: Optional path to population variant VCF (gnomAD, 1000G)
        conservation_threshold: Minimum agreement for conserved region (default: 0.95)
        population_variant_threshold: Minimum allele frequency to include (default: 0.01)
        chromosomes: Optional list of chromosomes to process (default: all)
        threads: Number of threads (default: 1)

    Returns:
        Dict mapping output type to file path
    """
    builder = SuperpositionConsensusBuilder(
        conservation_threshold=conservation_threshold,
        population_variant_threshold=population_variant_threshold,
        use_graph_structure=True,
        verbose=True
    )

    return builder.build_superposition_consensus(
        references=references,
        output_dir=output_dir,
        chromosomes=chromosomes,
        population_vcf=population_vcf
    )
