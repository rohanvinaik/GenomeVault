#!/usr/bin/env python3
"""
Multi-Lens Biophysical Encoder

Each chemical property (AT/GC, Pu/Py, Am/Ke, etc.) is an independent "lens" with:
- Its own dimension (D)
- Its own chunk size (N)
- Its own overlap percentage
- Its own coverage (2x, 3x overlapping chunks)
- Independent codebook (or shared, configurable)

The lenses produce independent signatures that can be combined as multiplicative
bias factors for confidence adjustment.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LensConfig:
    """Configuration for a single chemical property lens."""
    name: str
    dimension: int = 10000
    chunk_size: int = 2000
    overlap_fraction: float = 0.0  # 0.0 = no overlap, 0.5 = 50% overlap
    coverage: float = 1.0  # 1.0 = 1x, 2.0 = 2x coverage (multiple codebooks)
    seed: int = 42

    # Chemical property encoding rules
    # Maps nucleotide -> contribution sign (+1, -1, or 0 for not in this projection)
    positive_nucleotides: Tuple[str, ...] = ('A',)  # Contribute +1
    negative_nucleotides: Tuple[str, ...] = ('T',)  # Contribute -1

    @property
    def stride(self) -> int:
        """Distance between chunk starts."""
        return int(self.chunk_size * (1 - self.overlap_fraction))

    @property
    def num_codebooks(self) -> int:
        """Number of independent codebooks for coverage."""
        return max(1, int(self.coverage))


@dataclass
class LensStatistics:
    """Detailed statistics for a single lens on a single chunk."""
    lens_name: str
    chunk_id: str

    # Vector statistics
    magnitude: float = 0.0
    mean_component: float = 0.0
    std_component: float = 0.0
    sparsity: float = 0.0  # Fraction of near-zero components

    # Compositional statistics
    positive_count: int = 0  # Number of positive nucleotides in chunk
    negative_count: int = 0  # Number of negative nucleotides in chunk
    neutral_count: int = 0   # Number of nucleotides not in this lens
    imbalance: float = 0.0   # |pos - neg| / (pos + neg)

    # Signal quality
    expected_magnitude: float = 0.0  # Theoretical: sqrt(pos + neg)
    magnitude_ratio: float = 0.0     # actual / expected (deviation from random)
    snr_estimate: float = 0.0        # Signal-to-noise ratio

    # Binary encoding statistics
    binary_positive_frac: float = 0.0  # Fraction of +1 in binary vector
    binary_entropy: float = 0.0        # Information content of binary vector


class ChemicalLens:
    """A single chemical property lens with independent parameters."""

    def __init__(self, config: LensConfig):
        self.config = config
        self.codebooks = []

        # Generate codebook(s) for coverage
        for i in range(config.num_codebooks):
            np.random.seed(config.seed + i * 1000)
            codebook = np.random.choice(
                [-1, 1],
                size=(config.chunk_size, config.dimension)
            ).astype(np.int8)
            self.codebooks.append(codebook)

        logger.info(f"  Lens '{config.name}': D={config.dimension}, N={config.chunk_size}, "
                   f"overlap={config.overlap_fraction:.0%}, coverage={config.coverage}x")

    def encode_chunk(self, sequence: str, chunk_id: str, codebook_idx: int = 0) -> Dict:
        """
        Encode a single chunk with this lens.

        Returns:
            dict with float32 vector, binary vector, and detailed statistics
        """
        assert len(sequence) == self.config.chunk_size
        assert codebook_idx < len(self.codebooks)

        codebook = self.codebooks[codebook_idx]
        vec_f32 = np.zeros(self.config.dimension, dtype=np.float32)

        # Count nucleotides
        pos_count = 0
        neg_count = 0
        neutral_count = 0

        # Single pass encoding
        for i, nuc in enumerate(sequence):
            if nuc in self.config.positive_nucleotides:
                vec_f32 += codebook[i].astype(np.float32)
                pos_count += 1
            elif nuc in self.config.negative_nucleotides:
                vec_f32 -= codebook[i].astype(np.float32)
                neg_count += 1
            else:
                neutral_count += 1

        # Compute statistics
        stats = self._compute_statistics(
            vec_f32, chunk_id, pos_count, neg_count, neutral_count
        )

        # Binary encoding
        bin_vec = np.sign(vec_f32).astype(np.int8)

        return {
            'chunk_id': chunk_id,
            'codebook_idx': codebook_idx,
            'vec_f32': vec_f32,
            'bin_vec': bin_vec,
            'statistics': stats
        }

    def _compute_statistics(self, vec_f32: np.ndarray, chunk_id: str,
                           pos_count: int, neg_count: int, neutral_count: int) -> LensStatistics:
        """Compute detailed statistics for this encoding."""

        # Vector statistics
        magnitude = float(np.linalg.norm(vec_f32))
        mean_comp = float(np.mean(vec_f32))
        std_comp = float(np.std(vec_f32))

        # Sparsity: fraction of components near zero (|x| < 0.1 * std)
        threshold = 0.1 * std_comp if std_comp > 0 else 0.1
        sparsity = float(np.mean(np.abs(vec_f32) < threshold))

        # Compositional statistics
        total_in_lens = pos_count + neg_count
        if total_in_lens > 0:
            imbalance = abs(pos_count - neg_count) / total_in_lens
        else:
            imbalance = 0.0

        # Expected magnitude for random walk
        expected_mag = np.sqrt(total_in_lens) if total_in_lens > 0 else 0.0
        mag_ratio = magnitude / expected_mag if expected_mag > 0 else 0.0

        # SNR estimate
        # Signal: expected dot product with correct position = D
        # Noise std: sqrt(total_in_lens) after normalization
        if total_in_lens > 0:
            snr = self.config.dimension / np.sqrt(total_in_lens * self.config.dimension)
        else:
            snr = 0.0

        # Binary encoding statistics
        bin_vec = np.sign(vec_f32)
        binary_pos_frac = float(np.mean(bin_vec > 0))

        # Binary entropy (information content)
        p = binary_pos_frac
        if 0 < p < 1:
            binary_entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        else:
            binary_entropy = 0.0

        return LensStatistics(
            lens_name=self.config.name,
            chunk_id=chunk_id,
            magnitude=magnitude,
            mean_component=mean_comp,
            std_component=std_comp,
            sparsity=sparsity,
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neutral_count,
            imbalance=imbalance,
            expected_magnitude=expected_mag,
            magnitude_ratio=mag_ratio,
            snr_estimate=snr,
            binary_positive_frac=binary_pos_frac,
            binary_entropy=binary_entropy
        )

    def query_position(self, bin_vec: np.ndarray, position: int,
                      codebook_idx: int = 0) -> Tuple[int, float]:
        """
        Query a position from binary vector.

        Returns:
            (sign, confidence) where sign is +1 (positive nuc), -1 (negative nuc), or 0 (neutral)
        """
        codebook = self.codebooks[codebook_idx]
        pos_vec = codebook[position].astype(np.int32)
        similarity = np.dot(pos_vec, bin_vec.astype(np.int32))

        # Confidence is normalized similarity
        max_possible = self.config.dimension
        confidence = abs(similarity) / max_possible

        return int(np.sign(similarity)), float(confidence)


class MultiLensEncoder:
    """
    Orchestrates multiple independent chemical lenses.

    Each lens has its own parameters and produces independent statistics.
    Results can be combined for confidence-weighted retrieval.
    """

    def __init__(self):
        self.lenses: Dict[str, ChemicalLens] = {}
        logger.info("Initializing MultiLensEncoder")

    def add_lens(self, config: LensConfig) -> None:
        """Add a chemical property lens."""
        self.lenses[config.name] = ChemicalLens(config)

    def add_standard_lenses(self, D: int = 10000, N: int = 2000,
                           overlap: float = 0.0, coverage: float = 1.0) -> None:
        """Add the standard set of chemical property lenses with shared parameters."""

        # Watson-Crick: AT vs GC (actually two lenses: AT and GC)
        self.add_lens(LensConfig(
            name="AT",
            dimension=D, chunk_size=N, overlap_fraction=overlap, coverage=coverage,
            positive_nucleotides=('A',), negative_nucleotides=('T',)
        ))

        self.add_lens(LensConfig(
            name="GC",
            dimension=D, chunk_size=N, overlap_fraction=overlap, coverage=coverage,
            positive_nucleotides=('G',), negative_nucleotides=('C',)
        ))

        # Purine/Pyrimidine (ring structure)
        self.add_lens(LensConfig(
            name="PuPy",
            dimension=D, chunk_size=N, overlap_fraction=overlap, coverage=coverage,
            positive_nucleotides=('A', 'G'), negative_nucleotides=('T', 'C')
        ))

        # Amino/Keto (H-bond donors)
        self.add_lens(LensConfig(
            name="AmKe",
            dimension=D, chunk_size=N, overlap_fraction=overlap, coverage=coverage,
            positive_nucleotides=('A', 'C'), negative_nucleotides=('G', 'T')
        ))

        # Strong/Weak (H-bond count: G/C have 3, A/T have 2)
        self.add_lens(LensConfig(
            name="StWk",
            dimension=D, chunk_size=N, overlap_fraction=overlap, coverage=coverage,
            positive_nucleotides=('G', 'C'), negative_nucleotides=('A', 'T')
        ))

    def encode_sequence(self, sequence: str, seq_id: str = "seq") -> Dict:
        """
        Encode a sequence through all lenses.

        Handles different chunk sizes and overlaps per lens.

        Returns:
            Dictionary with per-lens results and combined statistics
        """
        results = {
            'sequence_id': seq_id,
            'sequence_length': len(sequence),
            'lenses': {}
        }

        for lens_name, lens in self.lenses.items():
            lens_results = self._encode_with_lens(sequence, seq_id, lens)
            results['lenses'][lens_name] = lens_results

        # Compute cross-lens statistics
        results['cross_lens'] = self._compute_cross_lens_stats(results['lenses'])

        return results

    def _encode_with_lens(self, sequence: str, seq_id: str, lens: ChemicalLens) -> Dict:
        """Encode sequence with a single lens, handling overlap and coverage."""

        config = lens.config
        stride = config.stride

        # Generate chunks with overlap
        chunks = []
        start = 0
        chunk_idx = 0

        while start + config.chunk_size <= len(sequence):
            chunk_seq = sequence[start:start + config.chunk_size]
            chunk_id = f"{seq_id}:{start}"

            # Encode with each codebook (for coverage > 1x)
            for cb_idx in range(config.num_codebooks):
                result = lens.encode_chunk(chunk_seq, chunk_id, cb_idx)
                chunks.append(result)

            start += stride
            chunk_idx += 1

        # Aggregate statistics across chunks
        all_stats = [c['statistics'] for c in chunks]

        return {
            'config': {
                'dimension': config.dimension,
                'chunk_size': config.chunk_size,
                'overlap_fraction': config.overlap_fraction,
                'coverage': config.coverage,
                'stride': stride,
                'num_chunks': len(chunks),
                'positive_nucs': config.positive_nucleotides,
                'negative_nucs': config.negative_nucleotides
            },
            'chunks': chunks,
            'aggregate_stats': self._aggregate_lens_stats(all_stats)
        }

    def _aggregate_lens_stats(self, stats_list: List[LensStatistics]) -> Dict:
        """Aggregate statistics across all chunks for a lens."""
        if not stats_list:
            return {}

        return {
            'magnitude': {
                'mean': float(np.mean([s.magnitude for s in stats_list])),
                'std': float(np.std([s.magnitude for s in stats_list])),
                'min': float(np.min([s.magnitude for s in stats_list])),
                'max': float(np.max([s.magnitude for s in stats_list]))
            },
            'imbalance': {
                'mean': float(np.mean([s.imbalance for s in stats_list])),
                'std': float(np.std([s.imbalance for s in stats_list])),
                'min': float(np.min([s.imbalance for s in stats_list])),
                'max': float(np.max([s.imbalance for s in stats_list]))
            },
            'magnitude_ratio': {
                'mean': float(np.mean([s.magnitude_ratio for s in stats_list])),
                'std': float(np.std([s.magnitude_ratio for s in stats_list]))
            },
            'snr_estimate': {
                'mean': float(np.mean([s.snr_estimate for s in stats_list])),
                'std': float(np.std([s.snr_estimate for s in stats_list]))
            },
            'binary_positive_frac': {
                'mean': float(np.mean([s.binary_positive_frac for s in stats_list])),
                'std': float(np.std([s.binary_positive_frac for s in stats_list]))
            },
            'binary_entropy': {
                'mean': float(np.mean([s.binary_entropy for s in stats_list])),
                'std': float(np.std([s.binary_entropy for s in stats_list]))
            },
            'composition': {
                'positive_frac': float(np.mean([
                    s.positive_count / (s.positive_count + s.negative_count + s.neutral_count)
                    for s in stats_list
                ])),
                'negative_frac': float(np.mean([
                    s.negative_count / (s.positive_count + s.negative_count + s.neutral_count)
                    for s in stats_list
                ])),
                'neutral_frac': float(np.mean([
                    s.neutral_count / (s.positive_count + s.negative_count + s.neutral_count)
                    for s in stats_list
                ]))
            }
        }

    def _compute_cross_lens_stats(self, lenses_results: Dict) -> Dict:
        """Compute statistics that compare across lenses."""

        cross_stats = {}

        # Agreement matrix: how often do lenses agree on polarity?
        lens_names = list(lenses_results.keys())

        # Magnitude correlations
        if len(lens_names) >= 2:
            magnitudes = {}
            for name in lens_names:
                chunks = lenses_results[name]['chunks']
                magnitudes[name] = [c['statistics'].magnitude for c in chunks]

            # Compute pairwise correlations
            correlations = {}
            for i, name1 in enumerate(lens_names):
                for name2 in lens_names[i+1:]:
                    if len(magnitudes[name1]) == len(magnitudes[name2]):
                        corr = np.corrcoef(magnitudes[name1], magnitudes[name2])[0, 1]
                        correlations[f"{name1}_vs_{name2}"] = float(corr)

            cross_stats['magnitude_correlations'] = correlations

        return cross_stats


def create_diagnostic_report(results: Dict) -> str:
    """Generate a human-readable diagnostic report."""

    report = []
    report.append("=" * 80)
    report.append("MULTI-LENS ENCODING DIAGNOSTIC REPORT")
    report.append("=" * 80)
    report.append("")

    report.append(f"Sequence ID: {results['sequence_id']}")
    report.append(f"Sequence Length: {results['sequence_length']:,} bp")
    report.append("")

    # Per-lens statistics
    for lens_name, lens_data in results['lenses'].items():
        report.append("-" * 80)
        report.append(f"LENS: {lens_name}")
        report.append("-" * 80)

        config = lens_data['config']
        report.append(f"  Configuration:")
        report.append(f"    Dimension (D): {config['dimension']:,}")
        report.append(f"    Chunk size (N): {config['chunk_size']:,}")
        report.append(f"    Overlap: {config['overlap_fraction']:.0%}")
        report.append(f"    Coverage: {config['coverage']}x")
        report.append(f"    Num chunks: {config['num_chunks']}")
        report.append(f"    Positive nucleotides: {config['positive_nucs']}")
        report.append(f"    Negative nucleotides: {config['negative_nucs']}")
        report.append("")

        stats = lens_data['aggregate_stats']
        report.append(f"  Vector Statistics:")
        report.append(f"    Magnitude: {stats['magnitude']['mean']:.2f} ± {stats['magnitude']['std']:.2f}")
        report.append(f"    Mag ratio (actual/expected): {stats['magnitude_ratio']['mean']:.4f}")
        report.append(f"    SNR estimate: {stats['snr_estimate']['mean']:.4f}")
        report.append("")

        report.append(f"  Compositional Statistics:")
        report.append(f"    Positive fraction: {stats['composition']['positive_frac']:.2%}")
        report.append(f"    Negative fraction: {stats['composition']['negative_frac']:.2%}")
        report.append(f"    Neutral fraction: {stats['composition']['neutral_frac']:.2%}")
        report.append(f"    Imbalance: {stats['imbalance']['mean']:.4f} ± {stats['imbalance']['std']:.4f}")
        report.append("")

        report.append(f"  Binary Encoding:")
        report.append(f"    Positive bit fraction: {stats['binary_positive_frac']['mean']:.4f}")
        report.append(f"    Binary entropy: {stats['binary_entropy']['mean']:.4f} bits")
        report.append("")

    # Cross-lens statistics
    if results['cross_lens']:
        report.append("=" * 80)
        report.append("CROSS-LENS CORRELATIONS")
        report.append("=" * 80)

        if 'magnitude_correlations' in results['cross_lens']:
            for pair, corr in results['cross_lens']['magnitude_correlations'].items():
                report.append(f"  {pair}: r = {corr:.4f}")
        report.append("")

    return "\n".join(report)
