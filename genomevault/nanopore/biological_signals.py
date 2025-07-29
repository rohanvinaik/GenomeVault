"""
Biological signal detection from hypervector variance patterns.

Detects methylation, structural variants, and other biological
signals from nanopore HV instability patterns.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class BiologicalSignalType(Enum):
    """Types of biological signals detectable from HV variance."""

    METHYLATION_5MC = "5mC"
    METHYLATION_6MA = "6mA"
    OXIDATIVE_DAMAGE = "8oxoG"
    STRUCTURAL_VARIANT = "SV"
    REPEAT_EXPANSION = "repeat"
    RNA_MODIFICATION = "RNA_mod"
    SECONDARY_STRUCTURE = "secondary"
    UNKNOWN_MODIFICATION = "unknown"


@dataclass
class BiologicalSignal:
    """Detected biological signal."""

    signal_type: BiologicalSignalType
    genomic_position: int
    confidence: float
    variance_score: float
    context: str  # Sequence context
    metadata: dict[str, Any]


@dataclass
class ModificationProfile:
    """Profile of base modifications."""

    modification_type: str
    expected_variance_ratio: float
    sequence_motif: str
    dwell_time_change: float


class BiologicalSignalDetector:
    """
    Detects biological signals from HV variance patterns.

    Maps variance spikes and patterns to specific biological
    features like methylation, damage, or structural variants.
    """

    def __init__(
        self,
        anomaly_threshold: float = 3.0,
        min_signal_length: int = 5,
    ):
        """
        Initialize detector.

        Args:
            anomaly_threshold: Z-score threshold for anomalies
            min_signal_length: Minimum consecutive anomalies for signal
        """
        self.anomaly_threshold = anomaly_threshold
        self.min_signal_length = min_signal_length

        # Known modification profiles
        self.modification_profiles = {
            BiologicalSignalType.METHYLATION_5MC: ModificationProfile(
                modification_type="5-methylcytosine",
                expected_variance_ratio=1.8,
                sequence_motif="CG",
                dwell_time_change=1.3,
            ),
            BiologicalSignalType.METHYLATION_6MA: ModificationProfile(
                modification_type="6-methyladenine",
                expected_variance_ratio=1.5,
                sequence_motif="GATC",
                dwell_time_change=1.2,
            ),
            BiologicalSignalType.OXIDATIVE_DAMAGE: ModificationProfile(
                modification_type="8-oxoguanine",
                expected_variance_ratio=2.2,
                sequence_motif="GGG",
                dwell_time_change=1.4,
            ),
        }

        # Initialize anomaly detector
        self.anomaly_detector = IsolationForest(
            contamination=0.01,
            random_state=42,
        )

    def detect_signals(
        self,
        variance_array: np.ndarray,
        dwell_times: np.ndarray | None = None,
        sequence_context: str | None = None,
        genomic_positions: np.ndarray | None = None,
    ) -> list[BiologicalSignal]:
        """
        Detect biological signals from variance patterns.

        Args:
            variance_array: HV variance values
            dwell_times: Optional dwell time array
            sequence_context: Optional sequence string
            genomic_positions: Optional position array

        Returns:
            List of detected biological signals
        """
        signals = []

        # Detect variance anomalies
        anomaly_regions = self._detect_anomaly_regions(variance_array)

        # Classify each anomaly region
        for start, end in anomaly_regions:
            region_variance = variance_array[start:end]
            region_dwells = dwell_times[start:end] if dwell_times is not None else None

            # Extract features
            features = self._extract_region_features(
                region_variance,
                region_dwells,
                sequence_context[start:end] if sequence_context else None,
            )

            # Classify signal type
            signal_type, confidence = self._classify_signal(features)

            # Create signal object
            if confidence > 0.5:
                signal = BiologicalSignal(
                    signal_type=signal_type,
                    genomic_position=(
                        genomic_positions[start] if genomic_positions is not None else start
                    ),
                    confidence=confidence,
                    variance_score=float(np.max(region_variance)),
                    context=(
                        sequence_context[max(0, start - 5) : end + 5] if sequence_context else ""
                    ),
                    metadata={
                        "region_length": end - start,
                        "mean_variance": float(np.mean(region_variance)),
                        "features": features,
                    },
                )
                signals.append(signal)

        return signals

    def _detect_anomaly_regions(
        self,
        variance_array: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Detect contiguous anomaly regions."""
        # Z-score normalization
        z_scores = zscore(variance_array)
        anomalies = np.abs(z_scores) > self.anomaly_threshold

        # Find contiguous regions
        regions = []
        start = None

        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly and start is None:
                start = i
            elif not is_anomaly and start is not None:
                if i - start >= self.min_signal_length:
                    regions.append((start, i))
                start = None

        # Handle region extending to end
        if start is not None and len(anomalies) - start >= self.min_signal_length:
            regions.append((start, len(anomalies)))

        return regions

    def _extract_region_features(
        self,
        variance: np.ndarray,
        dwell_times: np.ndarray | None,
        sequence: str | None,
    ) -> dict[str, float]:
        """Extract features from anomaly region."""
        features = {
            "variance_mean": float(np.mean(variance)),
            "variance_std": float(np.std(variance)),
            "variance_max": float(np.max(variance)),
            "variance_skew": float(self._safe_skew(variance)),
            "length": len(variance),
        }

        # Dwell time features
        if dwell_times is not None:
            features.update(
                {
                    "dwell_mean": float(np.mean(dwell_times)),
                    "dwell_std": float(np.std(dwell_times)),
                    "dwell_ratio": (
                        float(np.mean(dwell_times) / np.median(dwell_times))
                        if np.median(dwell_times) > 0
                        else 1.0
                    ),
                }
            )

        # Sequence features
        if sequence:
            features.update(
                {
                    "gc_content": (
                        (sequence.count("G") + sequence.count("C")) / len(sequence)
                        if sequence
                        else 0.5
                    ),
                    "has_cpg": 1.0 if "CG" in sequence else 0.0,
                    "has_gatc": 1.0 if "GATC" in sequence else 0.0,
                    "homopolymer_frac": self._homopolymer_fraction(sequence),
                }
            )

        return features

    def _classify_signal(
        self,
        features: dict[str, float],
    ) -> tuple[BiologicalSignalType, float]:
        """
        Classify signal type based on features.

        Returns:
            (signal_type, confidence)
        """
        # Simple rule-based classification
        # In production, would use trained ML model

        scores = {}

        # Check for methylation patterns
        if features.get("has_cpg", 0) > 0 and features.get("dwell_ratio", 1) > 1.2:
            scores[BiologicalSignalType.METHYLATION_5MC] = 0.8

        if features.get("has_gatc", 0) > 0 and features.get("variance_mean", 0) > 2.0:
            scores[BiologicalSignalType.METHYLATION_6MA] = 0.7

        # Check for oxidative damage
        if features.get("gc_content", 0) > 0.7 and features.get("variance_max", 0) > 3.0:
            scores[BiologicalSignalType.OXIDATIVE_DAMAGE] = 0.75

        # Check for structural variants
        if features.get("length", 0) > 20 and features.get("variance_std", 0) < 0.5:
            scores[BiologicalSignalType.STRUCTURAL_VARIANT] = 0.6

        # Check for repeats
        if features.get("homopolymer_frac", 0) > 0.3:
            scores[BiologicalSignalType.REPEAT_EXPANSION] = 0.65

        # Default to unknown if no clear match
        if not scores:
            return BiologicalSignalType.UNKNOWN_MODIFICATION, 0.5

        # Return highest scoring type
        best_type = max(scores, key=scores.get)
        return best_type, scores[best_type]

    def _safe_skew(self, data: np.ndarray) -> float:
        """Calculate skewness safely."""
        if len(data) < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        return np.mean(((data - mean) / std) ** 3)

    def _homopolymer_fraction(self, sequence: str) -> float:
        """Calculate fraction of homopolymer runs."""
        if not sequence:
            return 0.0

        homopolymer_length = 0
        current_base = sequence[0]
        current_length = 1

        for base in sequence[1:]:
            if base == current_base:
                current_length += 1
            else:
                if current_length >= 3:
                    homopolymer_length += current_length
                current_base = base
                current_length = 1

        if current_length >= 3:
            homopolymer_length += current_length

        return homopolymer_length / len(sequence)

    def train_on_known_modifications(
        self,
        training_data: list[tuple[np.ndarray, BiologicalSignalType]],
    ):
        """
        Train detector on known modifications.

        Args:
            training_data: List of (variance_array, signal_type) pairs
        """
        # Extract features from training data
        X = []
        y = []

        for variance_array, signal_type in training_data:
            # Detect regions
            regions = self._detect_anomaly_regions(variance_array)

            for start, end in regions:
                features = self._extract_region_features(
                    variance_array[start:end],
                    None,
                    None,
                )

                # Convert to feature vector
                feature_vector = [features.get(k, 0) for k in sorted(features.keys())]
                X.append(feature_vector)
                y.append(signal_type.value)

        # Train anomaly detector
        if X:
            self.anomaly_detector.fit(X)
            logger.info(f"Trained on {len(X)} modification examples")

    def export_signal_track(
        self,
        signals: list[BiologicalSignal],
        output_format: str = "bedgraph",
    ) -> str:
        """
        Export signals as genome browser track.

        Args:
            signals: List of detected signals
            output_format: Track format (bedgraph, bed, gff)

        Returns:
            Track data as string
        """
        if output_format == "bedgraph":
            lines = [
                "track type=bedGraph name='HV_Biological_Signals' description='Nanopore HV variance signals'"
            ]

            for signal in signals:
                chrom = signal.metadata.get("chromosome", "chr1")
                start = signal.genomic_position
                end = start + signal.metadata.get("region_length", 1)
                score = signal.variance_score * signal.confidence

                lines.append(f"{chrom}\t{start}\t{end}\t{score:.2f}")

            return "\n".join(lines)

        elif output_format == "bed":
            lines = []

            for i, signal in enumerate(signals):
                chrom = signal.metadata.get("chromosome", "chr1")
                start = signal.genomic_position
                end = start + signal.metadata.get("region_length", 1)
                name = f"{signal.signal_type.value}_{i}"
                score = int(signal.confidence * 1000)

                color = self._get_signal_color(signal.signal_type)

                lines.append(
                    f"{chrom}\t{start}\t{end}\t{name}\t{score}\t.\t{start}\t{end}\t{color}"
                )

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _get_signal_color(self, signal_type: BiologicalSignalType) -> str:
        """Get RGB color for signal type."""
        colors = {
            BiologicalSignalType.METHYLATION_5MC: "255,0,0",  # Red
            BiologicalSignalType.METHYLATION_6MA: "255,128,0",  # Orange
            BiologicalSignalType.OXIDATIVE_DAMAGE: "0,0,255",  # Blue
            BiologicalSignalType.STRUCTURAL_VARIANT: "0,255,0",  # Green
            BiologicalSignalType.REPEAT_EXPANSION: "255,0,255",  # Magenta
            BiologicalSignalType.RNA_MODIFICATION: "128,0,128",  # Purple
            BiologicalSignalType.SECONDARY_STRUCTURE: "0,255,255",  # Cyan
            BiologicalSignalType.UNKNOWN_MODIFICATION: "128,128,128",  # Gray
        }

        return colors.get(signal_type, "0,0,0")


# Example usage
def example_signal_detection():
    """Example of biological signal detection."""
    # Create detector
    detector = BiologicalSignalDetector(
        anomaly_threshold=2.5,
        min_signal_length=3,
    )

    # Simulate variance data with known patterns
    n_positions = 1000
    variance = np.random.gamma(2, 0.5, n_positions)

    # Add methylation signal
    cpg_positions = [100, 250, 500, 750]
    for pos in cpg_positions:
        variance[pos : pos + 5] *= 2.5

    # Add structural variant
    variance[600:650] = np.mean(variance) * 3

    # Detect signals
    signals = detector.detect_signals(
        variance_array=variance,
        genomic_positions=np.arange(n_positions) * 100,  # Assume 100bp spacing
    )

    print(f"Detected {len(signals)} biological signals:")
    for signal in signals:
        print(
            f"  {signal.signal_type.value} at position {signal.genomic_position} "
            f"(confidence: {signal.confidence:.2f})"
        )

    # Export as track
    track_data = detector.export_signal_track(signals, "bedgraph")
    print(f"\nBedGraph track preview:")
    print("\n".join(track_data.split("\n")[:5]))


if __name__ == "__main__":
    example_signal_detection()
