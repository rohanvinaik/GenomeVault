"""
Model Snapshot Logging for Training Provenance

This module captures and stores model snapshots during training,
enabling cryptographic proof of training evolution.
"""
import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch

from genomevault.hypervector.encoding import create_hypervector
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelSnapshot:
    """Represents a single model snapshot during training"""

    snapshot_id: str
    epoch: int
    step: int
    timestamp: int
    model_hash: str
    weight_hash: str
    hypervector_hash: str
    loss: float
    metrics: Dict[str, float]
    gradient_stats: Dict[str, float]
    io_samples: List[str]  # Hashes of input/output pairs
    metadata: Dict[str, Any]


class ModelSnapshotLogger:
    """
    Logs model snapshots during training for proof-of-training.

    Captures:
    1. Model weights at checkpoints
    2. Hypervector representations
    3. Training metrics and gradients
    4. Sample input/output pairs
    """

    def __init__(
        self,
        session_id: str,
        output_dir: str,
        snapshot_frequency: int = 50,
        capture_gradients: bool = True,
        capture_io: bool = True,
        compression: str = "zlib",
    ) -> None:
           """TODO: Add docstring for __init__"""
     """
        Initialize snapshot logger.

        Args:
            session_id: Unique training session ID
            output_dir: Directory to store snapshots
            snapshot_frequency: Epochs between snapshots
            capture_gradients: Whether to capture gradient statistics
            capture_io: Whether to capture input/output samples
            compression: Compression method for snapshots
        """
        self.session_id = session_id
        self.output_dir = Path(output_dir) / session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_frequency = snapshot_frequency
        self.capture_gradients = capture_gradients
        self.capture_io = capture_io
        self.compression = compression

        self.snapshots: List[ModelSnapshot] = []
        self.snapshot_hashes: List[str] = []
        self.io_buffer: List[Tuple[str, str]] = []

        # Framework detection
        self.framework = self._detect_framework()

        logger.info(
            f"Model snapshot logger initialized for session {session_id} "
            f"(framework: {self.framework})"
        )

    def log_snapshot(
        self,
        model: Any,
        epoch: int,
        step: int,
        loss: float,
        metrics: Dict[str, float],
        gradients: Optional[Any] = None,
        force: bool = False,
    ) -> Optional[str]:
           """TODO: Add docstring for log_snapshot"""
     """
        Log a model snapshot if conditions are met.

        Args:
            model: The model object
            epoch: Current epoch
            step: Current training step
            loss: Current loss value
            metrics: Training metrics
            gradients: Model gradients (optional)
            force: Force snapshot regardless of frequency

        Returns:
            Snapshot ID if logged, None otherwise
        """
        # Check if we should snapshot
        if not force and epoch % self.snapshot_frequency != 0:
            return None

        logger.info(f"Creating snapshot at epoch {epoch}, step {step}")

        # Extract model weights
        weights = self._extract_weights(model)
        weight_hash = self._hash_weights(weights)

        # Create hypervector representation
        hypervector = self._create_model_hypervector(weights)
        hypervector_hash = hashlib.sha256(hypervector.tobytes()).hexdigest()

        # Compute gradient statistics
        gradient_stats = {}
        if self.capture_gradients and gradients is not None:
            gradient_stats = self._compute_gradient_stats(gradients)

        # Get IO samples
        io_sample_hashes = []
        if self.capture_io and self.io_buffer:
            # Take last 10 IO pairs
            io_samples = self.io_buffer[-10:]
            io_sample_hashes = [
                hashlib.sha256(f"{inp}{out}".encode()).hexdigest()[:8] for inp, out in io_samples
            ]

        # Create snapshot
        snapshot_data = f"{self.session_id}{epoch}{step}{weight_hash}{time.time()}"
        snapshot_id = hashlib.sha256(snapshot_data.encode()).hexdigest()[:16]

        # Compute model hash (includes architecture)
        model_hash = self._compute_model_hash(model, weight_hash)

        snapshot = ModelSnapshot(
            snapshot_id=snapshot_id,
            epoch=epoch,
            step=step,
            timestamp=int(time.time()),
            model_hash=model_hash,
            weight_hash=weight_hash,
            hypervector_hash=hypervector_hash,
            loss=loss,
            metrics=metrics,
            gradient_stats=gradient_stats,
            io_samples=io_sample_hashes,
            metadata={"framework": self.framework, "compression": self.compression},
        )

        # Save snapshot
        self._save_snapshot(snapshot, weights, hypervector)

        # Update tracking
        self.snapshots.append(snapshot)
        self.snapshot_hashes.append(snapshot.model_hash)

        # Clear IO buffer
        self.io_buffer = []

        logger.info(
            f"Snapshot {snapshot_id} saved: " f"loss={loss:.4f}, model_hash={model_hash[:8]}..."
        )

        return snapshot_id

    def log_io_pair(self, input_data: Any, output_data: Any) -> None:
           """TODO: Add docstring for log_io_pair"""
     """
        Log an input/output pair for training verification.

        Args:
            input_data: Model input
            output_data: Model output
        """
        if not self.capture_io:
            return

        # Hash the IO pair
        input_hash = self._hash_data(input_data)
        output_hash = self._hash_data(output_data)

        self.io_buffer.append((input_hash, output_hash))

        # Limit buffer size
        if len(self.io_buffer) > 1000:
            self.io_buffer = self.io_buffer[-1000:]

    def get_snapshot_merkle_root(self) -> str:
           """TODO: Add docstring for get_snapshot_merkle_root"""
     """
        Compute Merkle root of all snapshots.

        Returns:
            Merkle root hash
        """
        if not self.snapshot_hashes:
            return "0" * 64

        # Build Merkle tree
        current_level = self.snapshot_hashes.copy()

        while len(current_level) > 1:
            next_level = []

            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    left, right = current_level[i], current_level[i + 1]
                else:
                    left, right = current_level[i], current_level[i]

                combined = hashlib.sha256(f"{left}{right}".encode()).hexdigest()
                next_level.append(combined)

            current_level = next_level

        return current_level[0]

    def create_training_summary(self) -> Dict[str, Any]:
           """TODO: Add docstring for create_training_summary"""
     """
        Create a summary of the training session.

        Returns:
            Training summary
        """
        if not self.snapshots:
            return {"error": "No snapshots recorded"}

        # Compute statistics
        losses = [s.loss for s in self.snapshots]

        # Find best snapshot by loss
        best_snapshot_idx = np.argmin(losses)
        best_snapshot = self.snapshots[best_snapshot_idx]

        # Compute semantic drift
        drift_scores = []
        if len(self.snapshots) > 1:
            for i in range(1, len(self.snapshots)):
                prev_hv_hash = self.snapshots[i - 1].hypervector_hash
                curr_hv_hash = self.snapshots[i].hypervector_hash

                # Simple drift metric based on hash difference
                drift = sum(a != b for a, b in zip(prev_hv_hash, curr_hv_hash)) / len(prev_hv_hash)
                drift_scores.append(drift)

        summary = {
            "session_id": self.session_id,
            "total_snapshots": len(self.snapshots),
            "snapshot_frequency": self.snapshot_frequency,
            "start_time": self.snapshots[0].timestamp,
            "end_time": self.snapshots[-1].timestamp,
            "duration_seconds": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            "total_epochs": self.snapshots[-1].epoch,
            "merkle_root": self.get_snapshot_merkle_root(),
            "best_snapshot": {
                "id": best_snapshot.snapshot_id,
                "epoch": best_snapshot.epoch,
                "loss": best_snapshot.loss,
                "metrics": best_snapshot.metrics,
            },
            "loss_trajectory": {
                "initial": losses[0],
                "final": losses[-1],
                "best": min(losses),
                "improvement": losses[0] - losses[-1],
            },
            "semantic_drift": {
                "avg_drift": np.mean(drift_scores) if drift_scores else 0,
                "max_drift": max(drift_scores) if drift_scores else 0,
                "total_drift": sum(drift_scores) if drift_scores else 0,
            },
        }

        return summary

    def export_for_proof(self) -> Dict[str, Any]:
           """TODO: Add docstring for export_for_proof"""
     """
        Export snapshot data for proof generation.

        Returns:
            Proof-ready snapshot data
        """
        return {
            "snapshot_hashes": self.snapshot_hashes,
            "merkle_root": self.get_snapshot_merkle_root(),
            "snapshots": [asdict(s) for s in self.snapshots],
            "summary": self.create_training_summary(),
        }

    def _detect_framework(self) -> str:
           """TODO: Add docstring for _detect_framework"""
     """Detect ML framework being used"""
        try:
            import torch

            return "pytorch"
        except ImportError:
            pass

        try:
            import tensorflow

            return "tensorflow"
        except ImportError:
            pass

        return "unknown"

    def _extract_weights(self, model: Any) -> Dict[str, np.ndarray]:
           """TODO: Add docstring for _extract_weights"""
     """Extract weights from model"""
        weights = {}

        if self.framework == "pytorch":
            for name, param in model.named_parameters():
                weights[name] = param.detach().cpu().numpy()

        elif self.framework == "tensorflow":
            for layer in model.layers:
                for i, weight in enumerate(layer.weights):
                    weight_name = f"{layer.name}_weight_{i}"
                    weights[weight_name] = weight.numpy()

        return weights

    def _hash_weights(self, weights: Dict[str, np.ndarray]) -> str:
           """TODO: Add docstring for _hash_weights"""
     """Compute hash of model weights"""
        # Sort keys for deterministic hashing
        sorted_keys = sorted(weights.keys())

        hasher = hashlib.sha256()
        for key in sorted_keys:
            hasher.update(key.encode())
            hasher.update(weights[key].tobytes())

        return hasher.hexdigest()

    def _create_model_hypervector(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
           """TODO: Add docstring for _create_model_hypervector"""
     """Create hypervector representation of model"""
        # Flatten all weights
        all_weights = []
        for key in sorted(weights.keys()):
            all_weights.extend(weights[key].flatten())

        # Sample weights if too many
        if len(all_weights) > 10000:
            indices = np.linspace(0, len(all_weights) - 1, 10000, dtype=int)
            all_weights = [all_weights[i] for i in indices]

        # Create hypervector (simplified - use actual hypervector encoding)
        hypervector = np.zeros(10000)

        for i, w in enumerate(all_weights):
            # Simple encoding - in practice use proper hypervector operations
            idx = int(abs(w * 1000)) % len(hypervector)
            hypervector[idx] += np.sign(w)

        # Normalize
        norm = np.linalg.norm(hypervector)
        if norm > 0:
            hypervector = hypervector / norm

        return hypervector

    def _compute_gradient_stats(self, gradients: Any) -> Dict[str, float]:
           """TODO: Add docstring for _compute_gradient_stats"""
     """Compute gradient statistics"""
        stats = {}

        if self.framework == "pytorch":
            all_grads = []
            for param in gradients:
                if param.grad is not None:
                    all_grads.extend(param.grad.cpu().numpy().flatten())

            if all_grads:
                stats = {
                    "mean": float(np.mean(all_grads)),
                    "std": float(np.std(all_grads)),
                    "max": float(np.max(np.abs(all_grads))),
                    "norm": float(np.linalg.norm(all_grads)),
                }

        return stats

    def _compute_model_hash(self, model: Any, weight_hash: str) -> str:
           """TODO: Add docstring for _compute_model_hash"""
     """Compute hash including model architecture"""
        model_info = {"weight_hash": weight_hash, "framework": self.framework}

        if self.framework == "pytorch":
            model_info["architecture"] = str(model)
        elif self.framework == "tensorflow":
            model_info["architecture"] = (
                model.to_json() if hasattr(model, "to_json") else str(model)
            )

        model_str = json.dumps(model_info, sort_keys=True)
        return hashlib.sha256(model_str.encode()).hexdigest()

    def _hash_data(self, data: Any) -> str:
           """TODO: Add docstring for _hash_data"""
     """Hash arbitrary data"""
        if isinstance(data, (np.ndarray, torch.Tensor)):
            return hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()[:16]
        elif isinstance(data, (list, tuple)):
            combined = "".join(self._hash_data(item) for item in data)
            return hashlib.sha256(combined.encode()).hexdigest()[:16]
        else:
            return hashlib.sha256(str(data).encode()).hexdigest()[:16]

    def _save_snapshot(
        self, snapshot: ModelSnapshot, weights: Dict[str, np.ndarray], hypervector: np.ndarray
    ) -> None:
           """TODO: Add docstring for _save_snapshot"""
     """Save snapshot to disk"""
        snapshot_dir = self.output_dir / f"snapshot_{snapshot.snapshot_id}"
        snapshot_dir.mkdir(exist_ok=True)

        # Save metadata
        with open(snapshot_dir / "metadata.json", "w") as f:
            json.dump(asdict(snapshot), f, indent=2)

        # Save weights (compressed)
        if self.compression == "zlib":
            import zlib

            weight_data = pickle.dumps(weights)
            compressed = zlib.compress(weight_data)

            with open(snapshot_dir / "weights.pkl.zlib", "wb") as f:
                f.write(compressed)
        else:
            with open(snapshot_dir / "weights.pkl", "wb") as f:
                pickle.dump(weights, f)

        # Save hypervector
        np.save(snapshot_dir / "hypervector.npy", hypervector)

        logger.debug(f"Snapshot saved to {snapshot_dir}")


class SnapshotVerifier:
    """Verify model snapshots for proof generation"""

    @staticmethod
    def verify_snapshot_chain(snapshots: List[ModelSnapshot]) -> bool:
           """TODO: Add docstring for verify_snapshot_chain"""
     """Verify the integrity of a snapshot chain"""
        if not snapshots:
            return True

        # Check temporal ordering
        for i in range(1, len(snapshots)):
            if snapshots[i].timestamp <= snapshots[i - 1].timestamp:
                logger.error(f"Timestamp ordering violation at snapshot {i}")
                return False

            if snapshots[i].epoch < snapshots[i - 1].epoch:
                logger.error(f"Epoch ordering violation at snapshot {i}")
                return False

        # Verify hashes
        for snapshot in snapshots:
            # Recompute snapshot ID
            snapshot_data = (
                f"{snapshot.epoch}{snapshot.step}" f"{snapshot.weight_hash}{snapshot.timestamp}"
            )
            expected_id_prefix = hashlib.sha256(snapshot_data.encode()).hexdigest()[:16]

            if not snapshot.snapshot_id.startswith(expected_id_prefix[:8]):
                logger.error(f"Invalid snapshot ID: {snapshot.snapshot_id}")
                return False

        logger.info("Snapshot chain verification passed")
        return True

    @staticmethod
    def load_snapshot(snapshot_dir: str) -> Tuple[ModelSnapshot, Dict[str, np.ndarray], np.ndarray]:
           """TODO: Add docstring for load_snapshot"""
     """Load a snapshot from disk"""
        snapshot_path = Path(snapshot_dir)

        # Load metadata
        with open(snapshot_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        snapshot = ModelSnapshot(**metadata)

        # Load weights
        weight_file = snapshot_path / "weights.pkl.zlib"
        if weight_file.exists():
            import zlib

            with open(weight_file, "rb") as f:
                compressed = f.read()
            weight_data = zlib.decompress(compressed)
            weights = pickle.loads(weight_data)
        else:
            with open(snapshot_path / "weights.pkl", "rb") as f:
                weights = pickle.load(f)

        # Load hypervector
        hypervector = np.load(snapshot_path / "hypervector.npy")

        return snapshot, weights, hypervector
