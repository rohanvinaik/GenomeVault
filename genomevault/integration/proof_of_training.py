"""
Integration module for Proof-of-Training and ZKML features

This module integrates PoT and ZKML capabilities into the existing
GenomeVault infrastructure.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from genomevault.advanced_analysis.federated_learning.model_lineage import FederatedModelLineage
from genomevault.blockchain.contracts.training_attestation import TrainingAttestationContract
from genomevault.clinical.model_validation import ClinicalModelValidator
from genomevault.hypervector.visualization.projector import ModelEvolutionVisualizer
from genomevault.local_processing.differential_privacy_audit import DifferentialPrivacyAuditor
from genomevault.local_processing.drift_detection import RealTimeModelMonitor
from genomevault.local_processing.model_snapshot import ModelSnapshotLogger
from genomevault.utils.logging import get_logger
from genomevault.zk_proofs.circuits.multi_modal_training_proof import MultiModalTrainingProof
from genomevault.zk_proofs.circuits.training_proof import TrainingProofCircuit

logger = get_logger(__name__)


class ProofOfTrainingIntegration:
    """
    Main integration class for Proof-of-Training functionality in GenomeVault.

    Coordinates:
    1. Model snapshot logging during training
    2. Privacy budget tracking
    3. Training proof generation
    4. On-chain attestation
    5. Clinical validation
    6. Real-time monitoring
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize PoT integration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Component initialization
        self.snapshot_loggers: dict[str, ModelSnapshotLogger] = {}
        self.privacy_auditors: dict[str, DifferentialPrivacyAuditor] = {}
        self.model_monitors: dict[str, RealTimeModelMonitor] = {}
        self.clinical_validators: dict[str, ClinicalModelValidator] = {}
        self.federated_lineages: dict[str, FederatedModelLineage] = {}

        # Blockchain connection
        self.attestation_contract = None
        if config.get("blockchain_enabled"):
            self._init_blockchain()

        # Storage paths
        self.storage_base = Path(config.get("storage_path", "./genomevault_pot"))
        self.storage_base.mkdir(parents=True, exist_ok=True)

        logger.info("Proof-of-Training integration initialized")

    def start_training_session(
        self,
        session_id: str,
        model_type: str,
        dataset_info: dict[str, Any],
        privacy_budget: tuple[float, float] = (1.0, 1e-5),
        is_federated: bool = False,
    ) -> dict[str, Any]:
        """
        Start a new training session with PoT tracking.

        Args:
            session_id: Unique session identifier
            model_type: Type of model being trained
            dataset_info: Information about training dataset
            privacy_budget: (epsilon, delta) privacy budget
            is_federated: Whether this is federated learning

        Returns:
            Session initialization info
        """
        logger.info(f"Starting PoT session {session_id}")

        # Initialize snapshot logger
        snapshot_dir = self.storage_base / "snapshots" / session_id
        self.snapshot_loggers[session_id] = ModelSnapshotLogger(
            session_id=session_id,
            output_dir=str(snapshot_dir.parent),
            snapshot_frequency=self.config.get("snapshot_frequency", 50),
            capture_gradients=True,
            capture_io=True,
        )

        # Initialize privacy auditor
        epsilon, delta = privacy_budget
        self.privacy_auditors[session_id] = DifferentialPrivacyAuditor(
            session_id=session_id, total_epsilon=epsilon, total_delta=delta
        )

        # Initialize federated lineage if needed
        if is_federated:
            initial_model_hash = hashlib.sha256(
                f"{model_type}_{dataset_info.get('hash', 'unknown')}".encode()
            ).hexdigest()

            self.federated_lineages[session_id] = FederatedModelLineage(
                federation_id=session_id, initial_model_hash=initial_model_hash
            )

        return {
            "session_id": session_id,
            "snapshot_dir": str(snapshot_dir),
            "privacy_budget": privacy_budget,
            "is_federated": is_federated,
            "start_time": int(time.time()),
            "status": "active",
        }

    def log_training_step(
        self,
        session_id: str,
        model: Any,
        epoch: int,
        step: int,
        loss: float,
        metrics: dict[str, float],
        gradients: Any | None = None,
        io_pair: tuple[Any, Any] | None = None,
        privacy_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Log a training step with all PoT components.

        Args:
            session_id: Training session ID
            model: Current model state
            epoch: Current epoch
            step: Current step
            loss: Training loss
            metrics: Training metrics
            gradients: Model gradients
            io_pair: (input, output) pair
            privacy_params: Privacy mechanism parameters

        Returns:
            Logging result
        """
        result = {"session_id": session_id, "epoch": epoch, "step": step}

        # Log snapshot if needed
        if session_id in self.snapshot_loggers:
            snapshot_logger = self.snapshot_loggers[session_id]

            # Check if snapshot should be taken
            snapshot_id = snapshot_logger.log_snapshot(
                model=model, epoch=epoch, step=step, loss=loss, metrics=metrics, gradients=gradients
            )

            if snapshot_id:
                result["snapshot_id"] = snapshot_id

            # Log IO pair
            if io_pair:
                snapshot_logger.log_io_pair(io_pair[0], io_pair[1])

        # Track privacy budget
        if session_id in self.privacy_auditors and privacy_params:
            auditor = self.privacy_auditors[session_id]

            event_id, budget_ok = auditor.log_privacy_event(
                mechanism=privacy_params["mechanism"],
                epsilon=privacy_params["epsilon"],
                delta=privacy_params.get("delta", 0),
                sensitivity=privacy_params["sensitivity"],
                data_size=privacy_params.get("data_size", 1),
                operation=f"training_step_{step}",
                metadata={"epoch": epoch, "loss": loss},
            )

            if not budget_ok:
                logger.error(f"Privacy budget exceeded in session {session_id}")
                result["privacy_exceeded"] = True

            result["privacy_event"] = event_id

        return result

    def complete_training_session(
        self, session_id: str, final_model: Any, final_metrics: dict[str, float]
    ) -> dict[str, Any]:
        """
        Complete a training session and generate proof.

        Args:
            session_id: Training session ID
            final_model: Final trained model
            final_metrics: Final training metrics

        Returns:
            Session completion info including proof
        """
        logger.info(f"Completing PoT session {session_id}")

        completion_result = {"session_id": session_id, "completion_time": int(time.time())}

        # Finalize snapshot logging
        if session_id in self.snapshot_loggers:
            snapshot_logger = self.snapshot_loggers[session_id]

            # Force final snapshot
            final_snapshot_id = snapshot_logger.log_snapshot(
                model=final_model,
                epoch=snapshot_logger.snapshots[-1].epoch if snapshot_logger.snapshots else 0,
                step=snapshot_logger.snapshots[-1].step + 1 if snapshot_logger.snapshots else 0,
                loss=final_metrics.get("loss", 0),
                metrics=final_metrics,
                force=True,
            )

            # Generate training summary
            summary = snapshot_logger.create_training_summary()
            completion_result["training_summary"] = summary

            # Export for proof
            proof_data = snapshot_logger.export_for_proof()

            # Generate training proof
            proof = self._generate_training_proof(session_id, proof_data)
            completion_result["proof"] = proof

        # Finalize privacy audit
        if session_id in self.privacy_auditors:
            auditor = self.privacy_auditors[session_id]
            privacy_report = auditor.finalize_session()
            completion_result["privacy_report"] = privacy_report

        # Generate semantic analysis
        if session_id in self.snapshot_loggers:
            semantic_report = self._analyze_semantic_evolution(session_id)
            completion_result["semantic_analysis"] = semantic_report

        return completion_result

    def submit_attestation(
        self, session_id: str, proof: dict[str, Any], submitter_address: str
    ) -> str | None:
        """
        Submit training attestation to blockchain.

        Args:
            session_id: Training session ID
            proof: Training proof
            submitter_address: Blockchain address of submitter

        Returns:
            Attestation ID if successful
        """
        if not self.attestation_contract:
            logger.warning("Blockchain not enabled")
            return None

        try:
            attestation_id = self.attestation_contract.submit_attestation(
                model_hash=proof["public_inputs"]["final_model_hash"],
                dataset_hash=proof["public_inputs"]["dataset_hash"],
                training_start=proof["public_inputs"]["training_start_time"],
                training_end=proof["public_inputs"]["training_end_time"],
                snapshot_merkle_root=proof["commitments"]["snapshot_merkle_root"],
                proof_hash=hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest(),
                submitter=submitter_address,
                metadata={
                    "session_id": session_id,
                    "num_snapshots": proof["public_inputs"]["num_snapshots"],
                },
            )

            logger.info(f"Attestation {attestation_id} submitted for session {session_id}")
            return attestation_id

        except (DatabaseError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to submit attestation: {e}")
            return None

    def start_model_monitoring(
        self, model_id: str, model: Any, training_summary: dict[str, Any]
    ) -> RealTimeModelMonitor:
        """
        Start real-time monitoring for a deployed model.

        Args:
            model_id: Unique model identifier
            model: The deployed model
            training_summary: Training summary from PoT

        Returns:
            Model monitor instance
        """
        # Extract baseline statistics
        baseline_stats = {
            "feature_stats": training_summary.get("feature_statistics", {}),
            "prediction_stats": training_summary.get("prediction_statistics", {}),
            "performance": training_summary.get("best_snapshot", {}).get("metrics", {}),
            "model_hypervector": self._extract_model_hypervector(model),
        }

        # Create monitor
        monitor = RealTimeModelMonitor(
            model_id=model_id,
            baseline_stats=baseline_stats,
            monitoring_config=self.config.get("monitoring_config", {}),
        )

        self.model_monitors[model_id] = monitor

        logger.info(f"Started monitoring for model {model_id}")
        return monitor

    def validate_model_clinically(
        self,
        model_id: str,
        model: Any,
        clinical_domain: str,
        test_data: Any,
        validation_level: str = "clinical_trial",
    ) -> dict[str, Any]:
        """
        Perform clinical validation of a model.

        Args:
            model_id: Model identifier
            model: The model to validate
            clinical_domain: Clinical domain
            test_data: Clinical test dataset
            validation_level: Target validation level

        Returns:
            Validation results
        """
        # Get or create validator
        validator_id = f"validator_{self.config.get('institution_id', 'default')}"

        if validator_id not in self.clinical_validators:
            self.clinical_validators[validator_id] = ClinicalModelValidator(
                validator_id=validator_id
            )

        validator = self.clinical_validators[validator_id]

        # Perform validation
        from genomevault.clinical.model_validation import ClinicalDomain, ValidationLevel

        domain_enum = ClinicalDomain[clinical_domain.upper()]
        level_enum = ValidationLevel[validation_level.upper()]

        validation_result = validator.validate_model(
            model=model,
            model_hash=hashlib.sha256(str(model).encode()).hexdigest(),
            clinical_domain=domain_enum,
            test_data=test_data,
            validation_level=level_enum,
        )

        logger.info(
            f"Clinical validation for model {model_id}: "
            f"{'PASSED' if validation_result.passed else 'FAILED'}"
        )

        return {
            "model_id": model_id,
            "validation_id": validation_result.validation_id,
            "passed": validation_result.passed,
            "domain": clinical_domain,
            "level": validation_level,
            "metrics": validation_result.performance_metrics,
            "limitations": validation_result.limitations,
        }

    def _generate_training_proof(
        self, session_id: str, proof_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate training proof from snapshot data"""
        # Determine if multi-modal
        is_multimodal = self.config.get("multimodal", False)

        if is_multimodal:
            circuit = MultiModalTrainingProof(max_snapshots=len(proof_data["snapshot_hashes"]))

            # Add multi-modal specific inputs
            # (simplified for demo)
            modality_commits = {
                "genomic": hashlib.sha256(b"genomic").hexdigest(),
                "transcriptomic": hashlib.sha256(b"transcriptomic").hexdigest(),
            }

            proof_data["modality_commits"] = modality_commits

        else:
            circuit = TrainingProofCircuit(max_snapshots=len(proof_data["snapshot_hashes"]))

        # Setup inputs
        public_inputs = {
            "final_model_hash": proof_data["snapshot_hashes"][-1],
            "training_metadata": {
                "start_time": proof_data["summary"]["start_time"],
                "end_time": proof_data["summary"]["end_time"],
                "dataset_hash": self.config.get("dataset_hash", "unknown"),
            },
        }

        private_inputs = {
            "snapshot_hashes": proof_data["snapshot_hashes"],
            "model_commit": hashlib.sha256(f"{session_id}_model".encode()).hexdigest(),
            "io_sequence_commit": hashlib.sha256(f"{session_id}_io".encode()).hexdigest(),
            "training_snapshots": proof_data["snapshots"],
        }

        if is_multimodal:
            private_inputs["modality_commits"] = modality_commits

        circuit.setup(public_inputs, private_inputs)
        proof = circuit.generate_proof()

        return proof

    def _analyze_semantic_evolution(self, session_id: str) -> dict[str, Any]:
        """Analyze semantic evolution of model during training"""
        snapshot_logger = self.snapshot_loggers.get(session_id)
        if not snapshot_logger:
            return {}

        # Load hypervectors
        snapshot_dir = self.storage_base / "snapshots" / session_id
        hypervectors = []
        labels = []

        for snapshot in snapshot_logger.snapshots:
            hv_path = snapshot_dir / f"snapshot_{snapshot.snapshot_id}" / "hypervector.npy"
            if hv_path.exists():
                import numpy as np

                hypervectors.append(np.load(hv_path))
                labels.append(f"Epoch {snapshot.epoch}")

        if not hypervectors:
            return {}

        # Create visualizer
        visualizer = ModelEvolutionVisualizer()

        # Detect drift
        drift_scores, anomalies = visualizer.detect_semantic_drift(hypervectors)

        # Analyze trajectory
        visualizer.visualize_semantic_space(hypervectors, labels)
        trajectory_metrics = visualizer.analyze_trajectory_smoothness(
            visualizer.projections.get("umap", hypervectors)
        )

        return {
            "total_snapshots": len(hypervectors),
            "anomalies_detected": len(anomalies),
            "avg_drift": float(np.mean(drift_scores)) if drift_scores else 0,
            "max_drift": float(np.max(drift_scores)) if drift_scores else 0,
            "trajectory_metrics": trajectory_metrics,
            "anomaly_epochs": [labels[i] for i in anomalies] if anomalies else [],
        }

    def _extract_model_hypervector(self, model: Any) -> Any | None:
        """Extract hypervector representation from model"""
        # Simplified - in practice would use actual hypervector encoding
        import numpy as np

        try:
            # Get model parameters
            if hasattr(model, "parameters"):  # PyTorch
                params = []
                for p in model.parameters():
                    params.extend(p.detach().cpu().numpy().flatten()[:100])
            elif hasattr(model, "get_weights"):  # TensorFlow
                weights = model.get_weights()
                params = []
                for w in weights[:10]:
                    params.extend(w.flatten()[:100])
            else:
                return None

            # Create simple hypervector
            hypervector = np.zeros(10000)
            for i, p in enumerate(params[:1000]):
                idx = int(abs(p * 1000)) % len(hypervector)
                hypervector[idx] += np.sign(p)

            # Normalize
            norm = np.linalg.norm(hypervector)
            if norm > 0:
                hypervector = hypervector / norm

            return hypervector

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Failed to extract hypervector: {e}")
            return None

    def _init_blockchain(self):
        """Initialize blockchain connection"""
        try:
            contract_address = self.config["blockchain"]["contract_address"]
            chain_id = self.config["blockchain"]["chain_id"]

            self.attestation_contract = TrainingAttestationContract(
                contract_address=contract_address, chain_id=chain_id
            )

            # Initialize with configured verifiers
            owner = self.config["blockchain"]["owner_address"]
            verifiers = self.config["blockchain"]["authorized_verifiers"]

            self.attestation_contract.initialize(owner, verifiers)

            logger.info(f"Blockchain connection established: {contract_address}")

        except KeyError as e:
            logger.error(f"Failed to initialize blockchain: {e}")
            self.attestation_contract = None


# Export main integration class
__all__ = ["ProofOfTrainingIntegration"]
