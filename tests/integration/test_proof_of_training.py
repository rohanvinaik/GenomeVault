"""
Integration tests for Proof-of-Training system

This module tests the end-to-end functionality of the PoT implementation.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from genomevault.integration.proof_of_training import \
    ProofOfTrainingIntegration
from genomevault.local_processing.differential_privacy_audit import \
    PrivacyMechanism


class MockModel:
    """Mock model for testing"""

    def __init__(self):
        self.weights = {
            "layer1": np.random.randn(100, 50),
            "layer2": np.random.randn(50, 2),
        }

    def parameters(self):
        return [self.weights["layer1"], self.weights["layer2"]]

    def __str__(self):
        return "MockModel(layers=2)"


class TestProofOfTrainingIntegration:
    """Test suite for PoT integration"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def pot_config(self, temp_dir):
        """Create test configuration"""
        return {
            "storage_path": temp_dir,
            "snapshot_frequency": 2,
            "blockchain_enabled": False,
            "dataset_hash": "test_dataset_hash",
            "institution_id": "test_institution",
            "monitoring_config": {"window_size": 10, "drift_check_frequency": 5},
        }

    @pytest.fixture
    def pot_integration(self, pot_config):
        """Create PoT integration instance"""
        return ProofOfTrainingIntegration(pot_config)

    def test_training_session_lifecycle(self, pot_integration, temp_dir):
        """Test complete training session lifecycle"""
        # Start session
        session_id = "test_session_001"
        session_info = pot_integration.start_training_session(
            session_id=session_id,
            model_type="test_model",
            dataset_info={"name": "test_dataset", "size": 100},
            privacy_budget=(1.0, 1e-5),
        )

        assert session_info["session_id"] == session_id
        assert session_info["status"] == "active"
        assert "snapshot_dir" in session_info

        # Create mock model
        model = MockModel()

        # Log training steps
        for epoch in range(5):
            for step in range(10):
                loss = 1.0 / (epoch + 1)
                metrics = {"accuracy": 0.5 + epoch * 0.1}

                # Privacy parameters
                privacy_params = {
                    "mechanism": PrivacyMechanism.GAUSSIAN,
                    "epsilon": 0.1,
                    "delta": 1e-6,
                    "sensitivity": 1.0,
                    "data_size": 32,
                }

                result = pot_integration.log_training_step(
                    session_id=session_id,
                    model=model,
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    metrics=metrics,
                    privacy_params=privacy_params,
                )

                assert "session_id" in result

                # Check if snapshot was taken
                if epoch % 2 == 0 and step == 0:
                    assert "snapshot_id" in result

        # Complete session
        completion_result = pot_integration.complete_training_session(
            session_id=session_id,
            final_model=model,
            final_metrics={"accuracy": 0.9, "loss": 0.1},
        )

        assert "training_summary" in completion_result
        assert "privacy_report" in completion_result
        assert "proof" in completion_result

        # Verify proof structure
        proof = completion_result["proof"]
        assert proof["circuit_type"] == "training_proof"
        assert "public_inputs" in proof
        assert "commitments" in proof

    def test_multi_modal_training(self, pot_integration):
        """Test multi-modal training support"""
        pot_integration.config["multimodal"] = True

        session_id = "multimodal_test"
        session_info = pot_integration.start_training_session(
            session_id=session_id,
            model_type="multimodal_model",
            dataset_info={"modalities": ["genomic", "transcriptomic"], "size": 1000},
        )

        assert session_info["session_id"] == session_id

        # Log some training steps
        model = MockModel()
        for epoch in range(3):
            pot_integration.log_training_step(
                session_id=session_id,
                model=model,
                epoch=epoch,
                step=0,
                loss=0.5,
                metrics={"accuracy": 0.7},
            )

        # Complete and verify multi-modal proof
        result = pot_integration.complete_training_session(
            session_id=session_id, final_model=model, final_metrics={"accuracy": 0.85}
        )

        assert "proof" in result

    def test_privacy_budget_tracking(self, pot_integration):
        """Test privacy budget enforcement"""
        session_id = "privacy_test"

        # Start with small budget
        pot_integration.start_training_session(
            session_id=session_id,
            model_type="test_model",
            dataset_info={"name": "test", "size": 100},
            privacy_budget=(0.5, 1e-5),  # Small epsilon
        )

        model = MockModel()
        privacy_exceeded = False

        # Try to exceed budget
        for i in range(10):
            result = pot_integration.log_training_step(
                session_id=session_id,
                model=model,
                epoch=0,
                step=i,
                loss=0.5,
                metrics={},
                privacy_params={
                    "mechanism": PrivacyMechanism.GAUSSIAN,
                    "epsilon": 0.1,  # Each step consumes 0.1
                    "delta": 1e-6,
                    "sensitivity": 1.0,
                    "data_size": 32,
                },
            )

            if result.get("privacy_exceeded"):
                privacy_exceeded = True
                break

        assert privacy_exceeded, "Privacy budget should have been exceeded"

    def test_clinical_validation(self, pot_integration):
        """Test clinical validation functionality"""
        model = MockModel()
        test_data = (np.random.randn(100, 100), np.random.randint(0, 2, 100))

        result = pot_integration.validate_model_clinically(
            model_id="test_model_001",
            model=model,
            clinical_domain="oncology",
            test_data=test_data,
            validation_level="research",
        )

        assert "validation_id" in result
        assert "passed" in result
        assert "metrics" in result
        assert "limitations" in result

    def test_drift_monitoring(self, pot_integration):
        """Test real-time drift monitoring"""
        model = MockModel()

        # Start monitoring
        monitor = pot_integration.start_model_monitoring(
            model_id="monitor_test",
            model=model,
            training_summary={
                "feature_statistics": {},
                "best_snapshot": {"metrics": {"accuracy": 0.9}},
            },
        )

        assert monitor is not None

        # Process some predictions
        drift_detected = False
        for i in range(20):
            # Gradually shift input distribution
            features = {f"feature_{j}": np.random.randn() + i * 0.1 for j in range(5)}

            result = monitor.process_prediction(
                input_features=features, prediction=np.random.rand()
            )

            if result["alerts"]:
                drift_detected = True

        # Get summary
        summary = monitor.get_monitoring_summary()
        assert "total_predictions" in summary
        assert summary["total_predictions"] == 20

    def test_federated_learning_support(self, pot_integration):
        """Test federated learning lineage tracking"""
        session_id = "federated_test"

        session_info = pot_integration.start_training_session(
            session_id=session_id,
            model_type="federated_model",
            dataset_info={"sites": 3, "total_size": 3000},
            is_federated=True,
        )

        assert session_info["is_federated"] == True
        assert session_id in pot_integration.federated_lineages

        # Test lineage tracking
        lineage = pot_integration.federated_lineages[session_id]

        # Record local updates
        version1 = lineage.record_local_update(
            client_id="site_1",
            parent_version="v0",
            new_model_hash="hash_1",
            metrics={"loss": 0.5},
        )

        version2 = lineage.record_local_update(
            client_id="site_2",
            parent_version="v0",
            new_model_hash="hash_2",
            metrics={"loss": 0.6},
        )

        # Record aggregation
        agg_version = lineage.record_aggregation(
            aggregator_id="central",
            input_versions=[(version1, 0.5), (version2, 0.5)],
            aggregated_model_hash="hash_agg",
            aggregation_method="FedAvg",
            metrics={"loss": 0.55},
        )

        # Verify lineage
        assert len(lineage.versions) == 4  # Initial + 2 local + 1 aggregation
        assert lineage.current_round == 0

        # Test lineage path
        path = lineage.get_lineage_path(agg_version)
        assert len(path) >= 2

    def test_proof_verification(self, pot_integration, temp_dir):
        """Test proof verification process"""
        # Create and complete a session
        session_id = "verify_test"
        pot_integration.start_training_session(
            session_id=session_id,
            model_type="test_model",
            dataset_info={"name": "test", "size": 100},
        )

        model = MockModel()

        # Log some steps
        for epoch in range(3):
            pot_integration.log_training_step(
                session_id=session_id,
                model=model,
                epoch=epoch,
                step=0,
                loss=0.5,
                metrics={"accuracy": 0.7},
            )

        # Complete session
        result = pot_integration.complete_training_session(
            session_id=session_id, final_model=model, final_metrics={"accuracy": 0.85}
        )

        # Save proof
        proof_path = Path(temp_dir) / "test_proof.json"
        with open(proof_path, "w") as f:
            json.dump(result["proof"], f)

        # Verify proof exists and has correct structure
        assert proof_path.exists()

        with open(proof_path) as f:
            loaded_proof = json.load(f)

        assert loaded_proof["circuit_type"] == "training_proof"
        assert loaded_proof["constraints_satisfied"] == True

    def test_semantic_drift_analysis(self, pot_integration, temp_dir):
        """Test semantic drift analysis functionality"""
        session_id = "semantic_test"

        # Start session
        pot_integration.start_training_session(
            session_id=session_id,
            model_type="test_model",
            dataset_info={"name": "test", "size": 100},
        )

        model = MockModel()

        # Create multiple snapshots
        for epoch in range(10):
            # Modify model to simulate drift
            if epoch > 5:
                model.weights["layer1"] += (
                    np.random.randn(*model.weights["layer1"].shape) * 0.1
                )

            pot_integration.log_training_step(
                session_id=session_id,
                model=model,
                epoch=epoch,
                step=0,
                loss=1.0 / (epoch + 1),
                metrics={"accuracy": 0.5 + epoch * 0.05},
            )

        # Complete session
        result = pot_integration.complete_training_session(
            session_id=session_id, final_model=model, final_metrics={"accuracy": 0.95}
        )

        # Check semantic analysis
        if "semantic_analysis" in result:
            analysis = result["semantic_analysis"]
            assert "avg_drift" in analysis
            assert "max_drift" in analysis
            assert "anomalies_detected" in analysis

    def test_error_handling(self, pot_integration):
        """Test error handling in various scenarios"""

        # Test invalid session ID
        with pytest.raises(KeyError):
            pot_integration.log_training_step(
                session_id="nonexistent",
                model=MockModel(),
                epoch=0,
                step=0,
                loss=0.5,
                metrics={},
            )

        # Test completing non-existent session
        with pytest.raises(KeyError):
            pot_integration.complete_training_session(
                session_id="nonexistent", final_model=MockModel(), final_metrics={}
            )


@pytest.mark.integration
class TestEndToEndScenarios:
    """End-to-end integration scenarios"""

    def test_genomic_classifier_workflow(self):
        """Test complete workflow for genomic classifier"""
        config = {
            "storage_path": "./test_e2e_genomic",
            "snapshot_frequency": 10,
            "blockchain_enabled": False,
            "dataset_hash": "genomic_test_dataset",
        }

        pot = ProofOfTrainingIntegration(config)

        # 1. Start training
        session = pot.start_training_session(
            session_id="genomic_e2e",
            model_type="genomic_classifier",
            dataset_info={"name": "TCGA_subset", "size": 5000, "features": 20000},
            privacy_budget=(2.0, 1e-5),
        )

        # 2. Train model (simplified)
        model = MockModel()
        for epoch in range(5):
            pot.log_training_step(
                session_id="genomic_e2e",
                model=model,
                epoch=epoch,
                step=0,
                loss=1.0 / (epoch + 1),
                metrics={"auc": 0.7 + epoch * 0.05},
            )

        # 3. Complete training
        result = pot.complete_training_session(
            session_id="genomic_e2e", final_model=model, final_metrics={"auc": 0.95}
        )

        # 4. Clinical validation
        val_result = pot.validate_model_clinically(
            model_id="genomic_model_e2e",
            model=model,
            clinical_domain="oncology",
            test_data=(np.random.randn(100, 100), np.random.randint(0, 2, 100)),
            validation_level="research",
        )

        # 5. Deploy with monitoring
        monitor = pot.start_model_monitoring(
            model_id="genomic_model_e2e",
            model=model,
            training_summary=result.get("training_summary", {}),
        )

        # Cleanup
        shutil.rmtree("./test_e2e_genomic", ignore_errors=True)

        assert result is not None
        assert val_result is not None
        assert monitor is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
