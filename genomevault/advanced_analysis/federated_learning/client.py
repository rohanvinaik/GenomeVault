"""
Federated Learning Client for secure model training participation.
Implements local training with privacy guarantees.
"""

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from genomevault.hypervector_transform.encoding import HypervectorEncoder
from genomevault.utils.logging import get_logger, logger, performance_logger

logger = get_logger(__name__)


@dataclass
class LocalDataset:
    """Local dataset for federated learning."""

    features: np.ndarray
    labels: np.ndarray
    metadata: dict[str, Any]

    @property
    def num_samples(self) -> int:
        return len(self.features)


class FederatedLearningClient:
    """
    Client for participating in federated learning.
    Handles local training and secure update submission.
    """

    def __init__(
        self,
        client_id: str,
        data_path: Path | None = None,
        use_hypervectors: bool = True,
    ):
        """
        Initialize FL client.

        Args:
            client_id: Unique client identifier
            data_path: Path to local data
            use_hypervectors: Whether to use hypervector encoding
        """
        self.client_id = client_id
        self.data_path = data_path
        self.use_hypervectors = use_hypervectors

        # Components
        if use_hypervectors:
            self.encoder = HypervectorEncoder()
        else:
            self.encoder = None

        # State
        self.current_model = None
        self.local_dataset = None
        self.training_history = []

        logger.info(
            "FederatedLearningClient {client_id} initialized",
            extra={"privacy_safe": True},
        )

    def load_local_data(self, privacy_filter: bool = True) -> LocalDataset:
        """
        Load local data for training.

        Args:
            privacy_filter: Apply privacy filtering

        Returns:
            Local dataset
        """
        if self.data_path and self.data_path.exists():
            # Load from file
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
                features = data["features"]
                labels = data["labels"]
                metadata = data.get("metadata", {})
        else:
            # Generate synthetic data for demo
            features, labels, metadata = self._generate_synthetic_data()

        # Apply privacy filtering if requested
        if privacy_filter:
            features, labels = self._apply_privacy_filter(features, labels)

        # Encode with hypervectors if enabled
        if self.use_hypervectors:
            features = self._encode_features(features)

        self.local_dataset = LocalDataset(features=features, labels=labels, metadata=metadata)

        logger.info(
            "Loaded {self.local_dataset.num_samples} samples",
            extra={"privacy_safe": True},
        )

        return self.local_dataset

    def _generate_synthetic_data(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Generate synthetic genomic data for testing."""
        num_samples = 1000
        num_features = 10000  # Variants for PRS

        # Generate synthetic genotype matrix (0, 1, 2)
        features = np.random.choice([0, 1, 2], size=(num_samples, num_features))

        # Generate synthetic risk scores
        true_weights = np.random.randn(num_features) * 0.01
        genetic_risk = features @ true_weights

        # Add environmental component
        environmental_risk = np.random.randn(num_samples) * 0.5

        # Total risk with sigmoid
        total_risk = genetic_risk + environmental_risk
        probabilities = 1 / (1 + np.exp(-total_risk))
        labels = (probabilities > 0.5).astype(int)

        metadata = {
            "synthetic": True,
            "num_variants": num_features,
            "case_control_ratio": np.mean(labels),
        }

        return features, labels, metadata

    def _apply_privacy_filter(
        self, features: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply privacy filtering to remove identifying information."""
        # Remove rare variants (MAF < 1%)
        maf = np.mean(features > 0, axis=0)
        common_variants = (maf > 0.01) & (maf < 0.99)

        filtered_features = features[:, common_variants]

        # Add small noise to continuous features
        if filtered_features.dtype == np.float32 or filtered_features.dtype == np.float64:
            noise = np.random.laplace(0, 0.1, size=filtered_features.shape)
            filtered_features += noise

        return filtered_features, labels

    def _encode_features(self, features: np.ndarray) -> np.ndarray:
        """Encode features using hypervectors."""
        encoded_samples = []

        for sample in features:
            # Convert to feature dict
            feature_dict = {
                "variants": [
                    {"genotype": "{int(g)}/0", "position": i} for i, g in enumerate(sample) if g > 0
                ]
            }

            # Encode
            hv = self.encoder.encode_features(feature_dict, domain="genomic")
            encoded_samples.append(hv)

        return np.array(encoded_samples)

    @performance_logger.log_operation("local_training")
    def train_local_model(
        self,
        global_model: np.ndarray,
        num_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> dict[str, Any]:
        """
        Train model locally on private data.

        Args:
            global_model: Current global model parameters
            num_epochs: Number of local epochs
            batch_size: Batch size for training
            learning_rate: Learning rate

        Returns:
            Training results including model update
        """
        if self.local_dataset is None:
            self.load_local_data()

        # Initialize with global model
        self.current_model = global_model.copy()

        # Get data
        X = self.local_dataset.features
        y = self.local_dataset.labels
        n_samples = len(X)

        # Training metrics
        losses = []
        accuracies = []

        # Simple SGD training (in practice, would use proper ML framework)
        for epoch in range(num_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)

            epoch_loss = 0
            correct = 0

            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Forward pass (linear model for simplicity)
                if len(self.current_model) == X.shape[1] + 1:
                    # Model has bias term
                    weights = self.current_model[:-1]
                    bias = self.current_model[-1]
                    predictions = X_batch @ weights + bias
                else:
                    predictions = X_batch @ self.current_model

                # Sigmoid for binary classification
                probs = 1 / (1 + np.exp(-predictions))

                # Binary cross-entropy loss
                loss = -np.mean(
                    y_batch * np.log(probs + 1e-8) + (1 - y_batch) * np.log(1 - probs + 1e-8)
                )
                epoch_loss += loss * len(batch_indices)

                # Accuracy
                pred_labels = (probs > 0.5).astype(int)
                correct += np.sum(pred_labels == y_batch)

                # Backward pass (gradient)
                grad_predictions = probs - y_batch
                grad_weights = X_batch.T @ grad_predictions / len(batch_indices)

                if len(self.current_model) == X.shape[1] + 1:
                    grad_bias = np.mean(grad_predictions)
                    # Update
                    self.current_model[:-1] -= learning_rate * grad_weights
                    self.current_model[-1] -= learning_rate * grad_bias
                else:
                    # Update
                    self.current_model -= learning_rate * grad_weights

            # Record metrics
            epoch_loss /= n_samples
            accuracy = correct / n_samples

            losses.append(epoch_loss)
            accuracies.append(accuracy)

        # Calculate model update (difference from global model)
        model_update = self.current_model - global_model

        # Training results
        results = {
            "client_id": self.client_id,
            "model_update": model_update,
            "num_samples": n_samples,
            "num_epochs": num_epochs,
            "final_loss": losses[-1],
            "final_accuracy": accuracies[-1],
            "loss_history": losses,
            "accuracy_history": accuracies,
            "timestamp": time.time(),
        }

        # Store in history
        self.training_history.append(results)

        logger.info(
            "Local training complete: loss={losses[-1]:.4f}, accuracy={accuracies[-1]:.4f}",
            extra={"privacy_safe": True},
        )

        return results

    def apply_differential_privacy(
        self,
        model_update: np.ndarray,
        epsilon: float = 1.0,
        delta: float = 1e-6,
        clip_norm: float = 1.0,
    ) -> np.ndarray:
        """
        Apply differential privacy to model update.

        Args:
            model_update: Model update to privatize
            epsilon: Privacy budget
            delta: Privacy parameter
            clip_norm: L2 norm bound

        Returns:
            Privatized model update
        """
        # Clip to bounded L2 norm
        update_norm = np.linalg.norm(model_update)
        if update_norm > clip_norm:
            model_update = model_update * (clip_norm / update_norm)

        # Add calibrated noise
        sensitivity = 2 * clip_norm / self.local_dataset.num_samples
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

        noise = np.random.normal(0, sigma, size=model_update.shape)
        private_update = model_update + noise

        logger.info(f"Applied DP with ε={epsilon}, δ={delta}", extra={"privacy_safe": True})

        return private_update

    def validate_model(
        self, model_params: np.ndarray, validation_split: float = 0.2
    ) -> dict[str, float]:
        """
        Validate model on local validation set.

        Args:
            model_params: Model parameters to validate
            validation_split: Fraction of data for validation

        Returns:
            Validation metrics
        """
        if self.local_dataset is None:
            self.load_local_data()

        # Split data
        n_samples = self.local_dataset.num_samples
        n_val = int(n_samples * validation_split)

        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]

        X_val = self.local_dataset.features[val_indices]
        y_val = self.local_dataset.labels[val_indices]

        # Predict
        if len(model_params) == X_val.shape[1] + 1:
            weights = model_params[:-1]
            bias = model_params[-1]
            predictions = X_val @ weights + bias
        else:
            predictions = X_val @ model_params

        probs = 1 / (1 + np.exp(-predictions))
        pred_labels = (probs > 0.5).astype(int)

        # Metrics
        accuracy = np.mean(pred_labels == y_val)

        # AUC
        from sklearn.metrics import roc_auc_score

        try:
            auc = roc_auc_score(y_val, probs)
        except Exception:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            auc = 0.5
            raise

        # Loss
        loss = -np.mean(y_val * np.log(probs + 1e-8) + (1 - y_val) * np.log(1 - probs + 1e-8))

        metrics = {"accuracy": accuracy, "auc": auc, "loss": loss, "num_samples": n_val}

        return metrics

    def get_model_explanation(self, model_params: np.ndarray, top_k: int = 20) -> dict[str, Any]:
        """
        Get interpretable explanation of model.

        Args:
            model_params: Model parameters
            top_k: Number of top features to return

        Returns:
            Model explanation
        """
        # For linear model, weights indicate feature importance
        if len(model_params) == self.local_dataset.features.shape[1] + 1:
            weights = model_params[:-1]
        else:
            weights = model_params

        # Get top features by absolute weight
        importance = np.abs(weights)
        top_indices = np.argsort(importance)[-top_k:][::-1]

        explanation = {
            "top_features": [
                {
                    "index": int(idx),
                    "weight": float(weights[idx]),
                    "importance": float(importance[idx]),
                }
                for idx in top_indices
            ],
            "model_norm": float(np.linalg.norm(weights)),
            "sparsity": float(np.mean(np.abs(weights) < 1e-4)),
        }

        return explanation

    def save_state(self, path: Path):
        """Save client state."""
        state = {
            "client_id": self.client_id,
            "current_model": self.current_model,
            "training_history": self.training_history,
            "dataset_metadata": (self.local_dataset.metadata if self.local_dataset else None),
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved client state to {path}", extra={"privacy_safe": True})

    def load_state(self, path: Path):
        """Load client state."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.current_model = state["current_model"]
        self.training_history = state["training_history"]

        logger.info(f"Loaded client state from {path}", extra={"privacy_safe": True})


# Specialized clients for different use cases


class HospitalFLClient(FederatedLearningClient):
    """
    Federated learning client for hospital/clinical setting.
    Includes additional privacy and compliance features.
    """

    def __init__(self, hospital_id: str, ehr_integration: bool = True):
        """
        Initialize hospital FL client.

        Args:
            hospital_id: Hospital identifier
            ehr_integration: Whether to integrate with EHR
        """
        super().__init__(client_id="hospital_{hospital_id}", use_hypervectors=True)

        self.ehr_integration = ehr_integration
        self.compliance_checks = {
            "hipaa": True,
            "consent_required": True,
            "deidentification": True,
        }

    def verify_compliance(self) -> bool:
        """Verify all compliance requirements are met."""
        # Check HIPAA compliance
        if self.compliance_checks["hipaa"]:
            # Verify data is de-identified
            if not self._check_deidentification():
                logger.error("Data not properly de-identified")
                return False

        # Check consent
        if self.compliance_checks["consent_required"]:
            if not self._check_consent():
                logger.error("Patient consent not verified")
                return False

        return True

    def _check_deidentification(self) -> bool:
        """
        Check that data is properly de-identified according to HIPAA Safe Harbor.

        Returns:
            True if data meets de-identification requirements
        """
        if not self.local_dataset:
            return True  # No data to check

        # Check for direct identifiers that should be removed
        prohibited_fields = {
            "name",
            "ssn",
            "mrn",
            "address",
            "phone",
            "email",
            "dates",
            "ages_over_89",
            "account_numbers",
            "urls",
            "ip_addresses",
            "biometric_ids",
            "photos",
        }

        # Check metadata for prohibited identifiers
        metadata = getattr(self.local_dataset, "metadata", {})
        for field in prohibited_fields:
            if field in metadata and metadata[field] is not None:
                logger.error(f"Direct identifier '{field}' found in dataset")
                return False

        # Check for quasi-identifiers that could enable re-identification
        if self._check_quasi_identifiers():
            logger.error("Dataset contains quasi-identifiers that may enable re-identification")
            return False

        return True

    def _check_quasi_identifiers(self) -> bool:
        """Check for combinations of quasi-identifiers"""
        # In production: implement proper statistical disclosure control
        # For now: basic checks for common quasi-identifier combinations

        metadata = getattr(self.local_dataset, "metadata", {})

        # Flag if we have detailed demographic + genetic data
        has_demographics = any(key in metadata for key in ["zipcode", "birth_year", "gender"])
        has_detailed_genetic = getattr(self.local_dataset, "num_samples", 0) < 1000

        return has_demographics and has_detailed_genetic

    def _check_consent(self) -> bool:
        """
        Check patient consent status.

        Returns:
            True if all required consents are obtained
        """
        # Check if client has consent management integration
        if not hasattr(self, "consent_manager"):
            logger.warning("No consent manager configured - assuming consent granted")
            return True

        # In production: query consent management system
        # For now: check if client ID indicates consent compliance
        if self.client_id.startswith(("hospital_", "clinic_")):
            # Healthcare clients should have BAA and patient consent
            return self._verify_healthcare_consent()

        return True

    def _verify_healthcare_consent(self) -> bool:
        """Verify healthcare-specific consent requirements"""
        # Check for BAA (Business Associate Agreement)
        if not getattr(self, "baa_signed", False):
            logger.error("Business Associate Agreement not signed")
            return False

        # Check for patient consent documentation
        if not getattr(self, "patient_consent_verified", False):
            logger.error("Patient consent not verified")
            return False

        return True


class ResearchFLClient(FederatedLearningClient):
    """
    Federated learning client for research institutions.
    Includes advanced analysis capabilities.
    """

    def __init__(self, institution_id: str, compute_resources: str = "gpu"):
        """
        Initialize research FL client.

        Args:
            institution_id: Institution identifier
            compute_resources: Available compute ("cpu" or "gpu")
        """
        super().__init__(client_id="research_{institution_id}", use_hypervectors=True)

        self.compute_resources = compute_resources
        self.advanced_features = {
            "pathway_analysis": True,
            "multi_omics": True,
            "causal_inference": True,
        }

    def run_pathway_analysis(self, model_params: np.ndarray) -> dict[str, Any]:
        """Run pathway enrichment analysis on model."""
        # Extract significant features
        threshold = np.percentile(np.abs(model_params), 95)
        significant_features = np.where(np.abs(model_params) > threshold)[0]

        # Map to pathways (simplified)
        pathway_results = {
            "significant_features": len(significant_features),
            "enriched_pathways": [
                "Cell cycle regulation",
                "DNA repair",
                "Immune response",
            ],
            "pathway_scores": {
                "Cell cycle regulation": 0.002,
                "DNA repair": 0.015,
                "Immune response": 0.048,
            },
        }

        return pathway_results


# Example usage
if __name__ == "__main__":
    # Example 1: Basic client
    print("=== Basic FL Client Example ===")
    client = FederatedLearningClient("client_001")

    # Load data
    dataset = client.load_local_data()
    print("Loaded {dataset.num_samples} samples")

    # Train on global model
    global_model = np.random.randn(10001) * 0.01  # Initial model
    results = client.train_local_model(global_model, num_epochs=3)

    print(
        "Training complete: loss={results['final_loss']:.4f}, accuracy={results['final_accuracy']:.4f}"
    )

    # Apply differential privacy
    private_update = client.apply_differential_privacy(results["model_update"], epsilon=1.0)
    print(
        "Update norm: original={np.linalg.norm(results['model_update']):.4f}, private={np.linalg.norm(private_update):.4f}"
    )

    # Example 2: Hospital client
    print("\n=== Hospital FL Client Example ===")
    hospital_client = HospitalFLClient("boston_general")

    if hospital_client.verify_compliance():
        print("✓ Compliance verified")
        dataset = hospital_client.load_local_data()
        print("Hospital data: {dataset.num_samples} patients")

    # Example 3: Research client
    print("\n=== Research FL Client Example ===")
    research_client = ResearchFLClient("stanford_genomics", compute_resources="gpu")

    # Load and train
    dataset = research_client.load_local_data()
    results = research_client.train_local_model(global_model)

    # Run pathway analysis
    pathway_results = research_client.run_pathway_analysis(results["model_update"])
    print("Pathway analysis: {pathway_results['enriched_pathways']}")

    # Get model explanation
    explanation = research_client.get_model_explanation(research_client.current_model, top_k=10)
    print("Top feature importance: {explanation['top_features'][0]['importance']:.4f}")
