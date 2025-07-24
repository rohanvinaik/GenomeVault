# GenomeVault Proof-of-Training (PoT) and ZKML Integration

## Overview

This implementation adds cryptographic Proof-of-Training (PoT) and Zero-Knowledge Machine Learning (ZKML) capabilities to GenomeVault, enabling verifiable and privacy-preserving AI/ML for genomic data processing.

### Key Features

1. **Cryptographic Training Attestation**: Prove that models were trained on declared datasets with specific parameters
2. **Zero-Knowledge Proofs**: Verify model properties without revealing sensitive training data or model internals
3. **Multi-Modal Support**: Handle genomic, transcriptomic, and proteomic data with cross-modal verification
4. **Clinical Compliance**: FDA/EMA-ready validation framework with regulatory attestations
5. **Real-Time Monitoring**: Detect model drift and trigger automated retraining
6. **Privacy Budget Tracking**: Differential privacy audit trails with cryptographic verification
7. **Federated Learning Lineage**: Track model evolution across distributed training

## Architecture

```
genomevault/
├── zk_proofs/
│   └── circuits/
│       ├── training_proof.py          # Core training proof circuit
│       └── multi_modal_training_proof.py  # Multi-modal extension
├── local_processing/
│   ├── model_snapshot.py              # Snapshot logging during training
│   ├── differential_privacy_audit.py  # Privacy budget tracking
│   └── drift_detection.py             # Real-time drift monitoring
├── blockchain/
│   └── contracts/
│       └── training_attestation.py    # On-chain attestation contract
├── clinical/
│   └── model_validation.py            # Clinical validation framework
├── advanced_analysis/
│   └── federated_learning/
│       └── model_lineage.py           # Federated lineage tracking
├── hypervector/
│   └── visualization/
│       └── projector.py               # Semantic drift visualization
├── cli/
│   └── training_proof_cli.py          # CLI tools for verification
└── integration/
    └── proof_of_training.py           # Main integration module
```

## Quick Start

### 1. Basic Training with PoT

```python
from genomevault.integration.proof_of_training import ProofOfTrainingIntegration

# Configure PoT
config = {
    "storage_path": "./pot_output",
    "snapshot_frequency": 50,
    "blockchain_enabled": True,
    "dataset_hash": "your_dataset_hash"
}

# Initialize
pot = ProofOfTrainingIntegration(config)

# Start session
session = pot.start_training_session(
    session_id="training_001",
    model_type="genomic_classifier",
    dataset_info={"name": "TCGA_subset", "size": 10000},
    privacy_budget=(1.0, 1e-5)
)

# During training, log steps
pot.log_training_step(
    session_id="training_001",
    model=model,
    epoch=epoch,
    step=step,
    loss=loss,
    metrics={"accuracy": acc},
    privacy_params={
        "mechanism": PrivacyMechanism.GAUSSIAN,
        "epsilon": 0.1,
        "sensitivity": 1.0
    }
)

# Complete and generate proof
result = pot.complete_training_session(
    session_id="training_001",
    final_model=model,
    final_metrics={"accuracy": 0.95}
)
```

### 2. Multi-Modal Training Verification

```python
# For multi-modal models (genomic + transcriptomic)
config["multimodal"] = True

# The system automatically tracks cross-modal alignment
# and generates proofs of consistent learning across modalities
```

### 3. Clinical Validation

```python
# Validate model for clinical use
validation = pot.validate_model_clinically(
    model_id="model_001",
    model=model,
    clinical_domain="oncology",
    test_data=clinical_test_set,
    validation_level="clinical_decision_support"
)

if validation["passed"]:
    print(f"Model validated for {validation['domain']} use")
```

### 4. Real-Time Monitoring

```python
# Start monitoring deployed model
monitor = pot.start_model_monitoring(
    model_id="deployed_model_001",
    model=model,
    training_summary=training_result["training_summary"]
)

# Process predictions
for input_data, prediction in production_stream:
    result = monitor.process_prediction(
        input_features=input_data,
        prediction=prediction
    )
    
    if result["status"] == "critical":
        # Trigger retraining
        retrain_request = monitor.trigger_retraining_protocol()
```

## CLI Tools

### Verify Training Proof

```bash
# Verify a training proof
genomevault-pot verify-proof \
    --proof-file ./proofs/training_001.json \
    --snapshot-dir ./snapshots/training_001 \
    --verbose

# Analyze semantic drift
genomevault-pot analyze-drift \
    --snapshot-dir ./snapshots/training_001 \
    --output-dir ./drift_analysis \
    --threshold 0.15

# Check on-chain attestation
genomevault-pot check-attestation \
    --contract-address 0x1234... \
    --attestation-id att_001 \
    --verify
```

## Core Components

### 1. Training Proof Circuit

The `TrainingProofCircuit` creates zero-knowledge proofs that verify:
- Model evolved through declared training snapshots
- Training followed expected loss descent patterns
- Final model hash matches the commitment
- Input/output sequence integrity

### 2. Differential Privacy Auditor

Tracks privacy budget consumption with:
- Event-based logging of all privacy operations
- Composition theorem application
- Cryptographic audit trail
- Budget overflow prevention

### 3. Model Snapshot Logger

Captures model state during training:
- Weight hashes at checkpoints
- Hypervector representations
- Gradient statistics
- Sample I/O pairs

### 4. Clinical Validator

Ensures models meet regulatory standards:
- Performance validation on clinical datasets
- Safety and bias assessment
- Regulatory compliance checking
- Capability attestation generation

### 5. Real-Time Monitor

Detects drift in production:
- Covariate shift detection
- Prediction distribution monitoring
- Performance degradation alerts
- Automated retraining triggers

## Configuration

### Basic Configuration

```json
{
    "storage_path": "./genomevault_pot",
    "snapshot_frequency": 50,
    "blockchain_enabled": true,
    "multimodal": false,
    "dataset_hash": "sha256_of_dataset",
    "institution_id": "your_institution",
    "blockchain": {
        "contract_address": "0x...",
        "chain_id": 1,
        "owner_address": "0x...",
        "authorized_verifiers": ["0x...", "0x..."]
    },
    "monitoring_config": {
        "window_size": 1000,
        "drift_check_frequency": 100,
        "alert_cooldown_seconds": 3600
    }
}
```

### Privacy Configuration

```json
{
    "privacy": {
        "default_epsilon": 1.0,
        "default_delta": 1e-5,
        "composition_method": "advanced",
        "noise_mechanisms": {
            "training": "gaussian",
            "inference": "laplace"
        }
    }
}
```

## Security Considerations

1. **Proof Storage**: Store proofs in immutable storage (IPFS, blockchain)
2. **Key Management**: Use secure key storage for signing attestations
3. **Privacy Budgets**: Never exceed configured privacy budgets
4. **Access Control**: Implement proper access controls for clinical validation

## Regulatory Compliance

The system supports compliance with:
- FDA 510(k) and De Novo pathways
- CE Mark requirements
- ISO 13485 and IEC 62304
- HIPAA privacy rules
- GDPR data protection

## Advanced Features

### Federated Learning Support

```python
# Start federated session
session = pot.start_training_session(
    session_id="fed_001",
    model_type="federated_genomic",
    dataset_info={"sites": 5, "total_samples": 50000},
    is_federated=True
)

# Track lineage across federated rounds
lineage = pot.federated_lineages["fed_001"]
lineage.record_local_update(client_id="site_1", ...)
lineage.record_aggregation(aggregator_id="central", ...)
```

### Semantic Drift Visualization

```python
from genomevault.hypervector.visualization import ModelEvolutionVisualizer

visualizer = ModelEvolutionVisualizer()
visualizer.visualize_semantic_space(
    hypervectors=model_evolution,
    labels=epoch_labels,
    save_path="./semantic_evolution.png"
)
```

## Performance Considerations

- **Snapshot Frequency**: Balance between proof granularity and storage
- **Privacy Composition**: Use advanced composition for tighter bounds
- **Proof Generation**: Can be computationally intensive for large models
- **Monitoring Window**: Adjust based on prediction volume

## Troubleshooting

### Common Issues

1. **Privacy Budget Exceeded**
   - Reduce per-step epsilon
   - Increase total budget
   - Use advanced composition

2. **Proof Generation Fails**
   - Check snapshot integrity
   - Verify all snapshots present
   - Ensure proper circuit initialization

3. **Drift Detection Too Sensitive**
   - Adjust threshold parameters
   - Increase monitoring window size
   - Check baseline statistics

## Contributing

See the main GenomeVault contributing guidelines. For PoT-specific contributions:

1. Add tests for new circuits
2. Document privacy guarantees
3. Ensure clinical validation compliance
4. Update CLI tools as needed

## License

Same as GenomeVault main project.

## Citations

If you use the PoT features in your research, please cite:

```bibtex
@software{genomevault_pot,
  title={GenomeVault: Proof-of-Training for Privacy-Preserving Genomic AI},
  author={GenomeVault Contributors},
  year={2024},
  url={https://github.com/genomevault/genomevault}
}
```
