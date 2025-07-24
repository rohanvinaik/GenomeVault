# GenomeVault Proof-of-Training Quick Start Guide

## Installation

1. **Install additional dependencies:**
```bash
pip install -r requirements-pot.txt
```

2. **Configure GenomeVault for PoT:**
```bash
cp config/pot_config.yaml config/my_pot_config.yaml
# Edit my_pot_config.yaml with your settings
```

## Basic Usage

### 1. Training with Proof-of-Training

```python
from genomevault.integration.proof_of_training import ProofOfTrainingIntegration

# Initialize with configuration
config = {
    "storage_path": "./my_pot_data",
    "snapshot_frequency": 50,
    "blockchain_enabled": False,
    "dataset_hash": "sha256_of_your_dataset"
}

pot = ProofOfTrainingIntegration(config)

# Start training session
session = pot.start_training_session(
    session_id="my_training_001",
    model_type="genomic_classifier",
    dataset_info={"name": "My Dataset", "size": 10000},
    privacy_budget=(1.0, 1e-5)  # epsilon, delta
)

# During training loop:
for epoch in range(num_epochs):
    # ... training code ...
    
    # Log training progress
    pot.log_training_step(
        session_id="my_training_001",
        model=model,
        epoch=epoch,
        step=step,
        loss=loss_value,
        metrics={"accuracy": acc, "auc": auc}
    )

# After training completes:
result = pot.complete_training_session(
    session_id="my_training_001",
    final_model=model,
    final_metrics=final_metrics
)

# Save the proof
with open("training_proof.json", "w") as f:
    json.dump(result["proof"], f)
```

### 2. Clinical Validation

```python
# Validate model for clinical use
validation = pot.validate_model_clinically(
    model_id="my_model_001",
    model=model,
    clinical_domain="oncology",  # or "cardiology", "rare_disease", etc.
    test_data=(X_test, y_test),
    validation_level="clinical_trial"  # or "research", "diagnostic"
)

if validation["passed"]:
    print("Model validated for clinical use!")
    print(f"Performance metrics: {validation['metrics']}")
```

### 3. Real-Time Monitoring

```python
# Deploy model with monitoring
monitor = pot.start_model_monitoring(
    model_id="deployed_model_001",
    model=model,
    training_summary=result["training_summary"]
)

# In production:
for input_data in production_stream:
    prediction = model.predict(input_data)
    
    # Monitor for drift
    monitoring_result = monitor.process_prediction(
        input_features=input_data,
        prediction=prediction
    )
    
    if monitoring_result["status"] == "critical":
        print("Critical drift detected!")
        retrain_request = monitor.trigger_retraining_protocol()
```

### 4. Federated Learning

```python
# Start federated learning session
session = pot.start_training_session(
    session_id="federated_001",
    model_type="federated_genomic_model",
    dataset_info={"sites": 5, "total_samples": 50000},
    is_federated=True
)

# Track updates from different sites
lineage = pot.federated_lineages["federated_001"]

# Record local update from a site
version_id = lineage.record_local_update(
    client_id="hospital_1",
    parent_version="v0",
    new_model_hash=model_hash,
    metrics={"loss": 0.5, "accuracy": 0.85}
)

# Record aggregation
agg_version = lineage.record_aggregation(
    aggregator_id="central_server",
    input_versions=[("v1", 0.5), ("v2", 0.5)],
    aggregated_model_hash=aggregated_hash,
    aggregation_method="FedAvg",
    metrics={"loss": 0.45}
)
```

## CLI Tools

### Verify a training proof:
```bash
genomevault-pot verify-proof \
    --proof-file ./proofs/training_001.json \
    --snapshot-dir ./snapshots/training_001 \
    --output verification_report.json
```

### Analyze semantic drift:
```bash
genomevault-pot analyze-drift \
    --snapshot-dir ./snapshots/training_001 \
    --output-dir ./drift_analysis \
    --threshold 0.15
```

### Generate proof from snapshots:
```bash
genomevault-pot generate-proof \
    --session-id training_001 \
    --proof-file proof.json \
    --snapshot-dir ./snapshots \
    --dataset-hash abc123...
```

## Common Workflows

### Privacy-Preserving Model Training

```python
from genomevault.local_processing.differential_privacy_audit import PrivacyMechanism

# Configure privacy parameters
privacy_params = {
    "mechanism": PrivacyMechanism.GAUSSIAN,
    "epsilon": 0.1,        # Privacy loss per step
    "delta": 1e-6,         # Failure probability
    "sensitivity": 1.0,    # Query sensitivity
    "data_size": batch_size
}

# Log with privacy tracking
pot.log_training_step(
    session_id=session_id,
    model=model,
    epoch=epoch,
    step=step,
    loss=loss,
    metrics=metrics,
    privacy_params=privacy_params
)
```

### Multi-Modal Model Verification

```python
# Enable multi-modal support
config["multimodal"] = True

# Train with multiple data types
# The system automatically tracks cross-modal correlations
# and generates proofs of aligned learning
```

### Blockchain Attestation

```python
# Enable blockchain in config
config["blockchain_enabled"] = True
config["blockchain"] = {
    "contract_address": "0x...",
    "chain_id": 1,
    "owner_address": "0x...",
    "authorized_verifiers": ["0x...", "0x..."]
}

# Submit attestation after training
attestation_id = pot.submit_attestation(
    session_id="training_001",
    proof=result["proof"],
    submitter_address="0x..."
)
```

## Best Practices

1. **Snapshot Frequency**: Balance between proof granularity and storage
   - Every 50 epochs for standard training
   - Every 10 epochs for critical models

2. **Privacy Budget**: Plan your privacy budget carefully
   - Use advanced composition for tighter bounds
   - Reserve budget for validation and testing

3. **Monitoring Thresholds**: Adjust based on your use case
   - Stricter for clinical applications
   - More lenient for research models

4. **Storage Management**: 
   - Archive old snapshots after proof generation
   - Keep proofs for regulatory compliance (7 years for FDA)

## Troubleshooting

### Issue: "Privacy budget exceeded"
- Reduce per-step epsilon
- Increase total privacy budget
- Use gradient clipping to reduce sensitivity

### Issue: "Snapshot storage full"
- Increase snapshot frequency (fewer snapshots)
- Enable compression in config
- Archive completed training sessions

### Issue: "Proof verification fails"
- Ensure all snapshots are present
- Check snapshot integrity
- Verify dataset hash matches

### Issue: "Drift alerts too frequent"
- Increase drift thresholds
- Extend monitoring window size
- Check for data quality issues

## Example: Complete Genomic Classifier Pipeline

```python
# See examples/proof_of_training_demo.py for a complete example
python examples/proof_of_training_demo.py
```

## Resources

- [Full Documentation](README_PROOF_OF_TRAINING.md)
- [API Reference](docs/api/proof_of_training.md)
- [Configuration Guide](config/pot_config.yaml)
- [Example Scripts](examples/)

## Support

For issues related to Proof-of-Training:
1. Check the troubleshooting section
2. Review the full documentation
3. Open an issue with the `pot` label
