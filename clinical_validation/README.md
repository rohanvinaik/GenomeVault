# GenomeVault Clinical Validation Module

## Overview

The Clinical Validation Module demonstrates GenomeVault's privacy-preserving capabilities using real clinical data. It validates that our zero-knowledge proofs, hypervector encoding, and private information retrieval work correctly with actual patient data while maintaining complete privacy.

## Features

- **Zero-Knowledge Proofs**: Proves clinical thresholds are exceeded without revealing actual values
- **Hypervector Encoding**: Compresses clinical data 10,000x while preserving relationships
- **Private Information Retrieval**: Queries reference data without revealing what's being looked up
- **Clinical Algorithm Validation**: Tests diabetes risk assessment with real patient data
- **Adaptive Architecture**: Works with both real GenomeVault components and simulations

## Quick Start

```bash
# Run the complete setup and validation
./setup_and_test_clinical_validation.sh

# Or run validation directly
python clinical_validation/run_validation.py
```

## Architecture

```
clinical_validation/
├── __init__.py              # Module exports
├── core.py                  # Main validator implementation
├── clinical_circuits.py     # ZK proof circuits for clinical use
├── zk_wrapper.py           # Wrapper for ZK proof generation
├── test_validation.py      # Unit tests for the module
├── run_validation.py       # Main entry point
├── data_sources/           # Clinical data source adapters
│   ├── base.py            # Base data source class
│   ├── nhanes.py          # NHANES dataset adapter
│   └── pima.py            # Pima Indians dataset adapter
└── reports/               # Generated validation reports
```

## Clinical Validation Process

### 1. Data Loading
The validator can load data from multiple sources:
- NHANES (National Health and Nutrition Examination Survey)
- Pima Indians Diabetes Dataset
- Custom clinical datasets

### 2. Privacy-Preserving Analysis

#### Zero-Knowledge Proofs
For diabetes risk assessment, the system proves:
```
(glucose > 126 mg/dL) AND (HbA1c > 6.5%) AND (genetic_risk > threshold)
```
Without revealing the actual glucose, HbA1c, or genetic risk values!

#### Example ZK Proof
```python
# Private inputs (never revealed)
private_inputs = {
    'glucose': 140,      # Actual glucose reading
    'hba1c': 7.2,       # Actual HbA1c percentage
    'genetic_risk_score': 0.8
}

# Public inputs (thresholds)
public_inputs = {
    'glucose_threshold': 126,
    'hba1c_threshold': 6.5,
    'risk_threshold': 0.5
}

# Generate proof
proof = prover.generate_proof(circuit, private_inputs, public_inputs)

# Output: Only reveals "HIGH_RISK" or "NORMAL", not the values!
print(proof.public_output)  # "HIGH_RISK"
```

### 3. Performance Metrics

Typical performance on consumer hardware:
- **ZK Proof Generation**: ~1000ms (simulated), target <1000ms (real)
- **ZK Proof Verification**: ~25ms
- **Proof Size**: 384 bytes
- **Hypervector Encoding**: ~30ms per patient
- **PIR Query**: ~210ms per variant

### 4. Clinical Validity

The system maintains clinical accuracy while preserving privacy:
- **Sensitivity**: 75-85% (detecting true diabetes cases)
- **Specificity**: 80-90% (correctly identifying non-diabetic patients)
- **PPV**: 70-80% (positive predictive value)
- **NPV**: 85-95% (negative predictive value)

## Integration with GenomeVault

The clinical validation module seamlessly integrates with GenomeVault's core components:

1. **Hypervector Transform**: Uses `genomevault.hypervector_transform` for encoding
2. **ZK Proofs**: Leverages `genomevault.zk_proofs` for proof generation
3. **PIR**: Utilizes `genomevault.pir` for private queries
4. **Configuration**: Reads from `genomevault.utils.config`

## Testing

Run the test suite:
```bash
python clinical_validation/test_validation.py
```

This tests:
- Module imports and initialization
- ZK proof generation and verification
- Different risk scenarios
- Clinical data processing
- Integration with GenomeVault components

## Report Generation

After validation, a comprehensive report is generated:
- `genomevault_clinical_validation_report.md`

The report includes:
- Component test results
- Performance metrics
- Clinical algorithm accuracy
- Privacy preservation verification

## Privacy Guarantees

1. **No Raw Data Exposure**: Clinical values never leave the local system
2. **Cryptographic Proofs**: Only risk classifications are revealed
3. **Differential Privacy**: Added noise prevents individual identification
4. **Information-Theoretic PIR**: Reference queries reveal nothing about the query

## Extending the Module

### Adding New Clinical Circuits

```python
from clinical_validation.clinical_circuits import BaseCircuit, ProofData

class MyBiomarkerCircuit(BaseCircuit):
    def __init__(self):
        super().__init__()
        self.name = "MyBiomarkerCircuit"
        
    def generate_witness(self, private_inputs, public_inputs):
        # Generate witness for your biomarker
        pass
        
    def prove(self, witness, public_inputs):
        # Generate ZK proof
        pass
```

### Adding New Data Sources

```python
from clinical_validation.data_sources.base import BaseDataSource

class MyDataSource(BaseDataSource):
    def load_data(self):
        # Load your clinical data
        pass
        
    def get_glucose_column(self):
        return 'glucose_mg_dl'
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- GenomeVault core modules (optional, will use simulation if not available)

## License

Part of the GenomeVault project. See main LICENSE file.

## Support

For questions or issues:
- Open an issue on GitHub
- Check the main GenomeVault documentation
- Run tests with verbose logging: `python -v clinical_validation/test_validation.py`
