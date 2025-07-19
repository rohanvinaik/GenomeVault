# GenomeVault 3.0

![CI Status](https://github.com/genomevault/genomevault/workflows/GenomeVault%20CI/badge.svg)
![Coverage](https://codecov.io/gh/genomevault/genomevault/branch/main/graph/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

A revolutionary privacy-preserving genomic data platform that enables secure analysis and research while maintaining complete individual data sovereignty.

## Overview

GenomeVault 3.0 solves the fundamental tension between advancing precision medicine and protecting individual genetic privacy by combining:
- Hyperdimensional computing for 10,000x data compression
- Zero-knowledge cryptography for verifiable computations  
- Information-theoretic PIR for private queries
- Federated AI for distributed learning
- Blockchain governance with dual-axis node model
- HIPAA fast-track for healthcare providers

## Current Implementation Status

### ✅ Completed Features

1. **Compression System** - Three-tier system with Mini (25KB), Clinical (300KB), and Full HDC (100-200KB) profiles
2. **Hierarchical Hypervector System** - Multi-resolution encoding with domain-specific projections
3. **Zero-Knowledge Proofs** - PLONK-based circuits for variant verification and risk scores
4. **Diabetes Pilot** - Complete implementation with privacy-preserving alerts
5. **HIPAA Fast-Track** - Automated verification for healthcare providers
6. **Core API** - All network endpoints implemented with FastAPI
7. **Blockchain Governance** - DAO with committee structure and dual-axis voting

### 🚧 In Progress

- Multi-omics processors (transcriptomics, epigenetics, proteomics)
- PIR server implementation
- Post-quantum cryptography migration
- UI/UX components

## Architecture

```
genomevault/
├── local_processing/      # Multi-omics collection & local preprocessing
├── hypervector_transform/ # HDC encoding & similarity-preserving mappings
├── zk_proofs/            # Zero-knowledge proof generation & verification
├── pir/                  # Private Information Retrieval network components
├── blockchain/           # Smart contracts & governance layer
│   └── hipaa/           # HIPAA fast-track verification system
├── api/                  # Core network API endpoints
├── clinical/             # Clinical applications (diabetes pilot)
├── advanced_analysis/    # Research modules & AI integration
├── examples/             # Usage examples and demos
├── tests/                # Comprehensive test suite
└── utils/                # Shared utilities and configuration
```

## Key Features

### Privacy Guarantees
- **Zero-Knowledge Proofs**: 384-byte proofs, <25ms verification
- **PIR Privacy**: P_fail(k,q) = (1-q)^k with configurable server trust
- **Differential Privacy**: ε=1.0 with adaptive noise calibration
- **No Raw Data Exposure**: All processing happens locally

### Performance Metrics
- **Compression**: 10,000:1 ratio for genomic data
- **Hypervector Operations**: <1ms for similarity calculations
- **Proof Generation**: 1-30s depending on complexity
- **PIR Queries**: ~210ms for 3-shard configuration

### Governance Model
- **Dual-Axis Voting**: w = c + s (hardware class + signatory status)
- **HIPAA Fast-Track**: Healthcare providers get s=10 weight
- **Credit System**: Block rewards = c + 2×[s>0]
- **Committee Structure**: Scientific, Ethics, Security, User committees

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+ (for smart contracts)
- Docker (optional, for containerized processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/rohanvinaik/GenomeVault.git
cd genomevault

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from genomevault.local_processing import SequencingProcessor
from genomevault.hypervector_transform import HierarchicalEncoder
from genomevault.zk_proofs import CircuitManager

# Process genomic data locally
processor = SequencingProcessor()
genomic_data = processor.process_vcf("path/to/genome.vcf")

# Generate privacy-preserving hypervector
encoder = HierarchicalEncoder()
hypervector = encoder.encode(genomic_data, compression_tier="clinical")

# Create zero-knowledge proof
circuit_manager = CircuitManager()
proof = circuit_manager.prove_variant_presence(
    variant_hash="hash_of_variant",
    hypervector=hypervector
)

# Verify proof (anyone can do this)
is_valid = circuit_manager.verify_proof(proof)
```

### HIPAA Provider Registration

```python
from genomevault.blockchain.hipaa import HIPAAVerifier, HIPAACredentials

# Initialize verifier
verifier = HIPAAVerifier()

# Submit credentials
credentials = HIPAACredentials(
    npi="1234567893",  # Your NPI
    baa_hash="sha256_of_baa",
    risk_analysis_hash="sha256_of_risk_analysis", 
    hsm_serial="HSM-12345"
)

# Get verified as Trusted Signatory
verification_id = await verifier.submit_verification(credentials)
record = await verifier.process_verification(verification_id)

# Now you have enhanced voting power!
```

## Examples

See the `examples/` directory for comprehensive demos:
- `basic_usage.py` - Simple workflow demonstration
- `demo_hypervector_encoding.py` - Hypervector operations
- `hipaa_fasttrack_demo.py` - Complete HIPAA registration flow
- `integration_example.py` - End-to-end integration

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests

# Run with coverage
pytest --cov=genomevault
```

## Documentation

- [HIPAA Fast-Track Guide](docs/HIPAA_FASTTRACK.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)
- [API Reference](docs/api/)
- [Architecture Overview](docs/architecture/)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with cutting-edge research in:
- Hyperdimensional Computing (Kanerva, 2009)
- Zero-Knowledge Proofs (PLONK, Gabizon et al., 2019)
- Information-Theoretic PIR (Chor et al., 1995)
- Differential Privacy (Dwork et al., 2006)

## Contact

- Repository: https://github.com/rohanvinaik/GenomeVault
- Issues: https://github.com/rohanvinaik/GenomeVault/issues
