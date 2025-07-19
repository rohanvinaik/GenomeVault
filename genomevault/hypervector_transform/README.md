# Hypervector Transform Module

## Overview

The hypervector transform module implements hierarchical hyperdimensional computing (HDC) for transforming sensitive genomic and multi-omics data into privacy-preserving representations. This module is the core of GenomeVault's privacy guarantees, enabling complex biological analyses while ensuring that raw genomic data never leaves the user's control.

## Key Features

### 1. **Hypervector Encoding** (`encoding.py`)
- Transforms biological features into high-dimensional vectors (10,000+ dimensions)
- Multiple projection types: Gaussian, Sparse Random, Orthogonal
- Multi-resolution encoding (10k, 15k, 20k dimensions)
- Domain-specific encodings for different omics types
- Similarity preservation with privacy guarantees

### 2. **Binding Operations** (`binding.py`)
- Combine multiple hypervectors while preserving relationships
- Multiple binding types:
  - **Circular Convolution**: For reversible binding
  - **Element-wise Multiplication**: Fast binding with simple unbinding
  - **Permutation-based**: Position-aware binding
  - **XOR**: For binary hypervectors
  - **Fourier**: Frequency domain binding
- Specialized binders:
  - **PositionalBinder**: For genomic sequences
  - **CrossModalBinder**: For multi-omics integration

### 3. **Holographic Representations** (`holographic.py`)
- Encode complex structured data (variants, gene expression, interactions)
- Query specific components from composite representations
- Memory traces for storing multiple items
- Similarity-preserving hashing

### 4. **Similarity-Preserving Mappings** (`mapping.py`)
- Maintain biological relationships during transformation
- Domain-specific similarity metrics for different omics types
- Manifold preservation for complex data structures
- Optimization-based learning of mappings

## Installation

The hypervector module is part of the GenomeVault package:

```bash
pip install genomevault
```

For development:
```bash
pip install -e .[dev]
```

## Usage Examples

### Basic Hypervector Encoding

```python
from hypervector_transform.encoding import create_encoder
from core.constants import OmicsType

# Create an encoder
encoder = create_encoder(dimension=10000)

# Encode genomic data
genomic_data = {
    "variants": {
        "snps": [1, 2, 3, 4, 5],
        "indels": [10, 11],
        "cnvs": [20]
    },
    "quality_metrics": {
        "mean_coverage": 30.0,
        "uniformity": 0.95
    }
}

# Transform to hypervector
hypervector = encoder.encode(genomic_data, OmicsType.GENOMIC)
print(f"Hypervector shape: {hypervector.shape}")  # torch.Size([10000])
```

### Binding Operations

```python
from hypervector_transform.binding import HypervectorBinder, BindingType

# Create binder
binder = HypervectorBinder(dimension=10000)

# Create hypervectors for gene and expression
gene_hv = encoder.encode({"gene": "BRCA1"}, OmicsType.GENOMIC)
expression_hv = encoder.encode({"expression": 5.2}, OmicsType.TRANSCRIPTOMIC)

# Bind them together
bound_hv = binder.bind([gene_hv, expression_hv], BindingType.CIRCULAR)

# Later, recover the gene vector
recovered_gene = binder.unbind(bound_hv, [expression_hv], BindingType.CIRCULAR)
```

### Holographic Encoding

```python
from hypervector_transform.holographic import HolographicEncoder

# Create holographic encoder
holo_encoder = HolographicEncoder(dimension=10000)

# Encode a genomic variant
variant_hv = holo_encoder.encode_genomic_variant(
    chromosome="chr17",
    position=41276045,
    ref="C",
    alt="T",
    annotations={
        "gene": "BRCA1",
        "effect": "nonsense",
        "clinical_significance": "pathogenic"
    }
)

# Query for specific information
gene_info = holo_encoder.query(variant_hv, "gene")
```

### Cross-Modal Integration

```python
from hypervector_transform.binding import CrossModalBinder

# Create cross-modal binder
cross_binder = CrossModalBinder(dimension=10000)

# Bind multiple omics modalities
modalities = {
    "genomic": genomic_hypervector,
    "transcriptomic": transcriptomic_hypervector,
    "epigenomic": epigenomic_hypervector
}

# Create integrated representation
integrated = cross_binder.bind_modalities(modalities)
combined_hv = integrated["combined"]
```

## Technical Details

### Privacy Guarantees

The hypervector encoding provides mathematical privacy guarantees through:

1. **Irreversibility**: Random projection from low to high dimensions is not invertible
2. **Distributed Information**: Information is spread across all dimensions
3. **Noise Addition**: Optional differential privacy through controlled noise
4. **High Dimensionality**: Concentration of measure phenomenon protects individual features

### Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Encoding | O(d × D) | O(D) |
| Binding | O(D) or O(D log D) | O(D) |
| Similarity | O(D) | O(1) |
| Compression | O(D log k) | O(k) |

Where:
- d = input dimension
- D = hypervector dimension (typically 10,000)
- k = compressed size

### Similarity Preservation

The encoding preserves biological similarities:

```python
# Similar genomic profiles have high cosine similarity
similarity = encoder.similarity(hypervector1, hypervector2, metric="cosine")

# Different profiles have low similarity
similarity = encoder.similarity(hypervector1, hypervector3, metric="cosine")
```

## Architecture

```
hypervector_transform/
├── __init__.py           # Module initialization
├── encoding.py           # Core hypervector encoding
├── binding.py            # Binding operations
├── holographic.py        # Holographic representations
├── mapping.py            # Similarity-preserving mappings
└── compression.py        # Tiered compression (in local_processing)
```

## Testing

Run unit tests:
```bash
python -m pytest tests/unit/test_hypervector_encoding.py -v
```

Run demonstrations:
```bash
python examples/demo_hypervector_encoding.py
```

## Mathematical Foundations

### Hyperdimensional Computing

Hypervectors are high-dimensional random vectors with useful properties:

1. **Quasi-orthogonality**: Random hypervectors are nearly orthogonal
2. **Robustness**: Tolerant to noise and component failures
3. **Compositionality**: Complex structures through simple operations

### Key Operations

**Binding** (⊗): Combines two hypervectors
- Circular convolution: reversible, associative
- Element-wise multiplication: fast, commutative

**Bundling** (+): Superimposes multiple hypervectors
- Preserves similarity to all components
- Used for set representation

**Permutation** (ρ): Encodes order/position
- Shifts vector elements cyclically
- Creates unique representation for sequences

## Best Practices

1. **Dimension Selection**
   - Use 10,000D for general genomic data
   - Use 15,000D for complex multi-omics
   - Use 20,000D for population-scale analyses

2. **Compression Tiers**
   - Mini (25KB): Basic analyses, mobile apps
   - Clinical (300KB): Medical applications
   - Full (200KB/modality): Research use

3. **Binding Strategy**
   - Use circular convolution for reversible operations
   - Use multiplication for speed when unbinding not needed
   - Use XOR for binary data

4. **Privacy Considerations**
   - Never store projection matrices with data
   - Use different seeds for different users
   - Apply differential privacy for population queries

## Troubleshooting

### Common Issues

1. **Memory Usage**
   ```python
   # Use sparse projections for lower memory
   encoder = create_encoder(projection_type="sparse_random")
   ```

2. **Performance**
   ```python
   # Enable GPU acceleration if available
   if torch.cuda.is_available():
       hypervector = hypervector.cuda()
   ```

3. **Similarity Computation**
   ```python
   # Normalize vectors before similarity computation
   hv1_norm = hv1 / torch.norm(hv1)
   hv2_norm = hv2 / torch.norm(hv2)
   similarity = torch.dot(hv1_norm, hv2_norm)
   ```

## Contributing

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. Key areas for contribution:

- Additional binding operations
- Domain-specific encoders
- Performance optimizations
- Hardware acceleration

## References

1. Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors.
2. Plate, T. A. (2003). Holographic reduced representations.
3. Ge, L., & Parhi, K. K. (2020). Classification using hyperdimensional computing: A review.

## License

This module is part of GenomeVault and is licensed under the Apache License 2.0.
