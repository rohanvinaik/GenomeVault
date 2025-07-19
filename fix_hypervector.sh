#!/bin/bash
# Fix hypervector imports

echo "🔧 Fixing hypervector imports..."

cd /Users/rohanvinaik/genomevault

# Add the missing functions to binding.py
cat >> hypervector_transform/binding.py << 'EOF'

# Additional convenience functions for compatibility
def bind_vectors(vectors: List[torch.Tensor], 
                binding_type: BindingType = BindingType.CIRCULAR) -> torch.Tensor:
    """Bind multiple vectors together"""
    if not vectors:
        raise ValueError("No vectors provided")
    binder = HypervectorBinder(vectors[0].shape[-1])
    return binder.bind(vectors, binding_type)


def unbind_vectors(bound_vector: torch.Tensor,
                  known_vectors: List[torch.Tensor],
                  binding_type: BindingType = BindingType.CIRCULAR) -> torch.Tensor:
    """Unbind vectors"""
    binder = HypervectorBinder(bound_vector.shape[-1])
    return binder.unbind(bound_vector, known_vectors, binding_type)


def bundle_vectors(vectors: List[torch.Tensor], normalize: bool = True) -> torch.Tensor:
    """Bundle vectors using superposition"""
    if not vectors:
        raise ValueError("No vectors provided")
    binder = HypervectorBinder(vectors[0].shape[-1])
    return binder.bundle(vectors, normalize)


def permute_vector(vector: torch.Tensor, positions: int = 1) -> torch.Tensor:
    """Permute a vector by rotating positions"""
    if positions == 0:
        return vector
    return torch.roll(vector, shifts=positions, dims=-1)


def circular_convolution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Perform circular convolution"""
    binder = HypervectorBinder(x.shape[-1])
    return binder._circular_convolve(x, y)


def cross_modal_binding(modality_vectors: Dict[str, torch.Tensor],
                       preserve_individual: bool = True) -> Dict[str, torch.Tensor]:
    """Bind multiple modalities together"""
    if not modality_vectors:
        raise ValueError("No modality vectors provided")
    
    # Get dimension from first vector
    first_vector = next(iter(modality_vectors.values()))
    binder = CrossModalBinder(first_vector.shape[-1])
    return binder.bind_modalities(modality_vectors, preserve_individual)


# Import hashlib for cross-modal binding
import hashlib
EOF

echo "✅ Added missing functions to binding.py"

# Now run the tests
echo ""
echo "🧪 Running tests..."
python3 minimal_test.py
