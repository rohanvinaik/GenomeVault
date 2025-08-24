"""Utility functions for handling both numpy arrays and torch tensors."""

import numpy as np
import torch
from typing import Union, Tuple


def get_tensor_size(tensor: Union[np.ndarray, torch.Tensor]) -> int:
    """
    Get the total number of elements in a tensor or array.
    
    Args:
        tensor: A numpy array or torch tensor
        
    Returns:
        Total number of elements
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.numel()  # torch method for total elements
    elif isinstance(tensor, np.ndarray):
        return tensor.size  # numpy property
    else:
        # Fallback for other types
        shape = getattr(tensor, 'shape', (len(tensor),))
        return int(np.prod(shape))


def to_numpy(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert a tensor to numpy array.
    
    Args:
        tensor: A numpy array or torch tensor
        
    Returns:
        Numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        return np.array(tensor)


def calculate_sparsity(tensor: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate sparsity (proportion of zero elements) for a tensor.
    
    Args:
        tensor: A numpy array or torch tensor
        
    Returns:
        Sparsity value between 0 and 1
    """
    # Handle sparse matrices
    if hasattr(tensor, 'todense'):
        tensor = tensor.todense()
    
    # Get total elements and convert to numpy
    total_elements = get_tensor_size(tensor)
    dense_array = to_numpy(tensor)
    
    # Calculate sparsity
    non_zero = np.count_nonzero(dense_array)
    sparsity = 1 - (non_zero / total_elements)
    
    return sparsity


def tensor_stats(tensor: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    Get comprehensive statistics for a tensor.
    
    Args:
        tensor: A numpy array or torch tensor
        
    Returns:
        Dictionary with tensor statistics
    """
    # Handle sparse matrices
    if hasattr(tensor, 'todense'):
        tensor = tensor.todense()
    
    # Convert to numpy for statistics
    arr = to_numpy(tensor)
    
    return {
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'size': get_tensor_size(tensor),
        'sparsity': calculate_sparsity(tensor),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'non_zero': int(np.count_nonzero(arr))
    }