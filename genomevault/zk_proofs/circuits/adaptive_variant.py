"""Adaptive variant presence circuit with size-optimized implementations."""

from typing import Dict, Any, List
import numpy as np
from abc import ABC, abstractmethod


class VariantCircuit(ABC):
    """Base class for variant presence circuits."""
    
    @abstractmethod
    def generate_witness(self, inputs: Dict) -> Dict:
        pass


class SmallVariantCircuit(VariantCircuit):
    """Optimized for small inputs (<50 variants)."""
    
    def __init__(self):
        # Pre-allocate small buffers
        self.buffer_size = 50
        self.variant_buffer = np.zeros((self.buffer_size, 4), dtype=np.int32)
    
    def generate_witness(self, inputs: Dict) -> Dict:
        """Fast witness generation for small inputs."""
        variants = inputs['variants']
        query = inputs['query']
        
        # Direct comparison without heavy setup
        found = False
        index = -1
        for i, var in enumerate(variants[:self.buffer_size]):
            if (var['chr'] == query['chr'] and 
                var['pos'] == query['pos'] and
                var['alt'] == query['alt']):
                found = True
                index = i
                break
        
        return {
            'found': found,
            'index': index,
            'num_variants': len(variants)
        }


class LargeVariantCircuit(VariantCircuit):
    """Optimized for large inputs (>=50 variants)."""
    
    def __init__(self):
        # Use hash table for O(1) lookup
        self.variant_index = {}
    
    def generate_witness(self, inputs: Dict) -> Dict:
        """Efficient witness generation for large inputs."""
        variants = inputs['variants']
        query = inputs['query']
        
        # Build index if needed
        if not self.variant_index:
            for i, var in enumerate(variants):
                key = f"{var['chr']}:{var['pos']}:{var['alt']}"
                self.variant_index[key] = i
        
        # Fast lookup
        query_key = f"{query['chr']}:{query['pos']}:{query['alt']}"
        found = query_key in self.variant_index
        
        return {
            'found': found,
            'index': self.variant_index.get(query_key, -1),
            'num_variants': len(variants),
            'used_index': True
        }


class AdaptiveVariantPresenceCircuit:
    """Adaptive circuit that selects optimal implementation based on input size."""
    
    def __init__(self):
        self.small_circuit = SmallVariantCircuit()
        self.large_circuit = LargeVariantCircuit()
        self.size_threshold = 50
        
        # Performance tracking
        self.performance_stats = {
            'small': {'count': 0, 'total_time': 0},
            'large': {'count': 0, 'total_time': 0}
        }
    
    def select_circuit(self, input_size: int) -> VariantCircuit:
        """Select optimal circuit based on input size."""
        if input_size < self.size_threshold:
            return self.small_circuit
        else:
            return self.large_circuit
    
    def generate_witness(self, inputs: Dict) -> Dict:
        """Generate witness using adaptive selection."""
        import time
        
        variants = inputs.get('variants', [])
        input_size = len(variants)
        
        # Select appropriate circuit
        circuit = self.select_circuit(input_size)
        circuit_type = 'small' if input_size < self.size_threshold else 'large'
        
        # Generate witness with timing
        start = time.perf_counter()
        witness = circuit.generate_witness(inputs)
        elapsed = time.perf_counter() - start
        
        # Track performance
        self.performance_stats[circuit_type]['count'] += 1
        self.performance_stats[circuit_type]['total_time'] += elapsed
        
        # Add metadata
        witness['circuit_type'] = circuit_type
        witness['generation_time_ms'] = elapsed * 1000
        
        return witness
    
    def auto_tune(self):
        """Automatically adjust threshold based on performance."""
        if self.performance_stats['small']['count'] > 100:
            small_avg = (self.performance_stats['small']['total_time'] / 
                        self.performance_stats['small']['count'])
        else:
            small_avg = None
        
        if self.performance_stats['large']['count'] > 100:
            large_avg = (self.performance_stats['large']['total_time'] / 
                        self.performance_stats['large']['count'])
        else:
            large_avg = None
        
        # Find optimal crossover point
        # This is simplified - in practice would use more sophisticated analysis
        if small_avg and large_avg:
            if small_avg > large_avg:
                self.size_threshold = max(25, self.size_threshold - 5)
            else:
                self.size_threshold = min(100, self.size_threshold + 5)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = {}
        
        for circuit_type in ['small', 'large']:
            count = self.performance_stats[circuit_type]['count']
            total_time = self.performance_stats[circuit_type]['total_time']
            
            if count > 0:
                avg_time = total_time / count * 1000  # Convert to ms
                stats[circuit_type] = {
                    'count': count,
                    'avg_time_ms': avg_time,
                    'total_time_s': total_time
                }
            else:
                stats[circuit_type] = {
                    'count': 0,
                    'avg_time_ms': 0,
                    'total_time_s': 0
                }
        
        stats['threshold'] = self.size_threshold
        return stats