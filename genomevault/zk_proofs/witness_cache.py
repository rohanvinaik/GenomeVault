"""Witness generation caching system with LRU eviction."""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, List, Callable
from threading import RLock
import pickle


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update and move to end
                self.cache.move_to_end(key)
            self.cache[key] = value
            
            # Evict oldest if needed
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': max(0, self.misses - self.max_size)
            }


class WitnessCache:
    """Caching system for ZK witness generation."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = LRUCache(max_size)
        self.ttl_seconds = ttl_seconds
        self.computation_times = {}
        
    def _generate_cache_key(
        self, 
        circuit_name: str, 
        inputs: Dict[str, Any]
    ) -> str:
        """Generate stable cache key for circuit and inputs."""
        # Serialize inputs deterministically
        inputs_str = json.dumps(inputs, sort_keys=True, default=str)
        
        # Create hash
        hasher = hashlib.sha256()
        hasher.update(circuit_name.encode())
        hasher.update(inputs_str.encode())
        
        return hasher.hexdigest()
    
    def get_or_compute(
        self,
        circuit_name: str,
        inputs: Dict[str, Any],
        compute_fn: Callable
    ) -> Tuple[Dict[str, Any], bool]:
        """Get cached witness or compute if not found."""
        
        # Generate cache key
        cache_key = self._generate_cache_key(circuit_name, inputs)
        
        # Check cache
        cached_entry = self.cache.get(cache_key)
        
        if cached_entry is not None:
            # Check TTL
            if time.time() - cached_entry['timestamp'] < self.ttl_seconds:
                # Cache hit
                return cached_entry['witness'], True
            else:
                # Expired - will recompute
                pass
        
        # Cache miss - compute witness
        start_time = time.perf_counter()
        witness = compute_fn(circuit_name, inputs)
        computation_time = time.perf_counter() - start_time
        
        # Store in cache
        cache_entry = {
            'witness': witness,
            'timestamp': time.time(),
            'computation_time': computation_time
        }
        self.cache.put(cache_key, cache_entry)
        
        # Track computation time
        if circuit_name not in self.computation_times:
            self.computation_times[circuit_name] = []
        self.computation_times[circuit_name].append(computation_time)
        
        return witness, False
    
    def invalidate_circuit(self, circuit_name: str) -> None:
        """Invalidate all cached witnesses for a circuit."""
        keys_to_remove = []
        
        # Find all keys for this circuit
        # In production, maintain reverse index for efficiency
        for key in list(self.cache.cache.keys()):
            # This is simplified - would need proper key structure
            # For now, we can't efficiently determine circuit from hash
            # In production, would maintain circuit -> keys mapping
            pass
        
        # For now, clear entire cache if circuit invalidation requested
        # In production, would have better key structure
        self.cache.clear()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        cache_stats = self.cache.get_stats()
        
        # Calculate average computation times
        avg_times = {}
        for circuit, times in self.computation_times.items():
            if times:
                avg_times[circuit] = {
                    'avg_ms': sum(times) / len(times) * 1000,
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000,
                    'count': len(times)
                }
        
        return {
            'cache': cache_stats,
            'computation_times': avg_times,
            'estimated_time_saved_ms': cache_stats['hits'] * 1.3  # Avg 1.3ms
        }
    
    def warm_cache(
        self, 
        common_patterns: List[Tuple[str, Dict]]
    ) -> None:
        """Pre-warm cache with common patterns."""
        # Import here to avoid circular dependency
        from genomevault.zk_proofs.prover import Prover
        prover = Prover()
        
        warmed = 0
        for circuit_name, inputs in common_patterns:
            try:
                self.get_or_compute(
                    circuit_name,
                    inputs,
                    lambda c, i: prover._generate_witness_direct(c, i)
                )
                warmed += 1
            except Exception as e:
                # Log but continue warming
                print(f"Failed to warm cache for {circuit_name}: {e}")
        
        return warmed


# Singleton instance
_witness_cache = None


def get_witness_cache() -> WitnessCache:
    """Get global witness cache instance."""
    global _witness_cache
    if _witness_cache is None:
        _witness_cache = WitnessCache()
    return _witness_cache


def reset_witness_cache() -> None:
    """Reset the global witness cache."""
    global _witness_cache
    if _witness_cache is not None:
        _witness_cache.cache.clear()
    _witness_cache = WitnessCache()