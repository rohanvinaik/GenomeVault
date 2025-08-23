"""
Cache management for GenomeVault critical paths.

Implements caching strategies for:
- Hypervector transformations
- PIR query results  
- Database connections
- ZK proof verification results
"""

import hashlib
import json
import pickle
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

# Metrics
cache_hits = Counter(
    'genomevault_cache_hits_total',
    'Total number of cache hits',
    ['cache_type', 'operation']
)

cache_misses = Counter(
    'genomevault_cache_misses_total', 
    'Total number of cache misses',
    ['cache_type', 'operation']
)

cache_evictions = Counter(
    'genomevault_cache_evictions_total',
    'Total number of cache evictions',
    ['cache_type', 'reason']
)

cache_size = Gauge(
    'genomevault_cache_size_bytes',
    'Current cache size in bytes',
    ['cache_type']
)

cache_operation_duration = Histogram(
    'genomevault_cache_operation_seconds',
    'Cache operation duration',
    ['cache_type', 'operation'],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used  
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def access(self) -> None:
        """Record an access to this entry."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size_mb: int = 100, max_entries: int = 10000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        self.lock = RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
                
            entry = self.cache[key]
            if entry.is_expired():
                self._evict(key, reason="expired")
                return None
                
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.access()
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Put value in cache."""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Remove old entry if exists
            if key in self.cache:
                self._evict(key, reason="overwrite")
            
            # Evict entries if needed
            while (self.current_size_bytes + size_bytes > self.max_size_bytes or
                   len(self.cache) >= self.max_entries):
                if not self.cache:
                    break
                # Evict least recently used
                lru_key = next(iter(self.cache))
                self._evict(lru_key, reason="size_limit")
            
            # Add new entry
            now = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_at=now,
                last_accessed=now,
                ttl_seconds=ttl_seconds
            )
            self.cache[key] = entry
            self.current_size_bytes += size_bytes
    
    def _evict(self, key: str, reason: str) -> None:
        """Evict entry from cache."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size_bytes -= entry.size_bytes
            cache_evictions.labels(cache_type="lru", reason=reason).inc()
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, (str, bytes)):
            return len(value)
        else:
            # Fallback to pickle size
            try:
                return len(pickle.dumps(value))
            except:
                return 1000  # Default size
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "entries": len(self.cache),
                "size_bytes": self.current_size_bytes,
                "max_size_bytes": self.max_size_bytes,
                "utilization": self.current_size_bytes / self.max_size_bytes
            }


class HypervectorCache:
    """Specialized cache for hypervector transformations."""
    
    def __init__(self, max_size_mb: int = 500):
        self.cache = LRUCache(max_size_mb=max_size_mb)
        self.lock = Lock()
        
    def get_encoding(self, variant_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get cached hypervector encoding."""
        key = self._compute_key(variant_data)
        
        with cache_operation_duration.labels(
            cache_type="hypervector", operation="get"
        ).time():
            result = self.cache.get(key)
            
        if result is not None:
            cache_hits.labels(cache_type="hypervector", operation="encoding").inc()
        else:
            cache_misses.labels(cache_type="hypervector", operation="encoding").inc()
            
        return result
    
    def put_encoding(self, variant_data: Dict[str, Any], encoding: np.ndarray) -> None:
        """Cache hypervector encoding."""
        key = self._compute_key(variant_data)
        
        with cache_operation_duration.labels(
            cache_type="hypervector", operation="put"
        ).time():
            # Cache for 1 hour by default
            self.cache.put(key, encoding, ttl_seconds=3600)
            
        cache_size.labels(cache_type="hypervector").set(self.cache.current_size_bytes)
    
    def _compute_key(self, variant_data: Dict[str, Any]) -> str:
        """Compute cache key for variant data."""
        # Sort keys for consistent hashing
        sorted_data = json.dumps(variant_data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    def get_distance_matrix(self, vectors: List[np.ndarray]) -> Optional[np.ndarray]:
        """Get cached distance matrix."""
        # Create key from vector hashes
        vector_hashes = [hashlib.sha256(v.tobytes()).hexdigest()[:8] for v in vectors]
        key = f"dist_matrix:{':'.join(sorted(vector_hashes))}"
        
        result = self.cache.get(key)
        if result is not None:
            cache_hits.labels(cache_type="hypervector", operation="distance_matrix").inc()
        else:
            cache_misses.labels(cache_type="hypervector", operation="distance_matrix").inc()
            
        return result
    
    def put_distance_matrix(self, vectors: List[np.ndarray], matrix: np.ndarray) -> None:
        """Cache distance matrix."""
        vector_hashes = [hashlib.sha256(v.tobytes()).hexdigest()[:8] for v in vectors]
        key = f"dist_matrix:{':'.join(sorted(vector_hashes))}"
        
        # Cache for 30 minutes
        self.cache.put(key, matrix, ttl_seconds=1800)
        cache_size.labels(cache_type="hypervector").set(self.cache.current_size_bytes)


class ConnectionPool:
    """Database connection pooling."""
    
    def __init__(self, 
                 create_connection: Callable,
                 max_connections: int = 20,
                 min_connections: int = 5,
                 max_idle_time: int = 300):
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.max_idle_time = max_idle_time
        
        self.available_connections: List[Tuple[Any, float]] = []
        self.in_use_connections: Set[Any] = set()
        self.lock = Lock()
        
        # Pre-create minimum connections
        for _ in range(self.min_connections):
            conn = self.create_connection()
            self.available_connections.append((conn, time.time()))
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool."""
        conn = self._acquire()
        try:
            yield conn
        finally:
            self._release(conn)
    
    def _acquire(self) -> Any:
        """Acquire connection from pool."""
        with self.lock:
            # Remove expired connections
            now = time.time()
            self.available_connections = [
                (conn, last_used) for conn, last_used in self.available_connections
                if now - last_used < self.max_idle_time
            ]
            
            # Get available connection
            if self.available_connections:
                conn, _ = self.available_connections.pop(0)
                self.in_use_connections.add(conn)
                return conn
            
            # Create new connection if under limit
            if len(self.in_use_connections) < self.max_connections:
                conn = self.create_connection()
                self.in_use_connections.add(conn)
                return conn
            
            # Wait for connection to become available
            # In production, implement proper waiting with timeout
            raise RuntimeError("Connection pool exhausted")
    
    def _release(self, conn: Any) -> None:
        """Release connection back to pool."""
        with self.lock:
            if conn in self.in_use_connections:
                self.in_use_connections.remove(conn)
                self.available_connections.append((conn, time.time()))
    
    def close_all(self) -> None:
        """Close all connections."""
        with self.lock:
            for conn, _ in self.available_connections:
                try:
                    conn.close()
                except:
                    pass
            for conn in self.in_use_connections:
                try:
                    conn.close()
                except:
                    pass
            self.available_connections.clear()
            self.in_use_connections.clear()


class PIRQueryCache:
    """Cache for PIR query results."""
    
    def __init__(self, max_size_mb: int = 200):
        self.cache = LRUCache(max_size_mb=max_size_mb)
        
    def get_result(self, query_id: str, block_index: int) -> Optional[bytes]:
        """Get cached PIR query result."""
        key = f"pir:{query_id}:{block_index}"
        result = self.cache.get(key)
        
        if result is not None:
            cache_hits.labels(cache_type="pir", operation="query_result").inc()
        else:
            cache_misses.labels(cache_type="pir", operation="query_result").inc()
            
        return result
    
    def put_result(self, query_id: str, block_index: int, data: bytes) -> None:
        """Cache PIR query result."""
        key = f"pir:{query_id}:{block_index}"
        # Cache for 5 minutes (PIR results are ephemeral)
        self.cache.put(key, data, ttl_seconds=300)
        cache_size.labels(cache_type="pir").set(self.cache.current_size_bytes)


class ZKProofCache:
    """Cache for zero-knowledge proof verification results."""
    
    def __init__(self, max_size_mb: int = 100):
        self.cache = LRUCache(max_size_mb=max_size_mb)
        
    def get_verification(self, proof_hash: str) -> Optional[bool]:
        """Get cached proof verification result."""
        key = f"zk:verify:{proof_hash}"
        result = self.cache.get(key)
        
        if result is not None:
            cache_hits.labels(cache_type="zk", operation="verification").inc()
        else:
            cache_misses.labels(cache_type="zk", operation="verification").inc()
            
        return result
    
    def put_verification(self, proof_hash: str, is_valid: bool) -> None:
        """Cache proof verification result."""
        key = f"zk:verify:{proof_hash}"
        # Cache for 1 hour
        self.cache.put(key, is_valid, ttl_seconds=3600)
        cache_size.labels(cache_type="zk").set(self.cache.current_size_bytes)


class CacheManager:
    """Central cache management for GenomeVault."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.hypervector_cache = HypervectorCache(max_size_mb=500)
            self.pir_cache = PIRQueryCache(max_size_mb=200)
            self.zk_cache = ZKProofCache(max_size_mb=100)
            self.connection_pools: Dict[str, ConnectionPool] = {}
            self.initialized = True
    
    def get_connection_pool(self, 
                           pool_name: str,
                           create_connection: Callable,
                           max_connections: int = 20) -> ConnectionPool:
        """Get or create connection pool."""
        if pool_name not in self.connection_pools:
            self.connection_pools[pool_name] = ConnectionPool(
                create_connection=create_connection,
                max_connections=max_connections
            )
        return self.connection_pools[pool_name]
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.hypervector_cache.cache.clear()
        self.pir_cache.cache.clear()
        self.zk_cache.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            "hypervector": self.hypervector_cache.cache.stats(),
            "pir": self.pir_cache.cache.stats(),
            "zk": self.zk_cache.cache.stats(),
            "connection_pools": {
                name: {
                    "available": len(pool.available_connections),
                    "in_use": len(pool.in_use_connections)
                }
                for name, pool in self.connection_pools.items()
            }
        }


def cached_computation(cache_type: str = "generic", ttl_seconds: float = 3600):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(max_size_mb=50)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = hashlib.sha256(":".join(key_parts).encode()).hexdigest()
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                cache_hits.labels(cache_type=cache_type, operation=func.__name__).inc()
                return result
            
            # Compute and cache
            cache_misses.labels(cache_type=cache_type, operation=func.__name__).inc()
            result = func(*args, **kwargs)
            cache.put(key, result, ttl_seconds=ttl_seconds)
            
            return result
        
        wrapper._cache = cache
        return wrapper
    
    return decorator


# Global cache manager instance
cache_manager = CacheManager()