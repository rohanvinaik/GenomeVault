"""Secure hypervector encoder with per-session randomization and rate limiting."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging
import secrets

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ortho_group

from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
from genomevault.core.constants import OmicsType

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    max_queries_per_day: int = 1000
    max_queries_per_hour: int = 100
    max_queries_per_minute: int = 10
    enable_strict_mode: bool = True
    alert_threshold_percentage: float = 0.8  # Alert at 80% of limit


@dataclass
class SessionConfig:
    """Configuration for secure session management."""
    
    session_id: str = field(default_factory=lambda: secrets.token_hex(32))
    rotation_interval_seconds: int = 3600  # 1 hour default
    enable_randomization: bool = True
    noise_sigma: float = 0.001  # Small dithering noise
    seed: Optional[int] = None


class RateLimiter:
    """Token bucket rate limiter with multiple time windows."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.query_log: Dict[str, list] = {}
        self.alert_triggered: Dict[str, bool] = {}
    
    def check_and_update(self, client_id: str) -> Tuple[bool, Optional[str]]:
        """Check if client can make query and update counters.
        
        Returns:
            (allowed, reason) - allowed is True if query permitted
        """
        current_time = time.time()
        
        # Initialize client if new
        if client_id not in self.query_log:
            self.query_log[client_id] = []
            self.alert_triggered[client_id] = False
        
        # Clean old entries
        self.query_log[client_id] = [
            t for t in self.query_log[client_id] 
            if current_time - t < 86400  # Keep 24 hours
        ]
        
        # Check limits
        queries = self.query_log[client_id]
        
        # Minute limit
        recent_minute = [t for t in queries if current_time - t < 60]
        if len(recent_minute) >= self.config.max_queries_per_minute:
            return False, f"Rate limit exceeded: {self.config.max_queries_per_minute}/minute"
        
        # Hour limit
        recent_hour = [t for t in queries if current_time - t < 3600]
        if len(recent_hour) >= self.config.max_queries_per_hour:
            return False, f"Rate limit exceeded: {self.config.max_queries_per_hour}/hour"
        
        # Day limit
        if len(queries) >= self.config.max_queries_per_day:
            return False, f"Rate limit exceeded: {self.config.max_queries_per_day}/day"
        
        # Check alert thresholds
        if not self.alert_triggered[client_id]:
            if len(queries) >= self.config.max_queries_per_day * self.config.alert_threshold_percentage:
                logger.warning(f"Client {client_id} approaching daily limit: {len(queries)}/{self.config.max_queries_per_day}")
                self.alert_triggered[client_id] = True
        
        # Update log
        self.query_log[client_id].append(current_time)
        
        return True, None
    
    def get_usage_stats(self, client_id: str) -> Dict[str, int]:
        """Get current usage statistics for a client."""
        if client_id not in self.query_log:
            return {"minute": 0, "hour": 0, "day": 0}
        
        current_time = time.time()
        queries = self.query_log[client_id]
        
        return {
            "minute": len([t for t in queries if current_time - t < 60]),
            "hour": len([t for t in queries if current_time - t < 3600]),
            "day": len(queries),
        }


class SecureHypervectorEncoder:
    """Secure hypervector encoder with per-session randomization and rate limiting.
    
    Implements H̃(x) = sign(RPx + τ) where:
    - R is a per-session orthogonal matrix
    - P is the base projection matrix
    - τ is small dithering noise
    """
    
    def __init__(
        self,
        base_config: Optional[HypervectorConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        session_config: Optional[SessionConfig] = None,
    ):
        """Initialize secure encoder.
        
        Args:
            base_config: Configuration for base hypervector encoder
            rate_limit_config: Rate limiting configuration
            session_config: Session management configuration
        """
        self.base_encoder = HypervectorEncoder(base_config or HypervectorConfig())
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.session_config = session_config or SessionConfig()
        
        # Initialize session-specific randomization
        self._init_session_randomization()
        
        # Audit log
        self.audit_log: list = []
        
        logger.info(f"Initialized secure encoder with session {self.session_config.session_id[:8]}...")
    
    def _init_session_randomization(self):
        """Initialize per-session randomization matrix R."""
        if not self.session_config.enable_randomization:
            self.R = None
            return
        
        # Set seed for reproducibility if provided
        if self.session_config.seed is not None:
            np.random.seed(self.session_config.seed)
        
        # Generate random orthogonal matrix R
        dim = self.base_encoder.config.dimension
        self.R = ortho_group.rvs(dim)
        
        # Store creation time for rotation
        self.session_created = time.time()
        
        logger.debug(f"Generated {dim}x{dim} orthogonal matrix for session randomization")
    
    def _should_rotate_session(self) -> bool:
        """Check if session should be rotated."""
        if not self.session_config.enable_randomization:
            return False
        
        elapsed = time.time() - self.session_created
        return elapsed > self.session_config.rotation_interval_seconds
    
    def _add_dithering_noise(self, vector: NDArray[np.float32]) -> NDArray[np.float32]:
        """Add small dithering noise τ to vector."""
        if self.session_config.noise_sigma <= 0:
            return vector
        
        noise = np.random.normal(0, self.session_config.noise_sigma, vector.shape)
        return vector + noise.astype(np.float32)
    
    def encode_secure(
        self,
        data: NDArray[np.float32],
        omics_type: OmicsType,
        client_id: str,
    ) -> Tuple[NDArray[np.float32], Dict[str, any]]:
        """Securely encode data with rate limiting and randomization.
        
        Implements H̃(x) = sign(RPx + τ)
        
        Args:
            data: Input genomic data
            omics_type: Type of omics data
            client_id: Client identifier for rate limiting
            
        Returns:
            (encoded_vector, metadata) - metadata includes security info
        """
        # Check rate limits
        allowed, reason = self.rate_limiter.check_and_update(client_id)
        if not allowed:
            logger.warning(f"Rate limit exceeded for client {client_id}: {reason}")
            raise PermissionError(f"Rate limit exceeded: {reason}")
        
        # Check if session should rotate
        if self._should_rotate_session():
            logger.info("Rotating session randomization matrix")
            self._init_session_randomization()
        
        # Base encoding: Px
        start_time = time.time()
        base_encoded = self.base_encoder.encode(data, omics_type)
        
        # Convert to float for operations
        if hasattr(base_encoded, 'numpy'):
            base_encoded = base_encoded.numpy()
        base_encoded = base_encoded.astype(np.float32)
        
        # Apply session randomization: RPx
        if self.R is not None:
            randomized = self.R @ base_encoded
        else:
            randomized = base_encoded
        
        # Add dithering noise: RPx + τ
        noisy = self._add_dithering_noise(randomized)
        
        # Final sign: sign(RPx + τ)
        secure_encoded = np.sign(noisy)
        
        # Audit logging
        encoding_time = time.time() - start_time
        audit_entry = {
            "timestamp": time.time(),
            "client_id": client_id,
            "session_id": self.session_config.session_id[:8],
            "omics_type": omics_type.value,
            "data_shape": data.shape,
            "encoding_time_ms": encoding_time * 1000,
            "randomization_applied": self.R is not None,
            "noise_applied": self.session_config.noise_sigma > 0,
        }
        self.audit_log.append(audit_entry)
        
        # Metadata for client
        metadata = {
            "session_id": self.session_config.session_id[:8],
            "encoding_time_ms": encoding_time * 1000,
            "usage_stats": self.rate_limiter.get_usage_stats(client_id),
            "randomization_active": self.R is not None,
        }
        
        return secure_encoded, metadata
    
    def verify_security_posture(self) -> Dict[str, any]:
        """Verify current security configuration and status."""
        return {
            "rate_limiting": {
                "enabled": self.rate_limiter.config.enable_strict_mode,
                "limits": {
                    "per_minute": self.rate_limiter.config.max_queries_per_minute,
                    "per_hour": self.rate_limiter.config.max_queries_per_hour,
                    "per_day": self.rate_limiter.config.max_queries_per_day,
                },
                "active_clients": len(self.rate_limiter.query_log),
            },
            "randomization": {
                "enabled": self.session_config.enable_randomization,
                "session_age_seconds": time.time() - self.session_created if hasattr(self, 'session_created') else 0,
                "rotation_interval": self.session_config.rotation_interval_seconds,
                "noise_sigma": self.session_config.noise_sigma,
            },
            "audit": {
                "total_queries": len(self.audit_log),
                "last_query": self.audit_log[-1] if self.audit_log else None,
            },
        }


# Default production configuration
DEFAULT_PRODUCTION_CONFIG = {
    "rate_limit": RateLimitConfig(
        max_queries_per_day=1000,
        max_queries_per_hour=100,
        max_queries_per_minute=10,
        enable_strict_mode=True,
        alert_threshold_percentage=0.8,
    ),
    "session": SessionConfig(
        rotation_interval_seconds=3600,  # 1 hour
        enable_randomization=True,
        noise_sigma=0.001,
    ),
    "hypervector": HypervectorConfig(
        dimension=8192,
    ),
}


def create_production_encoder() -> SecureHypervectorEncoder:
    """Create a production-ready secure encoder with default settings."""
    return SecureHypervectorEncoder(
        base_config=DEFAULT_PRODUCTION_CONFIG["hypervector"],
        rate_limit_config=DEFAULT_PRODUCTION_CONFIG["rate_limit"],
        session_config=DEFAULT_PRODUCTION_CONFIG["session"],
    )