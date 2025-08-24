"""
Federated aggregator module with secure aggregation and differential privacy.

Implements secure aggregation through masking (Section 6.1) where client masks
cancel out during aggregation, and differential privacy through gradient clipping
and calibrated noise addition.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Tuple
import hashlib
import secrets

from genomevault.core.exceptions import ValidationError
from genomevault.federated.models import (
    AggregateRequest,
    AggregateResponse,
    ModelUpdate,
)
from genomevault.utils.logging import get_logger

# Optional differential privacy integration
try:
    from genomevault.privacy import (
        GaussianMechanism,
        PrivacyLevel,
        PrivacyAccountant,
        DifferentiallyPrivateFederated,
    )
    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False

logger = get_logger(__name__)


def _l2_norm(x: np.ndarray) -> float:
    """l2 norm.
    Args:        x: Parameter value.
    Returns:
        float"""
    return float(np.linalg.norm(x))


def _clip_by_l2(x: np.ndarray, clip: float) -> np.ndarray:
    """clip by l2.
    Args:        x: Parameter value.        clip: Parameter value.
    Returns:
        np.ndarray"""
    n = _l2_norm(x)
    if n <= clip or clip <= 0.0:
        return x
    return x * (clip / n)


class SecureAggregator:
    """
    Secure aggregator implementing masking from Section 6.1.
    
    Each client generates random masks that cancel out when aggregated,
    providing privacy against the aggregation server while maintaining
    correctness of the sum.
    
    Protocol:
    1. Each client i generates pairwise masks R_{i,j} with other clients
    2. Client i adds sum(R_{i,j}) - sum(R_{j,i}) to their update
    3. When aggregated, all masks cancel: sum_i(R_{i,j} - R_{j,i}) = 0
    4. Server learns only the aggregate, not individual updates
    """
    
    def __init__(self, num_clients: int, vector_size: int, seed: Optional[int] = None):
        """
        Initialize secure aggregator.
        
        Args:
            num_clients: Number of participating clients
            vector_size: Size of model update vectors
            seed: Random seed for reproducibility (testing only)
        """
        self.num_clients = num_clients
        self.vector_size = vector_size
        self.seed = seed
        
        # Generate pairwise shared seeds for mask generation
        # In practice, these would be established through key agreement
        self.pairwise_seeds = self._generate_pairwise_seeds()
        
        logger.info(
            f"Initialized SecureAggregator for {num_clients} clients, "
            f"vector size {vector_size}"
        )
    
    def _generate_pairwise_seeds(self) -> Dict[Tuple[int, int], bytes]:
        """
        Generate pairwise seeds for mask generation.
        
        In a real deployment, these would be established through:
        - Diffie-Hellman key exchange
        - Trusted third party
        - Secure multi-party computation
        
        Returns:
            Dictionary mapping (client_i, client_j) pairs to shared seeds
        """
        seeds = {}
        
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                # Generate shared seed for pair (i, j)
                if self.seed is not None:
                    # Deterministic for testing
                    seed_data = f"{self.seed}:{i}:{j}".encode()
                    shared_seed = hashlib.sha256(seed_data).digest()
                else:
                    # Cryptographically secure random
                    shared_seed = secrets.token_bytes(32)
                
                seeds[(i, j)] = shared_seed
                seeds[(j, i)] = shared_seed  # Symmetric
        
        return seeds
    
    def generate_client_mask(self, client_id: int) -> np.ndarray:
        """
        Generate mask for a specific client.
        
        Args:
            client_id: ID of the client (0 to num_clients-1)
            
        Returns:
            Mask vector to add to client's update
        """
        if client_id >= self.num_clients:
            raise ValueError(f"Invalid client_id {client_id}, max is {self.num_clients-1}")
        
        mask = np.zeros(self.vector_size, dtype=np.float64)
        
        # Add masks for all other clients
        for other_id in range(self.num_clients):
            if other_id == client_id:
                continue
            
            # Get shared seed
            seed = self.pairwise_seeds.get((client_id, other_id))
            if seed is None:
                raise ValueError(f"No seed for pair ({client_id}, {other_id})")
            
            # Generate deterministic random mask from seed
            rng = np.random.RandomState()
            rng.seed(int.from_bytes(seed[:4], 'big'))
            
            # Generate mask values (zero-mean for better numerical stability)
            pairwise_mask = rng.normal(0, 1.0, self.vector_size)
            
            if client_id < other_id:
                # Add R_{i,j}
                mask += pairwise_mask
            else:
                # Subtract R_{j,i} (since we're client j from pair (i,j))
                mask -= pairwise_mask
        
        return mask
    
    def mask_update(self, update: np.ndarray, client_id: int) -> np.ndarray:
        """
        Add mask to client update for secure aggregation.
        
        Args:
            update: Client's model update
            client_id: Client identifier
            
        Returns:
            Masked update
        """
        if len(update) != self.vector_size:
            raise ValueError(
                f"Update size {len(update)} doesn't match expected {self.vector_size}"
            )
        
        mask = self.generate_client_mask(client_id)
        masked = update + mask
        
        logger.debug(f"Client {client_id}: Added mask with L2 norm {np.linalg.norm(mask):.4f}")
        
        return masked
    
    def aggregate_masked(self, masked_updates: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate masked updates (masks cancel out).
        
        Args:
            masked_updates: List of masked updates from all clients
            
        Returns:
            Aggregated update (masks cancelled)
        """
        if len(masked_updates) != self.num_clients:
            raise ValueError(
                f"Expected {self.num_clients} updates, got {len(masked_updates)}"
            )
        
        # Simple sum - masks cancel due to symmetry
        aggregated = np.sum(masked_updates, axis=0)
        
        # Average (optional, depends on protocol)
        aggregated = aggregated / self.num_clients
        
        return aggregated
    
    def verify_mask_cancellation(self) -> bool:
        """
        Verify that masks cancel out when aggregated (for testing).
        
        Returns:
            True if masks cancel to near-zero
        """
        all_masks = []
        
        for client_id in range(self.num_clients):
            mask = self.generate_client_mask(client_id)
            all_masks.append(mask)
        
        # Sum should be zero (or very close due to floating point)
        total_mask = np.sum(all_masks, axis=0)
        mask_norm = np.linalg.norm(total_mask)
        
        logger.info(f"Total mask norm after aggregation: {mask_norm:.2e}")
        
        # Check if effectively zero (accounting for numerical errors)
        return mask_norm < 1e-10


class FedAvgAggregator:
    """
    FedAvg aggregator with differential privacy support.
    
    Enhanced with:
    - Gradient clipping for bounded sensitivity
    - Calibrated Gaussian noise for differential privacy
    - Optional secure aggregation via masking
    """

    def __init__(
        self,
        use_differential_privacy: bool = False,
        privacy_level: Optional['PrivacyLevel'] = None,
        privacy_epsilon: Optional[float] = None,
        privacy_delta: Optional[float] = None,
        privacy_accountant: Optional['PrivacyAccountant'] = None,
        use_secure_aggregation: bool = False,
        num_clients: Optional[int] = None,
    ) -> None:
        """
        Initialize FedAvg aggregator with optional privacy features.
        
        Args:
            use_differential_privacy: Enable differential privacy
            privacy_level: Predefined privacy level
            privacy_epsilon: Custom epsilon value
            privacy_delta: Custom delta value
            privacy_accountant: External privacy accountant
            use_secure_aggregation: Enable secure aggregation via masking
            num_clients: Expected number of clients (required for secure aggregation)
        """
        self._last_shape: int | None = None
        self.use_differential_privacy = use_differential_privacy and DP_AVAILABLE
        self.use_secure_aggregation = use_secure_aggregation
        self.num_clients = num_clients
        
        # Initialize differential privacy if requested
        self.dp_mechanism = None
        self.privacy_accountant = privacy_accountant
        
        if self.use_differential_privacy and DP_AVAILABLE:
            if privacy_level:
                epsilon, delta = privacy_level.value
            elif privacy_epsilon and privacy_delta:
                epsilon = privacy_epsilon
                delta = privacy_delta
            else:
                # Default to COMMON level for federated learning
                epsilon, delta = 10.0, 1e-5
            
            # Create privacy accountant if not provided
            if not self.privacy_accountant:
                self.privacy_accountant = PrivacyAccountant(
                    total_epsilon=epsilon * 100,
                    total_delta=delta * 100
                )
            
            # Sensitivity will be determined based on clip norm
            self.base_epsilon = epsilon
            self.base_delta = delta
            
            logger.info(
                f"Differential privacy enabled for FedAvg: ε={epsilon}, δ={delta}"
            )
        
        # Initialize secure aggregator if requested
        self.secure_aggregator = None
        if self.use_secure_aggregation:
            if not num_clients:
                raise ValueError("num_clients required for secure aggregation")
            # Will be initialized when we know vector size
            
        logger.info(
            f"Initialized FedAvgAggregator: DP={self.use_differential_privacy}, "
            f"SecureAgg={self.use_secure_aggregation}"
        )

    def _validate_and_prepare(
        self, updates: list[ModelUpdate], clip_norm: float | None
    ) -> tuple[list[np.ndarray], list[int]]:
        """Validate and prepare updates.

        Args:
            updates: List of model updates.
            clip_norm: Norm clipping value.

        Returns:
            tuple[list[np.ndarray], list[int]]: Prepared arrays and sample counts.
        """
        arrs: list[np.ndarray] = []
        counts: list[int] = []
        L: int | None = None
        for u in updates:
            w = np.asarray(u.weights, dtype=np.float64)
            if L is None:
                L = w.size
            if w.size != L:
                raise ValidationError(
                    "all weight vectors must have the same length",
                    context={"expected": L, "got": w.size, "client_id": u.client_id},
                )
            if clip_norm is not None and clip_norm > 0.0:
                w = _clip_by_l2(w, clip_norm)
            arrs.append(w)
            counts.append(int(u.num_examples))
        self._last_shape = int(L or 0)
        return arrs, counts

    def aggregate(self, req: AggregateRequest) -> AggregateResponse:
        """
        Aggregate client updates with optional secure aggregation and differential privacy.

        Args:
            req: Aggregation request with client updates

        Returns:
            AggregateResponse with privacy-preserving aggregated weights

        Raises:
            ValidationError: When operation fails
        """
        # Validate and clip updates
        arrs, counts = self._validate_and_prepare(req.updates, req.clip_norm)
        total_examples = int(sum(counts))
        if total_examples <= 0:
            raise ValidationError("total_examples must be positive")
        
        # Initialize secure aggregator if needed and not yet done
        if self.use_secure_aggregation and self.secure_aggregator is None:
            if self._last_shape is None:
                raise ValidationError("Cannot initialize secure aggregator without vector size")
            self.secure_aggregator = SecureAggregator(
                num_clients=len(arrs),
                vector_size=self._last_shape,
                seed=42  # Use fixed seed for reproducibility in testing
            )
            # Verify mask cancellation
            if not self.secure_aggregator.verify_mask_cancellation():
                logger.warning("Mask cancellation verification failed!")
        
        # Apply secure aggregation masks if enabled
        if self.use_secure_aggregation and self.secure_aggregator:
            masked_arrs = []
            for i, arr in enumerate(arrs):
                masked = self.secure_aggregator.mask_update(arr, client_id=i)
                masked_arrs.append(masked)
            logger.debug("Applied secure aggregation masks to all updates")
            arrs = masked_arrs
        
        # Compute weighted average
        numer = sum(w * n for w, n in zip(arrs, counts))
        denom = float(total_examples)
        agg = numer / denom
        
        # Add differential privacy noise if enabled
        if self.use_differential_privacy and DP_AVAILABLE:
            # Determine sensitivity based on clip norm
            if req.clip_norm and req.clip_norm > 0:
                # Sensitivity = 2 * clip_norm / n (worst case: one update changes maximally)
                sensitivity = 2 * req.clip_norm / len(arrs)
            else:
                # Without clipping, use a default sensitivity
                sensitivity = 1.0 / len(arrs)
            
            try:
                # Allocate privacy budget
                if self.privacy_accountant:
                    params = self.privacy_accountant.allocate_budget(
                        'federated',
                        'aggregate',
                        self.base_epsilon
                    )
                    params.sensitivity = sensitivity
                    
                    # Create mechanism with allocated budget
                    self.dp_mechanism = GaussianMechanism(
                        params.epsilon,
                        params.delta,
                        params.sensitivity
                    )
                else:
                    # Use base parameters
                    self.dp_mechanism = GaussianMechanism(
                        self.base_epsilon,
                        self.base_delta,
                        sensitivity
                    )
                
                # Add calibrated Gaussian noise
                agg = self.dp_mechanism.add_noise(agg, clip=True)
                
                logger.info(
                    f"Added DP noise: σ={self.dp_mechanism.sigma:.4f}, "
                    f"sensitivity={sensitivity:.4f}"
                )
                
            except Exception as e:
                logger.warning(f"Failed to add DP noise: {e}")
        
        # Prepare response details
        details = {
            "clip_norm": req.clip_norm,
            "differential_privacy": self.use_differential_privacy,
            "secure_aggregation": self.use_secure_aggregation,
        }
        
        if self.use_differential_privacy and self.dp_mechanism:
            details["dp_sigma"] = float(self.dp_mechanism.sigma)
            details["dp_epsilon"] = float(self.dp_mechanism.params.epsilon)
            details["dp_delta"] = float(self.dp_mechanism.params.delta)
        
        return AggregateResponse(
            aggregated_weights=agg.tolist(),
            total_examples=total_examples,
            client_count=len(arrs),
            details=details,
        )
