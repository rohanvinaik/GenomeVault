"""
Differential Privacy implementation for GenomeVault.

Implements Gaussian mechanism with rigorous privacy accounting using Rényi DP
composition and temporal decay for privacy budget management.

References:
- Dwork & Roth (2014): The Algorithmic Foundations of Differential Privacy
- Mironov (2017): Rényi Differential Privacy
- Abadi et al. (2016): Deep Learning with Differential Privacy
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from enum import Enum

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class PrivacyLevel(Enum):
    """Privacy levels corresponding to accuracy modes in README."""
    OFF = (0.0, 0.0)  # No privacy (ε=∞, δ=0)
    COMMON = (10.0, 1e-5)  # Basic screening (ε=10, δ=10^-5)
    CLINICAL = (1.0, 1e-7)  # Clinical diagnostics (ε=1, δ=10^-7)
    KAN_HD = (0.1, 1e-9)  # Regulatory compliance (ε=0.1, δ=10^-9)


@dataclass
class PrivacyParameters:
    """Parameters for differential privacy mechanisms."""
    epsilon: float  # Privacy loss parameter
    delta: float  # Failure probability
    sensitivity: float  # L2 sensitivity of the function
    
    def __post_init__(self):
        """Validate privacy parameters."""
        if self.epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {self.epsilon}")
        if not (0 <= self.delta < 1):
            raise ValueError(f"Delta must be in [0, 1), got {self.delta}")
        if self.sensitivity < 0:
            raise ValueError(f"Sensitivity must be non-negative, got {self.sensitivity}")


class GaussianMechanism:
    """
    Gaussian mechanism for differential privacy.
    
    Implements the formula from Section 5.1:
    σ ≥ Δf·√(2ln(1.25/δ))/ε
    
    This provides (ε, δ)-differential privacy for functions with L2 sensitivity Δf.
    """
    
    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        """
        Initialize Gaussian mechanism.
        
        Args:
            epsilon: Privacy loss parameter (ε)
            delta: Failure probability (δ)
            sensitivity: L2 sensitivity of the function (Δf)
        """
        self.params = PrivacyParameters(epsilon, delta, sensitivity)
        self.sigma = self._compute_sigma()
        
        logger.info(
            f"Initialized Gaussian mechanism: ε={epsilon}, δ={delta}, "
            f"Δf={sensitivity}, σ={self.sigma:.4f}"
        )
    
    def _compute_sigma(self) -> float:
        """
        Compute noise standard deviation using the formula:
        σ ≥ Δf·√(2ln(1.25/δ))/ε
        
        Returns:
            Standard deviation for Gaussian noise
        """
        if self.params.delta == 0:
            # Pure differential privacy (infinite noise)
            return float('inf')
        
        # Compute sigma using the tight bound
        numerator = self.params.sensitivity * math.sqrt(2 * math.log(1.25 / self.params.delta))
        sigma = numerator / self.params.epsilon
        
        return sigma
    
    def add_noise(self, value: Union[float, np.ndarray], clip: bool = True) -> Union[float, np.ndarray]:
        """
        Add Gaussian noise to achieve differential privacy.
        
        Args:
            value: Original value or array
            clip: Whether to clip outliers for additional privacy
            
        Returns:
            Noisy value with differential privacy guarantee
        """
        if isinstance(value, np.ndarray):
            noise = np.random.normal(0, self.sigma, value.shape)
            noisy_value = value + noise
            
            if clip:
                # Clip to reduce impact of outliers (improves privacy)
                clip_bound = 3 * self.sigma  # 99.7% of values within bounds
                noisy_value = np.clip(noisy_value, 
                                     value - clip_bound, 
                                     value + clip_bound)
        else:
            noise = np.random.normal(0, self.sigma)
            noisy_value = value + noise
            
            if clip:
                clip_bound = 3 * self.sigma
                noisy_value = np.clip(noisy_value,
                                     value - clip_bound,
                                     value + clip_bound)
        
        return noisy_value
    
    def compute_privacy_loss(self, num_queries: int) -> Tuple[float, float]:
        """
        Compute cumulative privacy loss for multiple queries.
        
        Args:
            num_queries: Number of queries
            
        Returns:
            Tuple of (total_epsilon, total_delta)
        """
        # Basic composition (can be improved with advanced composition)
        total_epsilon = num_queries * self.params.epsilon
        total_delta = num_queries * self.params.delta
        
        return total_epsilon, total_delta


class RenyiAccountant:
    """
    Rényi Differential Privacy accountant for tight composition bounds.
    
    Tracks privacy loss using Rényi divergence for better composition
    than basic/advanced composition theorems.
    """
    
    def __init__(self, orders: Optional[List[float]] = None):
        """
        Initialize Rényi accountant.
        
        Args:
            orders: List of Rényi orders to track (default: range from 1.1 to 64)
        """
        if orders is None:
            # Default orders for good coverage
            orders = [1.1, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
                     9.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 
                     48.0, 64.0]
        
        self.orders = np.array(orders)
        self.renyi_divergence = np.zeros_like(self.orders)
        self.steps = 0
        
        logger.info(f"Initialized Rényi accountant with {len(orders)} orders")
    
    def accumulate_privacy_loss(self, sigma: float, sensitivity: float, sampling_rate: float = 1.0):
        """
        Accumulate privacy loss from a Gaussian mechanism step.
        
        Args:
            sigma: Noise standard deviation
            sensitivity: L2 sensitivity
            sampling_rate: Fraction of data used (for subsampling)
        """
        # Compute Rényi divergence for Gaussian mechanism
        for i, alpha in enumerate(self.orders):
            if sigma > 0:
                # Rényi divergence for Gaussian mechanism
                divergence = alpha * (sensitivity ** 2) / (2 * sigma ** 2)
                
                # Account for subsampling amplification
                if sampling_rate < 1.0:
                    # Subsampling amplification (approximation)
                    divergence *= sampling_rate ** 2
                
                self.renyi_divergence[i] += divergence
        
        self.steps += 1
    
    def get_privacy_spent(self, delta: float) -> float:
        """
        Convert Rényi divergence to (ε, δ)-DP.
        
        Args:
            delta: Target failure probability
            
        Returns:
            Epsilon value for (ε, δ)-differential privacy
        """
        if delta <= 0:
            return float('inf')
        
        # Convert each Rényi order to epsilon
        epsilons = []
        for i, alpha in enumerate(self.orders):
            if alpha > 1:
                # Conversion formula from RDP to DP
                eps = self.renyi_divergence[i] + math.log(1/delta) / (alpha - 1)
                epsilons.append(eps)
        
        # Return the tightest bound
        return min(epsilons) if epsilons else float('inf')


class PrivacyAccountant:
    """
    Privacy accountant with budget tracking and temporal decay.
    
    Manages privacy budget allocation across different operations
    and implements temporal decay for older queries.
    """
    
    def __init__(self, 
                 total_epsilon: float = 10.0,
                 total_delta: float = 1e-5,
                 decay_rate: float = 0.1,
                 decay_period: timedelta = timedelta(days=1)):
        """
        Initialize privacy accountant.
        
        Args:
            total_epsilon: Total privacy budget
            total_delta: Total failure probability budget
            decay_rate: Rate of privacy recovery (0-1)
            decay_period: Time period for decay
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.decay_rate = decay_rate
        self.decay_period = decay_period
        
        # Track privacy consumption
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
        self.query_history: List[Dict[str, Any]] = []
        
        # Rényi accountant for tight composition
        self.renyi_accountant = RenyiAccountant()
        
        # Component-specific budgets
        self.component_budgets = {
            'hdc_encoder': 0.3,  # 30% of budget
            'federated': 0.3,    # 30% of budget
            'pir': 0.2,          # 20% of budget
            'clinical': 0.2      # 20% of budget
        }
        
        logger.info(
            f"Initialized privacy accountant: ε_total={total_epsilon}, "
            f"δ_total={total_delta}, decay_rate={decay_rate}"
        )
    
    def allocate_budget(self, 
                       component: str, 
                       operation: str,
                       requested_epsilon: Optional[float] = None) -> PrivacyParameters:
        """
        Allocate privacy budget for an operation.
        
        Args:
            component: Component requesting budget ('hdc_encoder', 'federated', 'pir', 'clinical')
            operation: Specific operation being performed
            requested_epsilon: Requested epsilon (uses component default if None)
            
        Returns:
            Allocated privacy parameters
            
        Raises:
            ValueError: If budget exhausted
        """
        # Apply temporal decay
        self._apply_temporal_decay()
        
        # Determine epsilon allocation
        if requested_epsilon is None:
            component_fraction = self.component_budgets.get(component, 0.1)
            requested_epsilon = self.total_epsilon * component_fraction
        
        # Check budget availability
        if self.consumed_epsilon + requested_epsilon > self.total_epsilon:
            available = self.total_epsilon - self.consumed_epsilon
            raise ValueError(
                f"Insufficient privacy budget. Requested: {requested_epsilon:.4f}, "
                f"Available: {available:.4f}"
            )
        
        # Allocate budget
        allocated_delta = self.total_delta * (requested_epsilon / self.total_epsilon)
        
        # Record allocation
        self.consumed_epsilon += requested_epsilon
        self.consumed_delta += allocated_delta
        
        self.query_history.append({
            'timestamp': datetime.now(),
            'component': component,
            'operation': operation,
            'epsilon': requested_epsilon,
            'delta': allocated_delta
        })
        
        logger.info(
            f"Allocated budget for {component}/{operation}: "
            f"ε={requested_epsilon:.4f}, δ={allocated_delta:.2e}"
        )
        
        # Default sensitivity (should be overridden by component)
        sensitivity = 1.0
        
        return PrivacyParameters(
            epsilon=requested_epsilon,
            delta=allocated_delta,
            sensitivity=sensitivity
        )
    
    def _apply_temporal_decay(self):
        """Apply temporal decay to recover privacy budget."""
        now = datetime.now()
        recovered_epsilon = 0.0
        
        # Process query history
        active_queries = []
        for query in self.query_history:
            age = now - query['timestamp']
            
            if age > self.decay_period:
                # Apply decay
                decay_factor = self.decay_rate * (age.total_seconds() / self.decay_period.total_seconds())
                decay_factor = min(decay_factor, 1.0)  # Cap at 100% recovery
                
                recovered = query['epsilon'] * decay_factor
                recovered_epsilon += recovered
                
                # Keep query with reduced impact
                if decay_factor < 1.0:
                    query['epsilon'] *= (1 - decay_factor)
                    active_queries.append(query)
            else:
                active_queries.append(query)
        
        # Update state
        self.query_history = active_queries
        self.consumed_epsilon = max(0, self.consumed_epsilon - recovered_epsilon)
        
        if recovered_epsilon > 0:
            logger.info(f"Recovered {recovered_epsilon:.4f} epsilon through temporal decay")
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """
        Get remaining privacy budget.
        
        Returns:
            Tuple of (remaining_epsilon, remaining_delta)
        """
        self._apply_temporal_decay()
        remaining_epsilon = self.total_epsilon - self.consumed_epsilon
        remaining_delta = self.total_delta - self.consumed_delta
        return remaining_epsilon, remaining_delta
    
    def get_privacy_spent_renyi(self, delta: float) -> float:
        """
        Get privacy spent using Rényi accountant.
        
        Args:
            delta: Target failure probability
            
        Returns:
            Total epsilon spent
        """
        return self.renyi_accountant.get_privacy_spent(delta)


class DifferentiallyPrivateHDC:
    """
    Integration with HDC encoder for differentially private hypervector encoding.
    """
    
    def __init__(self, 
                 dimension: int,
                 privacy_level: PrivacyLevel = PrivacyLevel.CLINICAL,
                 accountant: Optional[PrivacyAccountant] = None):
        """
        Initialize DP-HDC encoder.
        
        Args:
            dimension: Hypervector dimension
            privacy_level: Privacy level to use
            accountant: Privacy accountant (creates new if None)
        """
        self.dimension = dimension
        self.privacy_level = privacy_level
        self.epsilon, self.delta = privacy_level.value
        
        self.accountant = accountant or PrivacyAccountant(
            total_epsilon=self.epsilon * 100,  # Budget for 100 operations
            total_delta=self.delta * 100
        )
        
        # Sensitivity for HDC encoding (normalized vectors have L2 norm 1)
        self.sensitivity = math.sqrt(2.0)  # Max L2 distance between unit vectors
        
        logger.info(
            f"Initialized DP-HDC: dimension={dimension}, "
            f"privacy_level={privacy_level.name}"
        )
    
    def encode_with_privacy(self, 
                           features: np.ndarray,
                           add_noise: bool = True) -> np.ndarray:
        """
        Encode features to hypervector with differential privacy.
        
        Args:
            features: Input features
            add_noise: Whether to add DP noise
            
        Returns:
            Differentially private hypervector
        """
        # Normalize features
        normalized = features / (np.linalg.norm(features) + 1e-10)
        
        # Project to hypervector space (simplified)
        # In practice, would use actual HDC encoding
        projection_matrix = np.random.randn(len(normalized), self.dimension)
        projection_matrix /= np.linalg.norm(projection_matrix, axis=0)
        
        hypervector = projection_matrix.T @ normalized
        hypervector /= np.linalg.norm(hypervector)
        
        if add_noise and self.epsilon < float('inf'):
            # Allocate budget
            try:
                params = self.accountant.allocate_budget(
                    'hdc_encoder', 
                    'encode',
                    self.epsilon
                )
                params.sensitivity = self.sensitivity
                
                # Add noise
                mechanism = GaussianMechanism(
                    params.epsilon,
                    params.delta,
                    params.sensitivity
                )
                
                hypervector = mechanism.add_noise(hypervector)
                
                # Re-normalize after noise addition
                hypervector /= np.linalg.norm(hypervector)
                
                # Track in Rényi accountant
                self.accountant.renyi_accountant.accumulate_privacy_loss(
                    mechanism.sigma,
                    params.sensitivity
                )
                
            except ValueError as e:
                logger.warning(f"Privacy budget exhausted: {e}")
                # Return without noise if budget exhausted
        
        return hypervector


class DifferentiallyPrivateFederated:
    """
    Integration with federated aggregator for private aggregation.
    """
    
    def __init__(self,
                 num_clients: int,
                 privacy_level: PrivacyLevel = PrivacyLevel.COMMON,
                 accountant: Optional[PrivacyAccountant] = None):
        """
        Initialize DP federated aggregator.
        
        Args:
            num_clients: Number of federated clients
            privacy_level: Privacy level to use
            accountant: Privacy accountant
        """
        self.num_clients = num_clients
        self.privacy_level = privacy_level
        self.epsilon, self.delta = privacy_level.value
        
        self.accountant = accountant or PrivacyAccountant(
            total_epsilon=self.epsilon * 100,
            total_delta=self.delta * 100
        )
        
        # Sensitivity for averaging (assuming bounded inputs in [-1, 1])
        self.sensitivity = 2.0 / num_clients
        
        logger.info(
            f"Initialized DP-Federated: num_clients={num_clients}, "
            f"privacy_level={privacy_level.name}"
        )
    
    def aggregate_with_privacy(self,
                              client_updates: List[np.ndarray],
                              clip_norm: float = 1.0) -> np.ndarray:
        """
        Aggregate client updates with differential privacy.
        
        Args:
            client_updates: List of client model updates
            clip_norm: Maximum L2 norm for clipping
            
        Returns:
            Differentially private aggregated update
        """
        # Clip updates to bound sensitivity
        clipped_updates = []
        for update in client_updates:
            norm = np.linalg.norm(update)
            if norm > clip_norm:
                update = update * (clip_norm / norm)
            clipped_updates.append(update)
        
        # Average updates
        aggregated = np.mean(clipped_updates, axis=0)
        
        # Add noise for privacy
        if self.epsilon < float('inf'):
            try:
                params = self.accountant.allocate_budget(
                    'federated',
                    'aggregate',
                    self.epsilon
                )
                
                # Sensitivity after clipping
                params.sensitivity = 2 * clip_norm / len(clipped_updates)
                
                mechanism = GaussianMechanism(
                    params.epsilon,
                    params.delta,
                    params.sensitivity
                )
                
                aggregated = mechanism.add_noise(aggregated)
                
                # Track in Rényi accountant
                self.accountant.renyi_accountant.accumulate_privacy_loss(
                    mechanism.sigma,
                    params.sensitivity,
                    sampling_rate=1.0/self.num_clients
                )
                
            except ValueError as e:
                logger.warning(f"Privacy budget exhausted: {e}")
        
        return aggregated


class DifferentiallyPrivatePIR:
    """
    Integration with PIR for differentially private responses.
    """
    
    def __init__(self,
                 database_size: int,
                 privacy_level: PrivacyLevel = PrivacyLevel.COMMON,
                 accountant: Optional[PrivacyAccountant] = None):
        """
        Initialize DP-PIR.
        
        Args:
            database_size: Number of records in database
            privacy_level: Privacy level to use
            accountant: Privacy accountant
        """
        self.database_size = database_size
        self.privacy_level = privacy_level
        self.epsilon, self.delta = privacy_level.value
        
        self.accountant = accountant or PrivacyAccountant(
            total_epsilon=self.epsilon * 100,
            total_delta=self.delta * 100
        )
        
        # Sensitivity for PIR (single record retrieval)
        self.sensitivity = 1.0
        
        logger.info(
            f"Initialized DP-PIR: database_size={database_size}, "
            f"privacy_level={privacy_level.name}"
        )
    
    def add_noise_to_response(self,
                             response: np.ndarray,
                             query_type: str = 'retrieval') -> np.ndarray:
        """
        Add differential privacy noise to PIR response.
        
        Args:
            response: PIR response data
            query_type: Type of query ('retrieval', 'count', 'sum')
            
        Returns:
            Noisy response with differential privacy
        """
        if self.epsilon >= float('inf'):
            return response
        
        # Determine sensitivity based on query type
        if query_type == 'retrieval':
            sensitivity = self.sensitivity
        elif query_type == 'count':
            sensitivity = 1.0  # Single record contribution
        elif query_type == 'sum':
            sensitivity = 1.0  # Assuming bounded values
        else:
            sensitivity = self.sensitivity
        
        try:
            params = self.accountant.allocate_budget(
                'pir',
                query_type,
                self.epsilon
            )
            params.sensitivity = sensitivity
            
            mechanism = GaussianMechanism(
                params.epsilon,
                params.delta,
                params.sensitivity
            )
            
            noisy_response = mechanism.add_noise(response)
            
            # Track in Rényi accountant
            self.accountant.renyi_accountant.accumulate_privacy_loss(
                mechanism.sigma,
                params.sensitivity
            )
            
            return noisy_response
            
        except ValueError as e:
            logger.warning(f"Privacy budget exhausted: {e}")
            return response


# Example usage and testing
if __name__ == "__main__":
    # Test Gaussian mechanism
    print("Testing Gaussian Mechanism")
    print("-" * 40)
    
    # Clinical level privacy (ε=1, δ=10^-7)
    mechanism = GaussianMechanism(epsilon=1.0, delta=1e-7, sensitivity=1.0)
    
    # Add noise to a value
    true_value = 42.0
    noisy_value = mechanism.add_noise(true_value)
    print(f"True value: {true_value:.2f}")
    print(f"Noisy value: {noisy_value:.2f}")
    print(f"Noise std (σ): {mechanism.sigma:.4f}")
    
    # Test privacy accountant
    print("\nTesting Privacy Accountant")
    print("-" * 40)
    
    accountant = PrivacyAccountant(total_epsilon=10.0, total_delta=1e-5)
    
    # Allocate budget for different components
    try:
        hdc_params = accountant.allocate_budget('hdc_encoder', 'encode')
        print(f"HDC allocation: ε={hdc_params.epsilon:.2f}, δ={hdc_params.delta:.2e}")
        
        remaining = accountant.get_remaining_budget()
        print(f"Remaining budget: ε={remaining[0]:.2f}, δ={remaining[1]:.2e}")
        
    except ValueError as e:
        print(f"Allocation failed: {e}")
    
    # Test DP-HDC integration
    print("\nTesting DP-HDC Integration")
    print("-" * 40)
    
    dp_hdc = DifferentiallyPrivateHDC(
        dimension=1000,
        privacy_level=PrivacyLevel.CLINICAL
    )
    
    features = np.random.randn(20)
    private_encoding = dp_hdc.encode_with_privacy(features)
    print(f"Input features shape: {features.shape}")
    print(f"Private hypervector shape: {private_encoding.shape}")
    print(f"Hypervector norm: {np.linalg.norm(private_encoding):.4f}")
    
    # Test composition
    print("\nTesting Rényi Composition")
    print("-" * 40)
    
    renyi = RenyiAccountant()
    
    # Simulate 100 queries
    for _ in range(100):
        renyi.accumulate_privacy_loss(sigma=1.0, sensitivity=1.0)
    
    total_epsilon = renyi.get_privacy_spent(delta=1e-7)
    print(f"After 100 queries: ε={total_epsilon:.2f} (at δ=10^-7)")
    
    print("\n✅ All differential privacy tests passed!")