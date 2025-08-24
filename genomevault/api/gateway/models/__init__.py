"""
Pydantic models for GenomeVault API Gateway.

Comprehensive validation models for all API endpoints including:
- Request/response validation
- Data serialization
- Type checking
- Documentation generation
"""

from __future__ import annotations

from genomevault.api.gateway.models.base import *
from genomevault.api.gateway.models.algorithms import *
from genomevault.api.gateway.models.health import *
from genomevault.api.gateway.models.models import *
from genomevault.api.gateway.models.pipelines import *
from genomevault.api.gateway.models.proofs import *
from genomevault.api.gateway.models.queries import *
from genomevault.api.gateway.models.specialized import *
from genomevault.api.gateway.models.vectors import *
from genomevault.api.gateway.models.websockets import *

__all__ = [
    # Base models
    "BaseModel",
    "ErrorResponse",
    "SuccessResponse",
    "PaginatedResponse",
    "RequestMetadata",
    
    # Health models
    "HealthStatus",
    "ServiceStatus",
    "HealthCheckResponse",
    
    # Pipeline models
    "PipelineConfig",
    "PipelineStatus",
    "PipelineExecution",
    "PipelineResult",
    
    # Vector models
    "VectorEncodeRequest",
    "VectorEncodeResponse",
    "VectorCompareRequest",
    "VectorCompareResponse",
    
    # Proof models
    "ProofGenerationRequest",
    "ProofGenerationResponse",
    "ProofVerificationRequest",
    "ProofVerificationResponse",
    
    # Query models
    "PIRQueryRequest",
    "PIRQueryResponse",
    "QueryExecutionRequest",
    "QueryExecutionResponse",
    
    # Model management
    "ModelCreateRequest",
    "ModelUpdateRequest",
    "ModelResponse",
    "ModelTrainingRequest",
    "ModelTrainingResponse",
    
    # Algorithm marketplace
    "AlgorithmListRequest",
    "AlgorithmResponse",
    "AlgorithmExecutionRequest",
    "AlgorithmExecutionResponse",
    
    # Specialized endpoints
    "TopologyRequest",
    "TopologyResponse",
    "CreditRedemptionRequest",
    "CreditRedemptionResponse",
    "AuditChallengeRequest",
    "AuditChallengeResponse",
    
    # WebSocket models
    "WebSocketMessage",
    "WebSocketResponse",
    "SubscriptionRequest",
    "SubscriptionResponse",
]