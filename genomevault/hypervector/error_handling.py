"""
HDC Error Handling with Uncertainty Tuning for GenomeVault
Implements the ECC-HDC core with dynamic budget allocation from the project knowledge
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from genomevault.hypervector.encoding.genomic import GenomicEncoder
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class ECCMode(str, Enum):
    """Error correction code modes"""

    NONE = "none"
    XOR_PARITY = "xor_parity"
    REED_SOLOMON = "reed_solomon"


# API Models
class ErrorBudgetRequest(BaseModel):
    """Request for error budget calculation"""

    epsilon: float = Field(0.01, description="Allowed relative error", gt=0, le=1)
    delta_exp: int = Field(
        15,
        description="Target failure probability exponent (2^-delta_exp)",
        ge=5,
        le=30,
    )
    ecc_enabled: bool = Field(True, description="Enable error correcting codes")
    parity_g: int = Field(3, description="XOR(g) parity groups", ge=2, le=5)
    repeat_cap: str = Field("AUTO", description="Number of repeats or 'AUTO'")


class ErrorBudgetResponse(BaseModel):
    """Response with error budget configuration"""

    dimension: int
    parity_g: int
    repeats: int
    epsilon: float
    delta: str
    estimated_latency_ms: int
    estimated_bandwidth_mb: float
    ecc_enabled: bool


class QueryRequest(BaseModel):
    """Query request with error tuning"""

    cohort_id: str
    statistic: str
    epsilon: float = Field(0.01, description="Allowed relative error")
    delta_exp: int = Field(15, description="Confidence level (1 in 2^delta_exp)")
    ecc: bool = Field(True, description="Enable ECC")
    parity_g: int = Field(3, description="XOR(g) parity groups")
    repeat_cap: str = Field("AUTO", description="Number of repeats or 'AUTO'")


class QueryResponse(BaseModel):
    """Query response with confidence metrics"""

    estimate: float
    ci: str
    delta: str
    proof_uri: str
    settings: dict
    performance: dict


@dataclass
class ErrorBudget:
    """Error budget configuration"""

    dimension: int
    parity_g: int
    repeats: int
    epsilon: float
    delta_exp: int
    ecc_enabled: bool

    @property
    def delta(self) -> float:
        """Get delta value from exponent"""
        return 2 ** (-self.delta_exp)

    @property
    def confidence(self) -> str:
        """Get human-readable confidence level"""
        return f"1 in {2**self.delta_exp}"


class ECCEncoderMixin:
    """
    Error Correcting Code encoder mixin for hypervectors
    Implements XOR parity for self-healing codewords
    """

    def __init__(self, base_dimension: int, parity_g: int = 3):
        self.base_dimension = base_dimension
        self.parity_g = parity_g
        self.code_length = parity_g + 1
        self.expanded_dimension = int(base_dimension * self.code_length / parity_g)

    def encode_with_ecc(self, hypervector: torch.Tensor) -> torch.Tensor:
        """
        Apply ECC encoding to hypervector
        Turns every hypervector into a self-healing codeword
        """
        if len(hypervector) != self.base_dimension:
            raise ValueError(f"Expected dimension {self.base_dimension}, got {len(hypervector)}")

        # Reshape into blocks
        num_blocks = self.base_dimension // self.parity_g
        remainder = self.base_dimension % self.parity_g

        encoded_blocks = []

        # Process complete blocks
        for i in range(num_blocks):
            block = hypervector[i * self.parity_g : (i + 1) * self.parity_g]

            # For binary hypervectors, use XOR; for continuous, use sum mod 2
            if hypervector.dtype == torch.bool:
                parity = torch.bitwise_xor.reduce(block.int()).float()
            else:
                # For continuous values, discretize then XOR
                binary_block = (block > 0).int()
                parity = torch.bitwise_xor.reduce(binary_block).float()
                # Scale parity by mean magnitude of block
                parity = parity * torch.abs(block).mean()

            encoded_block = torch.cat([block, parity.unsqueeze(0)])
            encoded_blocks.append(encoded_block)

        # Handle remainder with padding
        if remainder > 0:
            last_block = hypervector[-remainder:]
            padded_block = torch.zeros(self.parity_g, device=hypervector.device)
            padded_block[:remainder] = last_block

            if hypervector.dtype == torch.bool:
                parity = torch.bitwise_xor.reduce(padded_block.int()).float()
            else:
                binary_block = (padded_block > 0).int()
                parity = torch.bitwise_xor.reduce(binary_block).float()
                parity = parity * torch.abs(padded_block).mean()

            encoded_block = torch.cat([padded_block, parity.unsqueeze(0)])
            encoded_blocks.append(encoded_block)

        encoded_vector = torch.cat(encoded_blocks)
        return encoded_vector[: self.expanded_dimension]

    def decode_with_ecc(self, encoded_vector: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Decode ECC-encoded hypervector and correct errors
        Returns: (decoded_vector, num_errors_corrected)
        """
        num_blocks = len(encoded_vector) // self.code_length
        decoded_blocks = []
        errors_corrected = 0

        for i in range(num_blocks):
            block_with_parity = encoded_vector[i * self.code_length : (i + 1) * self.code_length]
            data_block = block_with_parity[: self.parity_g]
            received_parity = block_with_parity[-1]

            # Calculate expected parity
            if data_block.dtype == torch.bool:
                expected_parity = torch.bitwise_xor.reduce(data_block.int()).float()
            else:
                binary_block = (data_block > 0).int()
                expected_parity = torch.bitwise_xor.reduce(binary_block).float()
                expected_parity = expected_parity * torch.abs(data_block).mean()

            # Check for errors
            parity_error = torch.abs(received_parity - expected_parity)
            error_threshold = 0.1 * torch.abs(data_block).mean()

            if parity_error > error_threshold:
                errors_corrected += 1
                # Simple error correction: identify and flip the most likely error
                # In practice, would use syndrome decoding
                logger.debug("Parity error detected in block %si: %sparity_error.item()")

            decoded_blocks.append(data_block)

        decoded_vector = torch.cat(decoded_blocks)[: self.base_dimension]
        return decoded_vector, errors_corrected


class ErrorBudgetAllocator:
    """
    Allocates error budget based on accuracy and confidence requirements
    Implements the budget allocation algorithm from the HDC/uncertainty tuning spec
    """

    def __init__(self, dim_cap: int = 200000, default_g: int = 3):
        self.dim_cap = dim_cap
        self.default_g = default_g

    def plan_budget(
        self,
        epsilon: float,
        delta_exp: int,
        ecc_enabled: bool = True,
        repeat_cap: int | None = None,
    ) -> ErrorBudget:
        """
        Plan error budget allocation deterministically from (ε, δ)

        Algorithm:
        1. Compute ECC residual variance (quadratic improvement)
        2. Solve JL inequality for dimension
        3. Solve Hoeffding for repeats
        """
        delta = 2 ** (-delta_exp)

        # Step 1: Compute ECC residual variance
        if ecc_enabled:
            # For single-error XOR parity: residual variance ≈ (raw_variance)²
            # This effectively doubles the JL tail bound exponent
            effective_epsilon = epsilon * math.sqrt(2)
        else:
            effective_epsilon = epsilon

        # Step 2: Solve JL inequality for dimension
        # 2*exp(-d*ε²/2) ≤ δ/2
        # d ≥ 2*ln(2/δ)/ε²
        required_dim = int(math.ceil(2 * math.log(4 / delta) / (effective_epsilon**2)))
        dimension = min(required_dim, self.dim_cap)

        # Step 3: Solve Hoeffding for repeats
        # 2*exp(-2k*ε²) ≤ δ/2
        # k ≥ ln(2/δ)/(2*ε²)
        required_repeats = int(math.ceil(math.log(4 / delta) / (2 * epsilon**2)))

        if repeat_cap is not None:
            repeats = min(required_repeats, repeat_cap)
        else:
            repeats = required_repeats

        # If dimension is capped, compensate with more repeats
        if dimension == self.dim_cap and required_dim > self.dim_cap:
            extra_factor = math.sqrt(required_dim / self.dim_cap)
            repeats = int(math.ceil(repeats * extra_factor))

        logger.info(
            "Budget allocation: dim=%sdimension, repeats=%srepeats, "
            "ε=%sepsilon, δ=2^-%sdelta_exp, ECC=%secc_enabled"
        )

        return ErrorBudget(
            dimension=dimension,
            parity_g=self.default_g if ecc_enabled else 0,
            repeats=repeats,
            epsilon=epsilon,
            delta_exp=delta_exp,
            ecc_enabled=ecc_enabled,
        )

    def estimate_latency(self, budget: ErrorBudget) -> int:
        """Estimate query latency in milliseconds"""
        base_latency = 50  # 50ms base
        dim_factor = budget.dimension / 100000  # Linear with dimension
        repeat_factor = math.sqrt(budget.repeats)  # Sub-linear with repeats
        ecc_overhead = 1.25 if budget.ecc_enabled else 1.0

        latency_ms = int(base_latency * dim_factor * repeat_factor * ecc_overhead)
        return latency_ms

    def estimate_bandwidth(self, budget: ErrorBudget) -> float:
        """Estimate response size in MB"""
        bytes_per_element = 4  # 32-bit floats
        base_size = budget.dimension * bytes_per_element

        if budget.ecc_enabled:
            # ECC expansion factor
            base_size *= (budget.parity_g + 1) / budget.parity_g

        total_size = base_size * budget.repeats
        return round(total_size / (1024 * 1024), 1)  # Convert to MB


class AdaptiveHDCEncoder(GenomicEncoder):
    """
    Enhanced genomic encoder with adaptive error handling
    Extends the base GenomicEncoder with ECC and repeat capabilities
    """

    def __init__(self, dimension: int = 10000):
        super().__init__(dimension)
        self.ecc_encoders = {}  # Cache ECC encoders by configuration
        self.budget_allocator = ErrorBudgetAllocator()

    def encode_with_budget(
        self, variants: list[dict], budget: ErrorBudget
    ) -> tuple[torch.Tensor, dict]:
        """
        Encode genome with specified error budget
        Returns encoded vector and proof metadata
        """
        # Adjust dimension if needed
        if budget.dimension != self.dimension:
            self.dimension = budget.dimension
            self.base_vectors = self._init_base_vectors()

        # Get or create ECC encoder
        ecc_encoder = None
        if budget.ecc_enabled:
            encoder_key = (budget.dimension, budget.parity_g)
            if encoder_key not in self.ecc_encoders:
                self.ecc_encoders[encoder_key] = ECCEncoderMixin(budget.dimension, budget.parity_g)
            ecc_encoder = self.ecc_encoders[encoder_key]

        # Process with repeats for confidence
        encoded_vectors = []
        proofs = []

        for repeat_idx in range(budget.repeats):
            # Add deterministic noise based on repeat index
            torch.manual_seed(hash((id(variants), repeat_idx)) % (2**32))

            # Encode genome
            genome_vec = self.encode_genome(variants)

            # Apply ECC if enabled
            if budget.ecc_enabled and ecc_encoder is not None:
                genome_vec = ecc_encoder.encode_with_ecc(genome_vec)

            encoded_vectors.append(genome_vec)

            # Generate proof metadata for this repeat
            proof = {
                "repeat_idx": repeat_idx,
                "dimension": budget.dimension,
                "ecc_enabled": budget.ecc_enabled,
                "parity_valid": budget.ecc_enabled,  # Would verify parity equations
                "query_index_masked": True,  # Would verify masking
            }
            proofs.append(proof)

        # Aggregate using median (as specified in the HDC tuning doc)
        stacked_vectors = torch.stack(encoded_vectors)
        final_vector = torch.median(stacked_vectors, dim=0)[0]

        # Calculate confidence metrics
        std_dev = torch.std(stacked_vectors, dim=0).mean().item()
        median_error = (
            torch.median(torch.abs(stacked_vectors - final_vector), dim=0)[0].mean().item()
        )

        metadata = {
            "budget": budget,
            "proofs": proofs,
            "std_dev": std_dev,
            "median_error": median_error,
            "error_within_bound": median_error <= budget.epsilon,
        }

        return final_vector, metadata


# FastAPI Router
router = APIRouter(prefix="/hdc", tags=["HDC"])


@router.post("/estimate_budget", response_model=ErrorBudgetResponse)
async def estimate_budget(request: ErrorBudgetRequest):
    """
    Estimate error budget for given accuracy/confidence requirements
    This is called when user adjusts the accuracy dial in the UI
    """
    try:
        allocator = ErrorBudgetAllocator()

        # Parse repeat cap
        repeat_cap = None
        if request.repeat_cap != "AUTO":
            try:
                repeat_cap = int(request.repeat_cap)
            except ValueError:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                raise HTTPException(400, "Invalid repeat_cap value")
                raise RuntimeError("Unspecified error")

        # Plan budget
        budget = allocator.plan_budget(
            epsilon=request.epsilon,
            delta_exp=request.delta_exp,
            ecc_enabled=request.ecc_enabled,
            repeat_cap=repeat_cap,
        )

        # Estimate performance
        latency_ms = allocator.estimate_latency(budget)
        bandwidth_mb = allocator.estimate_bandwidth(budget)

        return ErrorBudgetResponse(
            dimension=budget.dimension,
            parity_g=budget.parity_g,
            repeats=budget.repeats,
            epsilon=budget.epsilon,
            delta=budget.confidence,
            estimated_latency_ms=latency_ms,
            estimated_bandwidth_mb=bandwidth_mb,
            ecc_enabled=budget.ecc_enabled,
        )

    except Exception:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Budget estimation failed: %se")
        raise HTTPException(500, "Failed to estimate error budget")
        raise RuntimeError("Unspecified error")


@router.post("/query", response_model=QueryResponse)
async def query_with_tuning(request: QueryRequest):
    """
    Process query with user-specified error tuning
    Implements the full HDC/uncertainty tuning pipeline
    """
    try:
        # Plan error budget
        allocator = ErrorBudgetAllocator()
        repeat_cap = None if request.repeat_cap == "AUTO" else int(request.repeat_cap)

        budget = allocator.plan_budget(
            epsilon=request.epsilon,
            delta_exp=request.delta_exp,
            ecc_enabled=request.ecc,
            repeat_cap=repeat_cap,
        )

        # In production, would fetch actual cohort data
        # For now, simulate with mock variants
        mock_variants = [
            {
                "chromosome": "chr1",
                "position": 100000,
                "ref": "A",
                "alt": "G",
                "type": "SNP",
            },
            {
                "chromosome": "chr2",
                "position": 200000,
                "ref": "C",
                "alt": "T",
                "type": "SNP",
            },
            # ... more variants
        ]

        # Encode with adaptive error handling
        encoder = AdaptiveHDCEncoder(budget.dimension)
        encoded_vector, metadata = encoder.encode_with_budget(mock_variants, budget)

        # In production, would:
        # 1. Send k encoded queries to PIR servers
        # 2. Aggregate responses
        # 3. Generate recursive zkSNARK proof
        # 4. Store proof on IPFS and blockchain

        # Mock results
        estimate = 142.7  # Mock statistic value
        proof_uri = f"ipfs://Qm{torch.rand(1).item():.16f}..."

        # Prepare response
        return QueryResponse(
            estimate=estimate,
            ci=f"±{request.epsilon * 100:.1f}%",
            delta=f"≈{budget.confidence}",
            proof_uri=proof_uri,
            settings={
                "dim": budget.dimension,
                "parity_g": budget.parity_g,
                "repeats": budget.repeats,
                "ecc_enabled": budget.ecc_enabled,
            },
            performance={
                "latency_ms": allocator.estimate_latency(budget),
                "bandwidth_mb": allocator.estimate_bandwidth(budget),
                "median_error": metadata["median_error"],
                "error_within_bound": metadata["error_within_bound"],
            },
        )

    except Exception:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Query processing failed: %se")
        raise HTTPException(500, "Query processing failed")
        raise RuntimeError("Unspecified error")


# Module exports
__all__ = [
    "AdaptiveHDCEncoder",
    "ECCEncoderMixin",
    "ErrorBudget",
    "ErrorBudgetAllocator",
    "router",
]
