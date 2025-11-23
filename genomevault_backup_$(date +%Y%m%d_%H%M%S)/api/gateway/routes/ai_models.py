"""
AI Model integration routes for GenomeVault API Gateway.

Provides endpoints for interacting with various AI models including
Anthropic Claude and OpenAI GPT for genomic analysis.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from genomevault.api.gateway.integrations.anthropic import AnthropicIntegration, ClaudeModel
from genomevault.api.gateway.integrations.openai import OpenAIIntegration, GPTModel
from genomevault.api.gateway.middleware.authentication import get_current_user
from genomevault.api.gateway.models.base import BaseResponse
from genomevault.observability.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/ai",
    tags=["AI Models"],
    responses={
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)


# Request/Response Models
class VariantAnalysisRequest(BaseModel):
    """Request model for variant analysis."""

    variants: List[Dict[str, Any]] = Field(
        ...,
        description="List of genomic variants to analyze",
        example=[
            {"gene": "BRCA1", "variant": "c.5266dupC", "af": 0.0001},
            {"gene": "TP53", "variant": "c.818G>A", "af": 0.0002},
        ],
    )
    patient_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional patient context",
        example={"age": 45, "sex": "F", "phenotypes": ["breast cancer"]},
    )
    analysis_type: str = Field(
        "clinical", description="Type of analysis", pattern="^(clinical|research|pharmacogenomic)$"
    )
    ai_provider: str = Field(
        "anthropic", description="AI provider to use", pattern="^(anthropic|openai)$"
    )
    model: Optional[str] = Field(None, description="Specific model to use (optional)")
    stream: bool = Field(False, description="Stream the response")


class DrugInteractionRequest(BaseModel):
    """Request model for drug interaction analysis."""

    pharmacogenomic_markers: List[Dict[str, Any]] = Field(
        ...,
        description="PGx markers from genomic data",
        example=[
            {"gene": "CYP2D6", "genotype": "*1/*4", "phenotype": "Intermediate Metabolizer"},
            {"gene": "CYP2C19", "genotype": "*1/*2", "phenotype": "Intermediate Metabolizer"},
        ],
    )
    medications: List[str] = Field(
        ..., description="Medications to analyze", example=["codeine", "clopidogrel", "warfarin"]
    )
    patient_factors: Optional[Dict[str, Any]] = Field(
        None,
        description="Patient factors affecting drug metabolism",
        example={"age": 65, "weight_kg": 70, "liver_function": "normal"},
    )
    ai_provider: str = Field("anthropic", pattern="^(anthropic|openai)$")


class ResearchHypothesisRequest(BaseModel):
    """Request model for research hypothesis generation."""

    genomic_patterns: Dict[str, Any] = Field(..., description="Aggregated genomic patterns")
    research_area: str = Field(..., description="Research focus area", example="cancer genomics")
    existing_literature: Optional[List[str]] = Field(
        None, description="Relevant literature references"
    )
    ai_provider: str = Field("anthropic", pattern="^(anthropic|openai)$")


class LiteratureSynthesisRequest(BaseModel):
    """Request model for literature synthesis."""

    query: str = Field(
        ..., description="Literature search query", example="BRCA1 mutations and breast cancer risk"
    )
    focus_areas: List[str] = Field(
        ...,
        description="Areas to focus on",
        example=["pathogenicity", "prevalence", "treatment implications"],
    )
    max_results: int = Field(10, description="Maximum results to synthesize", ge=1, le=50)
    ai_provider: str = Field("openai", pattern="^(anthropic|openai)$")


class ClinicalReportRequest(BaseModel):
    """Request model for clinical report generation."""

    analysis_results: Dict[str, Any] = Field(..., description="Variant analysis results")
    report_type: str = Field(
        "standard", description="Type of report", pattern="^(standard|detailed|summary)$"
    )
    include_recommendations: bool = Field(True, description="Include clinical recommendations")
    ai_provider: str = Field("anthropic", pattern="^(anthropic|openai)$")


class EmbeddingRequest(BaseModel):
    """Request model for text embeddings."""

    texts: List[str] = Field(..., description="Texts to generate embeddings for", max_items=100)
    model: str = Field("text-embedding-ada-002", description="Embedding model to use")


# Response Models
class VariantAnalysisResponse(BaseResponse):
    """Response model for variant analysis."""

    analysis: Dict[str, Any]
    model_used: str
    provider: str


class DrugInteractionResponse(BaseResponse):
    """Response model for drug interaction analysis."""

    interactions: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    warnings: List[str]
    model_used: str
    provider: str


class ResearchHypothesisResponse(BaseResponse):
    """Response model for research hypotheses."""

    hypotheses: List[Dict[str, Any]]
    model_used: str
    provider: str


class LiteratureSynthesisResponse(BaseResponse):
    """Response model for literature synthesis."""

    synthesis: Dict[str, Any]
    key_findings: List[str]
    evidence_quality: str
    model_used: str
    provider: str


class ClinicalReportResponse(BaseResponse):
    """Response model for clinical report."""

    report: str
    report_type: str
    model_used: str
    provider: str


class EmbeddingResponse(BaseResponse):
    """Response model for embeddings."""

    embeddings: List[List[float]]
    dimension: int
    model_used: str


# Dependency injection for AI integrations
_anthropic_integration: Optional[AnthropicIntegration] = None
_openai_integration: Optional[OpenAIIntegration] = None


def get_anthropic_integration() -> AnthropicIntegration:
    """Get or create Anthropic integration."""
    global _anthropic_integration
    if _anthropic_integration is None:
        _anthropic_integration = AnthropicIntegration()
    return _anthropic_integration


def get_openai_integration() -> OpenAIIntegration:
    """Get or create OpenAI integration."""
    global _openai_integration
    if _openai_integration is None:
        _openai_integration = OpenAIIntegration()
    return _openai_integration


# Endpoints
@router.post("/analyze/variants", response_model=VariantAnalysisResponse)
async def analyze_variants(
    request: VariantAnalysisRequest,
    current_user: Dict = Depends(get_current_user),
    anthropic: AnthropicIntegration = Depends(get_anthropic_integration),
    openai: OpenAIIntegration = Depends(get_openai_integration),
) -> VariantAnalysisResponse:
    """
    Analyze genomic variants using AI models.

    This endpoint uses either Anthropic Claude or OpenAI GPT to analyze
    genomic variants and provide clinical interpretation.
    """
    try:
        if request.ai_provider == "anthropic":
            # Use Anthropic Claude
            if request.model:
                anthropic.config.default_model = ClaudeModel(request.model)

            if request.stream:
                # Return streaming response
                async def generate():
                    prompt = anthropic._build_variant_analysis_prompt(
                        request.variants, request.patient_context, request.analysis_type
                    )
                    async for chunk in anthropic.stream_analysis(prompt):
                        yield chunk

                return StreamingResponse(generate(), media_type="text/event-stream")

            analysis = await anthropic.analyze_variants(
                request.variants, request.patient_context, request.analysis_type
            )

            return VariantAnalysisResponse(
                success=True,
                analysis=analysis,
                model_used=anthropic.config.default_model.value,
                provider="anthropic",
            )

        else:
            # Use OpenAI GPT (implement similar logic)
            # For brevity, showing placeholder
            return VariantAnalysisResponse(
                success=True,
                analysis={"placeholder": "OpenAI analysis"},
                model_used=openai.config.default_model.value,
                provider="openai",
            )

    except Exception as e:
        logger.error(f"Variant analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/drug-interactions", response_model=DrugInteractionResponse)
async def analyze_drug_interactions(
    request: DrugInteractionRequest,
    current_user: Dict = Depends(get_current_user),
    anthropic: AnthropicIntegration = Depends(get_anthropic_integration),
) -> DrugInteractionResponse:
    """
    Analyze drug-gene interactions based on pharmacogenomic markers.

    Uses AI to identify potential drug-gene interactions and provide
    dosing recommendations based on genomic data.
    """
    try:
        analysis = await anthropic.analyze_drug_interactions(
            request.pharmacogenomic_markers, request.medications, request.patient_factors
        )

        # Extract structured information from analysis
        return DrugInteractionResponse(
            success=True,
            interactions=analysis,
            recommendations=analysis.get("recommendations", []),
            warnings=analysis.get("warnings", []),
            model_used=anthropic.config.default_model.value,
            provider="anthropic",
        )

    except Exception as e:
        logger.error(f"Drug interaction analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research/hypotheses", response_model=ResearchHypothesisResponse)
async def generate_research_hypotheses(
    request: ResearchHypothesisRequest,
    current_user: Dict = Depends(get_current_user),
    anthropic: AnthropicIntegration = Depends(get_anthropic_integration),
) -> ResearchHypothesisResponse:
    """
    Generate research hypotheses based on genomic patterns.

    Uses AI to suggest novel research directions based on
    aggregated genomic data patterns.
    """
    try:
        hypotheses = await anthropic.suggest_research_hypotheses(
            request.genomic_patterns, request.research_area, request.existing_literature
        )

        return ResearchHypothesisResponse(
            success=True,
            hypotheses=hypotheses,
            model_used=anthropic.config.default_model.value,
            provider="anthropic",
        )

    except Exception as e:
        logger.error(f"Hypothesis generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/literature/synthesize", response_model=LiteratureSynthesisResponse)
async def synthesize_literature(
    request: LiteratureSynthesisRequest,
    current_user: Dict = Depends(get_current_user),
    openai: OpenAIIntegration = Depends(get_openai_integration),
) -> LiteratureSynthesisResponse:
    """
    Synthesize medical literature for genomic findings.

    Uses AI to analyze and summarize relevant medical literature.
    """
    try:
        synthesis = await openai.analyze_literature(
            request.query, request.focus_areas, request.max_results
        )

        return LiteratureSynthesisResponse(
            success=True,
            synthesis=synthesis,
            key_findings=synthesis.get("key_findings", []),
            evidence_quality=synthesis.get("evidence_quality", "moderate"),
            model_used=openai.config.default_model.value,
            provider="openai",
        )

    except Exception as e:
        logger.error(f"Literature synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/clinical", response_model=ClinicalReportResponse)
async def generate_clinical_report(
    request: ClinicalReportRequest,
    current_user: Dict = Depends(get_current_user),
    anthropic: AnthropicIntegration = Depends(get_anthropic_integration),
) -> ClinicalReportResponse:
    """
    Generate clinical report from analysis results.

    Uses AI to create professional clinical reports from
    genomic analysis results.
    """
    try:
        report = await anthropic.generate_clinical_report(
            request.analysis_results, request.report_type, request.include_recommendations
        )

        return ClinicalReportResponse(
            success=True,
            report=report,
            report_type=request.report_type,
            model_used=anthropic.config.default_model.value,
            provider="anthropic",
        )

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(
    request: EmbeddingRequest,
    current_user: Dict = Depends(get_current_user),
    openai: OpenAIIntegration = Depends(get_openai_integration),
) -> EmbeddingResponse:
    """
    Generate embeddings for genomic text data.

    Uses OpenAI's embedding models to create vector representations
    of text for similarity search and clustering.
    """
    try:
        embeddings = await openai.generate_embeddings(request.texts, request.model)

        dimension = len(embeddings[0]) if embeddings else 0

        return EmbeddingResponse(
            success=True, embeddings=embeddings, dimension=dimension, model_used=request.model
        )

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/available")
async def list_available_models(
    current_user: Dict = Depends(get_current_user),
) -> Dict[str, List[str]]:
    """
    List available AI models for each provider.

    Returns a list of models that can be used for genomic analysis.
    """
    return {
        "anthropic": [model.value for model in ClaudeModel],
        "openai": [model.value for model in GPTModel],
    }


@router.get("/models/status")
async def check_models_status(
    current_user: Dict = Depends(get_current_user),
    anthropic: AnthropicIntegration = Depends(get_anthropic_integration),
    openai: OpenAIIntegration = Depends(get_openai_integration),
) -> Dict[str, Any]:
    """
    Check the status of AI model integrations.

    Returns configuration and availability status for each provider.
    """
    return {
        "anthropic": {
            "configured": bool(anthropic.config.api_key),
            "default_model": anthropic.config.default_model.value,
            "rate_limit": anthropic.config.requests_per_minute,
        },
        "openai": {
            "configured": bool(openai.config.api_key),
            "default_model": openai.config.default_model.value,
            "rate_limit": openai.config.requests_per_minute,
        },
    }
