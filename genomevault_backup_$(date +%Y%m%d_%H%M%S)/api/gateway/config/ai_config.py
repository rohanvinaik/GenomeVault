"""
AI Integration Configuration for GenomeVault API Gateway.

Centralized configuration for Anthropic Claude and OpenAI GPT integrations.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AIIntegrationSettings(BaseModel):
    """
    Settings for AI model integrations.

    Configured via environment variables with GENOMEVAULT_ prefix.
    """

    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(
        None, env="ANTHROPIC_API_KEY", description="Anthropic API key for Claude access"
    )
    anthropic_default_model: str = Field(
        "claude-3-5-sonnet-20241022",
        env="GENOMEVAULT_ANTHROPIC_MODEL",
        description="Default Claude model to use",
    )
    anthropic_max_tokens: int = Field(
        4096,
        env="GENOMEVAULT_ANTHROPIC_MAX_TOKENS",
        description="Maximum tokens for Claude responses",
    )
    anthropic_temperature: float = Field(
        0.0,
        env="GENOMEVAULT_ANTHROPIC_TEMPERATURE",
        description="Temperature for Claude (0.0 for deterministic)",
    )

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(
        None, env="OPENAI_API_KEY", description="OpenAI API key for GPT access"
    )
    openai_organization: Optional[str] = Field(
        None, env="OPENAI_ORGANIZATION", description="OpenAI organization ID"
    )
    openai_default_model: str = Field(
        "gpt-4o", env="GENOMEVAULT_OPENAI_MODEL", description="Default GPT model to use"
    )
    openai_max_tokens: int = Field(
        4096, env="GENOMEVAULT_OPENAI_MAX_TOKENS", description="Maximum tokens for GPT responses"
    )
    openai_temperature: float = Field(
        0.0,
        env="GENOMEVAULT_OPENAI_TEMPERATURE",
        description="Temperature for GPT (0.0 for deterministic)",
    )

    # Privacy and Security Settings
    enable_pii_filtering: bool = Field(
        True, env="GENOMEVAULT_AI_PII_FILTERING", description="Enable PII filtering in AI requests"
    )
    enable_audit_logging: bool = Field(
        True, env="GENOMEVAULT_AI_AUDIT_LOGGING", description="Enable audit logging for AI requests"
    )
    redact_genomic_identifiers: bool = Field(
        True,
        env="GENOMEVAULT_AI_REDACT_IDENTIFIERS",
        description="Redact genomic identifiers before sending to AI",
    )

    # Rate Limiting
    ai_requests_per_minute: int = Field(
        50, env="GENOMEVAULT_AI_RATE_LIMIT", description="Maximum AI requests per minute per user"
    )
    ai_max_concurrent: int = Field(
        10, env="GENOMEVAULT_AI_MAX_CONCURRENT", description="Maximum concurrent AI requests"
    )

    # Feature Flags
    enable_anthropic: bool = Field(
        True, env="GENOMEVAULT_ENABLE_ANTHROPIC", description="Enable Anthropic Claude integration"
    )
    enable_openai: bool = Field(
        True, env="GENOMEVAULT_ENABLE_OPENAI", description="Enable OpenAI GPT integration"
    )
    enable_streaming: bool = Field(
        True,
        env="GENOMEVAULT_AI_ENABLE_STREAMING",
        description="Enable streaming responses from AI models",
    )

    # Clinical Settings
    require_clinical_auth: bool = Field(
        True,
        env="GENOMEVAULT_AI_REQUIRE_CLINICAL_AUTH",
        description="Require clinical authentication for medical analysis",
    )
    clinical_disclaimer: str = Field(
        "AI-generated content should be reviewed by qualified healthcare professionals before clinical use.",
        env="GENOMEVAULT_AI_CLINICAL_DISCLAIMER",
        description="Disclaimer for clinical AI outputs",
    )

    model_config = {"env_prefix": "GENOMEVAULT_", "case_sensitive": False}

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "anthropic": {
                "enabled": self.enable_anthropic,
                "api_key_configured": bool(self.anthropic_api_key),
                "default_model": self.anthropic_default_model,
                "max_tokens": self.anthropic_max_tokens,
                "temperature": self.anthropic_temperature,
            },
            "openai": {
                "enabled": self.enable_openai,
                "api_key_configured": bool(self.openai_api_key),
                "default_model": self.openai_default_model,
                "max_tokens": self.openai_max_tokens,
                "temperature": self.openai_temperature,
            },
            "privacy": {
                "pii_filtering": self.enable_pii_filtering,
                "audit_logging": self.enable_audit_logging,
                "redact_identifiers": self.redact_genomic_identifiers,
            },
            "rate_limiting": {
                "requests_per_minute": self.ai_requests_per_minute,
                "max_concurrent": self.ai_max_concurrent,
            },
            "features": {
                "streaming_enabled": self.enable_streaming,
                "clinical_auth_required": self.require_clinical_auth,
            },
        }


# Singleton instance
_settings: Optional[AIIntegrationSettings] = None


def get_ai_settings() -> AIIntegrationSettings:
    """
    Get AI integration settings singleton.

    Returns:
        AIIntegrationSettings: Configured settings instance
    """
    global _settings
    if _settings is None:
        _settings = AIIntegrationSettings()
    return _settings


# Pre-configured prompts for specific use cases
GENOMIC_ANALYSIS_PROMPTS = {
    "variant_interpretation": """
You are a clinical genetics expert. Analyze the provided genomic variants and provide:
1. Clinical significance assessment (benign, likely benign, VUS, likely pathogenic, pathogenic)
2. Disease associations with evidence levels
3. Inheritance patterns
4. Recommended follow-up testing
5. Key clinical considerations

Use ACMG/AMP guidelines for variant classification. Cite relevant databases (ClinVar, gnomAD, COSMIC).
""",
    "pharmacogenomics": """
You are a pharmacogenomics specialist. Analyze drug-gene interactions and provide:
1. Metabolizer phenotype predictions
2. Dosing recommendations based on genotype
3. Drug interaction risks
4. Alternative medication suggestions
5. Monitoring recommendations

Reference CPIC, DPWG, and FDA guidelines. Include confidence levels for recommendations.
""",
    "research_hypothesis": """
You are a genomics research advisor. Based on the genomic patterns provided:
1. Identify novel associations or patterns
2. Suggest testable hypotheses
3. Recommend experimental approaches
4. Consider ethical implications
5. Identify potential confounders

Focus on translational potential and clinical relevance. Consider existing literature.
""",
    "clinical_report": """
You are a clinical genetics report writer. Generate a professional report that:
1. Summarizes key findings clearly
2. Provides clinical interpretation
3. Includes actionable recommendations
4. Lists limitations and caveats
5. Suggests follow-up actions

Use language appropriate for healthcare providers. Follow medical documentation standards.
""",
}


# Privacy-preserving templates
PRIVACY_TEMPLATES = {
    "redacted_variant": {
        "gene": "[GENE]",
        "variant": "[VARIANT]",
        "sample_id": "[REDACTED]",
        "patient_id": "[REDACTED]",
    },
    "anonymized_patient": {
        "id": "[PATIENT_ID]",
        "age_range": "[AGE_RANGE]",
        "sex": "[M/F/O]",
        "ethnicity": "[POPULATION_GROUP]",
    },
}


# Model selection criteria
MODEL_SELECTION_CRITERIA = {
    "clinical_analysis": {
        "preferred_provider": "anthropic",
        "preferred_model": "claude-3-5-sonnet-20241022",
        "reason": "Superior medical reasoning and safety features",
        "fallback": "openai/gpt-4o",
    },
    "literature_synthesis": {
        "preferred_provider": "openai",
        "preferred_model": "gpt-4o",
        "reason": "Excellent at synthesizing large amounts of text",
        "fallback": "anthropic/claude-3-5-sonnet-20241022",
    },
    "hypothesis_generation": {
        "preferred_provider": "anthropic",
        "preferred_model": "claude-3-5-sonnet-20241022",
        "reason": "Creative reasoning with scientific rigor",
        "fallback": "openai/gpt-4o",
    },
    "quick_queries": {
        "preferred_provider": "anthropic",
        "preferred_model": "claude-3-5-haiku-20241022",
        "reason": "Fast responses for simple queries",
        "fallback": "openai/gpt-4o-mini",
    },
}
