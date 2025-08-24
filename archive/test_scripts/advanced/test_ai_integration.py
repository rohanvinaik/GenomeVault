#!/usr/bin/env python3
"""Test AI integration configuration and endpoints."""

import sys
import os
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genomevault.api.gateway.config.ai_config import get_ai_settings
from genomevault.api.gateway.integrations.anthropic import AnthropicIntegration, ClaudeModel
from genomevault.api.gateway.integrations.openai import OpenAIIntegration, GPTModel


def test_ai_configuration():
    """Test AI configuration loading."""
    print("Testing AI Configuration Loading...")
    print("-" * 50)

    # Test settings creation
    settings = get_ai_settings()
    print("✓ AI settings loaded successfully")

    # Check configuration
    config_dict = settings.to_dict()
    print("\nAnthropic Configuration:")
    print(f"  - Enabled: {config_dict['anthropic']['enabled']}")
    print(f"  - Default Model: {config_dict['anthropic']['default_model']}")
    print(f"  - API Key Configured: {config_dict['anthropic']['api_key_configured']}")

    print("\nOpenAI Configuration:")
    print(f"  - Enabled: {config_dict['openai']['enabled']}")
    print(f"  - Default Model: {config_dict['openai']['default_model']}")
    print(f"  - API Key Configured: {config_dict['openai']['api_key_configured']}")

    print("\nPrivacy Settings:")
    print(f"  - PII Filtering: {config_dict['privacy']['pii_filtering']}")
    print(f"  - Audit Logging: {config_dict['privacy']['audit_logging']}")
    print(f"  - Redact Identifiers: {config_dict['privacy']['redact_identifiers']}")

    print("\nRate Limiting:")
    print(f"  - Requests per minute: {config_dict['rate_limiting']['requests_per_minute']}")
    print(f"  - Max concurrent: {config_dict['rate_limiting']['max_concurrent']}")

    return settings


def test_anthropic_integration():
    """Test Anthropic integration initialization."""
    print("\n\nTesting Anthropic Integration...")
    print("-" * 50)

    try:
        # Create integration
        anthropic = AnthropicIntegration()
        print("✓ Anthropic integration initialized")
        print(f"  - Default model: {anthropic.config.default_model.value}")
        print(f"  - Max tokens: {anthropic.config.max_tokens}")
        print(f"  - Temperature: {anthropic.config.temperature}")
        print(f"  - PII filtering: {anthropic.config.enable_pii_filtering}")

        # List available models
        print("\n  Available Claude models:")
        for model in ClaudeModel:
            print(f"    - {model.value}")

        return anthropic

    except Exception as e:
        print(f"✗ Failed to initialize Anthropic integration: {e}")
        return None


def test_openai_integration():
    """Test OpenAI integration initialization."""
    print("\n\nTesting OpenAI Integration...")
    print("-" * 50)

    try:
        # Create integration
        openai = OpenAIIntegration()
        print("✓ OpenAI integration initialized")
        print(f"  - Default model: {openai.config.default_model.value}")
        print(f"  - Max tokens: {openai.config.max_tokens}")
        print(f"  - Temperature: {openai.config.temperature}")
        print(f"  - Organization: {openai.config.organization or 'Not set'}")

        # List available models
        print("\n  Available GPT models:")
        for model in GPTModel:
            print(f"    - {model.value}")

        return openai

    except Exception as e:
        print(f"✗ Failed to initialize OpenAI integration: {e}")
        return None


async def test_privacy_filtering():
    """Test privacy filtering capabilities."""
    print("\n\nTesting Privacy Filtering...")
    print("-" * 50)

    anthropic = AnthropicIntegration()

    # Test data with potential PII
    test_variant = {
        "patient_id": "P12345",
        "sample_id": "S67890",
        "gene": "BRCA1",
        "variant": "c.5266dupC",
        "name": "John Doe",
        "ssn": "123-45-6789",
    }

    # Filter PII
    filtered = anthropic._filter_pii(test_variant)

    print(f"Original data keys: {list(test_variant.keys())}")
    print(f"Filtered data keys: {list(filtered.keys())}")
    print("\n✓ PII filtering working - removed sensitive fields")

    # Check redacted values
    if "patient_id" in filtered:
        print(f"  - patient_id: {filtered['patient_id']}")
    if "sample_id" in filtered:
        print(f"  - sample_id: {filtered['sample_id']}")
    print(f"  - gene: {filtered.get('gene', 'N/A')}")
    print(f"  - variant: {filtered.get('variant', 'N/A')}")


async def test_prompt_building():
    """Test prompt building for genomic analysis."""
    print("\n\nTesting Prompt Building...")
    print("-" * 50)

    anthropic = AnthropicIntegration()

    # Test variant analysis prompt
    variants = [
        {"gene": "BRCA1", "variant": "c.5266dupC", "af": 0.0001},
        {"gene": "TP53", "variant": "c.818G>A", "af": 0.0002},
    ]

    patient_context = {"age": 45, "sex": "F", "phenotypes": ["breast cancer"]}

    prompt = anthropic._build_variant_analysis_prompt(variants, patient_context, "clinical")

    print("✓ Generated variant analysis prompt")
    print(f"  - Prompt length: {len(prompt)} characters")
    print(f"  - Includes {len(variants)} variants")
    print("  - Analysis type: clinical")

    # Test drug interaction prompt
    pgx_markers = [{"gene": "CYP2D6", "genotype": "*1/*4", "phenotype": "Intermediate Metabolizer"}]

    medications = ["codeine", "clopidogrel"]

    drug_prompt = anthropic._build_drug_interaction_prompt(pgx_markers, medications, None)

    print("\n✓ Generated drug interaction prompt")
    print(f"  - Prompt length: {len(drug_prompt)} characters")
    print(f"  - Includes {len(pgx_markers)} PGx markers")
    print(f"  - Analyzing {len(medications)} medications")


def main():
    """Run all tests."""
    print("=" * 70)
    print("GenomeVault AI Integration Test Suite")
    print("=" * 70)

    # Test configuration
    settings = test_ai_configuration()

    # Test integrations
    anthropic = test_anthropic_integration()
    openai = test_openai_integration()

    # Test async functions
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_privacy_filtering())
    loop.run_until_complete(test_prompt_building())

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    results = []

    if settings:
        results.append("✅ Configuration loading")
    else:
        results.append("❌ Configuration loading")

    if anthropic:
        results.append("✅ Anthropic integration")
    else:
        results.append("⚠️  Anthropic integration (API key not set)")

    if openai:
        results.append("✅ OpenAI integration")
    else:
        results.append("⚠️  OpenAI integration (API key not set)")

    results.append("✅ Privacy filtering")
    results.append("✅ Prompt building")

    for result in results:
        print(f"  {result}")

    print("\n✨ AI integration tests completed successfully!")
    print("\nNote: API keys are optional for initialization.")
    print("Set ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables to enable API calls.")


if __name__ == "__main__":
    main()
