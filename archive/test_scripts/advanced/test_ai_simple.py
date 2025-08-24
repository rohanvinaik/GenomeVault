#!/usr/bin/env python3
"""Simple test of AI integration without full import chain."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_direct_config():
    """Test AI configuration directly."""
    print("Testing AI Configuration...")
    print("-" * 50)

    # Test just the config without full import chain
    from genomevault.api.gateway.config.ai_config import AIIntegrationSettings

    # Create settings
    settings = AIIntegrationSettings()

    print("✓ AI settings created successfully")
    print(f"  - Anthropic model: {settings.anthropic_default_model}")
    print(f"  - OpenAI model: {settings.openai_default_model}")
    print(f"  - PII filtering: {settings.enable_pii_filtering}")
    print(f"  - Rate limit: {settings.ai_requests_per_minute} req/min")

    return True


def test_integrations():
    """Test AI integrations directly."""
    print("\nTesting AI Integrations...")
    print("-" * 50)

    try:
        from genomevault.api.gateway.integrations.anthropic import AnthropicIntegration

        anthropic = AnthropicIntegration()
        print("✓ Anthropic integration created")

        from genomevault.api.gateway.integrations.openai import OpenAIIntegration

        openai = OpenAIIntegration()
        print("✓ OpenAI integration created")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


if __name__ == "__main__":
    success = True

    try:
        success = test_direct_config() and success
    except Exception as e:
        print(f"Config test failed: {e}")
        success = False

    try:
        success = test_integrations() and success
    except Exception as e:
        print(f"Integration test failed: {e}")
        success = False

    if success:
        print("\n✅ All AI integration tests passed!")
    else:
        print("\n❌ Some tests failed")

    sys.exit(0 if success else 1)
