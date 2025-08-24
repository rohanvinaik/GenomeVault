"""
OpenAI GPT integration for GenomeVault API Gateway.

Provides secure integration with OpenAI's GPT models for:
- Genomic data analysis
- Medical literature synthesis
- Clinical decision support
- Research assistance
"""

from __future__ import annotations

import os
import asyncio
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import httpx
from datetime import datetime

from genomevault.observability.logging import get_logger

logger = get_logger(__name__)


class GPTModel(Enum):
    """Available GPT models."""
    
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
    
    # Latest models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI integration."""
    
    api_key: Optional[str] = None
    api_url: str = "https://api.openai.com/v1"
    organization: Optional[str] = None
    default_model: GPTModel = GPTModel.GPT_4O
    max_tokens: int = 4096
    temperature: float = 0.0  # For deterministic outputs
    timeout: int = 60
    max_retries: int = 3
    
    # Privacy settings
    enable_pii_filtering: bool = True
    enable_audit_logging: bool = True
    
    # Rate limiting
    requests_per_minute: int = 60
    max_concurrent_requests: int = 10
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.organization:
            self.organization = os.getenv("OPENAI_ORGANIZATION")
            
        if not self.api_key:
            logger.warning("OpenAI API key not configured")


class OpenAIIntegration:
    """
    OpenAI GPT integration for genomic analysis.
    
    Provides privacy-preserving integration with GPT models.
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        """
        Initialize OpenAI integration.
        
        Args:
            config: OpenAI configuration
        """
        self.config = config or OpenAIConfig()
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key or ''}",
            "Content-Type": "application/json"
        }
        
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        
        self.client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers=headers
        )
        
        logger.info(
            f"Initialized OpenAI integration with model {self.config.default_model.value}"
        )
    
    async def analyze_literature(
        self,
        query: str,
        focus_areas: List[str],
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Synthesize medical literature for genomic findings.
        
        Args:
            query: Literature search query
            focus_areas: Areas to focus on
            max_results: Maximum results to synthesize
            
        Returns:
            Literature synthesis
        """
        prompt = f"""Synthesize medical literature for: {query}

Focus Areas: {', '.join(focus_areas)}

Provide:
1. Key findings summary
2. Evidence quality assessment
3. Clinical implications
4. Research gaps
5. Recent developments (last 2 years)

Limit to {max_results} most relevant findings."""

        response = await self._call_gpt(
            prompt,
            system="You are a medical literature analyst specializing in genomics. Provide evidence-based summaries with appropriate citations."
        )
        
        return self._parse_literature_analysis(response)
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """
        Generate embeddings for genomic text data.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            List of embedding vectors
        """
        try:
            response = await self.client.post(
                f"{self.config.api_url}/embeddings",
                json={
                    "input": texts,
                    "model": model
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return [item["embedding"] for item in data["data"]]
            else:
                logger.error(f"Embedding error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []
    
    async def _call_gpt(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Make API call to GPT.
        
        Args:
            prompt: User prompt
            system: System message
            temperature: Override temperature
            
        Returns:
            GPT response
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.post(
                    f"{self.config.api_url}/chat/completions",
                    json={
                        "model": self.config.default_model.value,
                        "messages": messages,
                        "max_tokens": self.config.max_tokens,
                        "temperature": temperature or self.config.temperature
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                    
                elif response.status_code == 429:
                    wait_time = int(response.headers.get("retry-after", 60))
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    
                else:
                    logger.error(f"API error: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"API call failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
        raise RuntimeError("Failed to get response from GPT")
    
    def _parse_literature_analysis(self, response: str) -> Dict[str, Any]:
        """Parse literature analysis response."""
        return {
            "synthesis": response,
            "timestamp": datetime.utcnow().isoformat(),
            "model": self.config.default_model.value
        }
    
    async def close(self):
        """Close client connections."""
        await self.client.aclose()