"""
Anthropic Claude integration for GenomeVault API Gateway.

Provides secure integration with Anthropic's Claude models for:
- Genomic data analysis and interpretation
- Privacy-preserving medical report generation
- Clinical decision support
- Research hypothesis generation
"""

from __future__ import annotations

import os
import asyncio
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import httpx
from datetime import datetime, timedelta

from genomevault.observability.logging import get_logger

logger = get_logger(__name__)


class ClaudeModel(Enum):
    """Available Claude models."""
    
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_2_1 = "claude-2.1"
    CLAUDE_2 = "claude-2.0"
    CLAUDE_INSTANT = "claude-instant-1.2"
    
    # Latest models
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"


@dataclass
class AnthropicConfig:
    """Configuration for Anthropic integration."""
    
    api_key: Optional[str] = None
    api_url: str = "https://api.anthropic.com/v1"
    default_model: ClaudeModel = ClaudeModel.CLAUDE_3_5_SONNET
    max_tokens: int = 4096
    temperature: float = 0.0  # For deterministic medical outputs
    timeout: int = 60
    max_retries: int = 3
    
    # Privacy settings
    enable_pii_filtering: bool = True
    enable_audit_logging: bool = True
    redact_genomic_identifiers: bool = True
    
    # Rate limiting
    requests_per_minute: int = 50
    max_concurrent_requests: int = 10
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            
        if not self.api_key:
            logger.warning("Anthropic API key not configured")


class AnthropicIntegration:
    """
    Anthropic Claude integration for genomic analysis.
    
    Provides privacy-preserving integration with Claude models for:
    - Clinical variant interpretation
    - Medical report generation
    - Research hypothesis generation
    - Drug interaction analysis
    """
    
    def __init__(self, config: Optional[AnthropicConfig] = None):
        """
        Initialize Anthropic integration.
        
        Args:
            config: Anthropic configuration
        """
        self.config = config or AnthropicConfig()
        self.client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers={
                "x-api-key": self.config.api_key or "",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "messages-2023-12-15"
            }
        )
        self._rate_limiter = RateLimiter(
            requests_per_minute=self.config.requests_per_minute,
            max_concurrent=self.config.max_concurrent_requests
        )
        
        logger.info(
            f"Initialized Anthropic integration with model {self.config.default_model.value}"
        )
    
    async def analyze_variants(
        self,
        variants: List[Dict[str, Any]],
        patient_context: Optional[Dict[str, Any]] = None,
        analysis_type: str = "clinical"
    ) -> Dict[str, Any]:
        """
        Analyze genomic variants using Claude.
        
        Args:
            variants: List of variant data
            patient_context: Optional patient context (age, symptoms, etc.)
            analysis_type: Type of analysis (clinical, research, pharmacogenomic)
            
        Returns:
            Analysis results with interpretations
        """
        # Redact PII if enabled
        if self.config.redact_genomic_identifiers:
            variants = self._redact_identifiers(variants)
            
        # Build prompt
        prompt = self._build_variant_analysis_prompt(
            variants, patient_context, analysis_type
        )
        
        # Call Claude API
        response = await self._call_claude(
            prompt,
            system="You are a genomic medicine expert assistant. Provide accurate, evidence-based analysis of genetic variants while maintaining patient privacy. Never store or learn from patient-specific data."
        )
        
        # Parse and structure response
        return self._parse_variant_analysis(response)
    
    async def generate_clinical_report(
        self,
        analysis_results: Dict[str, Any],
        report_type: str = "standard",
        include_recommendations: bool = True
    ) -> str:
        """
        Generate a clinical report from analysis results.
        
        Args:
            analysis_results: Variant analysis results
            report_type: Type of report (standard, detailed, summary)
            include_recommendations: Whether to include clinical recommendations
            
        Returns:
            Formatted clinical report
        """
        prompt = self._build_report_prompt(
            analysis_results, report_type, include_recommendations
        )
        
        response = await self._call_claude(
            prompt,
            system="You are a clinical genetics report writer. Generate clear, accurate, and actionable clinical reports that comply with medical documentation standards. Maintain strict patient confidentiality."
        )
        
        return self._format_clinical_report(response)
    
    async def suggest_research_hypotheses(
        self,
        genomic_data: Dict[str, Any],
        research_area: str,
        existing_literature: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate research hypotheses based on genomic data.
        
        Args:
            genomic_data: Aggregated genomic patterns
            research_area: Area of research focus
            existing_literature: Optional literature references
            
        Returns:
            List of research hypotheses with rationales
        """
        prompt = self._build_research_prompt(
            genomic_data, research_area, existing_literature
        )
        
        response = await self._call_claude(
            prompt,
            system="You are a genomics research advisor. Suggest novel, testable research hypotheses based on genomic patterns while ensuring all suggestions maintain patient privacy and follow research ethics guidelines.",
            temperature=0.7  # Higher temperature for creative hypothesis generation
        )
        
        return self._parse_research_hypotheses(response)
    
    async def analyze_drug_interactions(
        self,
        pharmacogenomic_markers: List[Dict[str, Any]],
        medications: List[str],
        patient_factors: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze potential drug-gene interactions.
        
        Args:
            pharmacogenomic_markers: PGx markers from genomic data
            medications: List of medications to analyze
            patient_factors: Optional factors (age, liver function, etc.)
            
        Returns:
            Drug interaction analysis and recommendations
        """
        prompt = self._build_pgx_prompt(
            pharmacogenomic_markers, medications, patient_factors
        )
        
        response = await self._call_claude(
            prompt,
            system="You are a pharmacogenomics specialist. Analyze drug-gene interactions and provide evidence-based dosing recommendations. Always prioritize patient safety and cite relevant guidelines (CPIC, DPWG, FDA)."
        )
        
        return self._parse_drug_interactions(response)
    
    async def stream_analysis(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream analysis results for real-time updates.
        
        Args:
            prompt: Analysis prompt
            system: System message
            
        Yields:
            Streamed response chunks
        """
        await self._rate_limiter.acquire()
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.config.api_url}/messages",
                json={
                    "model": self.config.default_model.value,
                    "messages": [{"role": "user", "content": prompt}],
                    "system": system or "You are a helpful genomics assistant.",
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "stream": True
                }
            ) as response:
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        yield chunk
                        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
    
    async def _call_claude(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Make API call to Claude.
        
        Args:
            prompt: User prompt
            system: System message
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Claude's response
        """
        await self._rate_limiter.acquire()
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.post(
                    f"{self.config.api_url}/messages",
                    json={
                        "model": self.config.default_model.value,
                        "messages": [{"role": "user", "content": prompt}],
                        "system": system or "You are a helpful assistant.",
                        "max_tokens": max_tokens or self.config.max_tokens,
                        "temperature": temperature or self.config.temperature
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["content"][0]["text"]
                    
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = int(response.headers.get("retry-after", 60))
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        raise RuntimeError("Failed to get response from Claude")
    
    def _redact_identifiers(self, data: Any) -> Any:
        """Redact patient identifiers from data."""
        # Implementation would redact PII, sample IDs, etc.
        # This is a placeholder for the actual implementation
        return data
    
    def _build_variant_analysis_prompt(
        self,
        variants: List[Dict[str, Any]],
        patient_context: Optional[Dict[str, Any]],
        analysis_type: str
    ) -> str:
        """Build prompt for variant analysis."""
        prompt = f"""Analyze the following genomic variants for {analysis_type} significance:

Variants:
"""
        for variant in variants[:100]:  # Limit to prevent token overflow
            prompt += f"- {variant.get('gene', 'Unknown')}: {variant.get('variant', 'Unknown')} (AF: {variant.get('af', 'Unknown')})\n"
        
        if patient_context:
            prompt += f"\nClinical Context:\n"
            prompt += f"- Age: {patient_context.get('age', 'Unknown')}\n"
            prompt += f"- Sex: {patient_context.get('sex', 'Unknown')}\n"
            prompt += f"- Phenotypes: {', '.join(patient_context.get('phenotypes', []))}\n"
        
        prompt += """
Please provide:
1. Clinical significance of identified variants
2. Disease associations with evidence levels
3. Recommended follow-up testing if applicable
4. Key findings summary

Format as structured JSON."""
        
        return prompt
    
    def _parse_variant_analysis(self, response: str) -> Dict[str, Any]:
        """Parse variant analysis response."""
        try:
            import json
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
                return json.loads(json_str)
            else:
                # Fallback to text parsing
                return {
                    "analysis": response,
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": self.config.default_model.value
                }
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return {"error": "Failed to parse analysis", "raw": response}
    
    def _build_report_prompt(
        self,
        analysis_results: Dict[str, Any],
        report_type: str,
        include_recommendations: bool
    ) -> str:
        """Build prompt for report generation."""
        prompt = f"""Generate a {report_type} clinical genetics report based on the following analysis:

{analysis_results}

Report Requirements:
- Use clear, non-technical language where appropriate
- Include relevant clinical guidelines and citations
- Organize findings by clinical significance
"""
        
        if include_recommendations:
            prompt += "- Provide actionable clinical recommendations\n"
            prompt += "- Suggest appropriate follow-up and monitoring\n"
        
        prompt += "\nFormat the report with appropriate sections and professional medical documentation standards."
        
        return prompt
    
    def _format_clinical_report(self, response: str) -> str:
        """Format clinical report response."""
        # Add standard headers and formatting
        report = f"""
================================================================================
                          CLINICAL GENETICS REPORT
================================================================================
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
Model: Anthropic {self.config.default_model.value}

{response}

================================================================================
DISCLAIMER: This report is generated using AI assistance and should be reviewed
by a qualified healthcare professional before clinical use.
================================================================================
"""
        return report
    
    def _build_research_prompt(
        self,
        genomic_data: Dict[str, Any],
        research_area: str,
        existing_literature: Optional[List[str]]
    ) -> str:
        """Build prompt for research hypothesis generation."""
        prompt = f"""Based on the following aggregated genomic patterns, suggest research hypotheses in {research_area}:

Genomic Patterns:
{genomic_data}
"""
        
        if existing_literature:
            prompt += f"\nRelevant Literature:\n"
            for ref in existing_literature[:10]:
                prompt += f"- {ref}\n"
        
        prompt += """
Generate 3-5 novel research hypotheses that:
1. Build on observed patterns
2. Are testable with current technology
3. Have potential clinical or biological significance
4. Consider ethical implications

For each hypothesis, provide:
- Clear hypothesis statement
- Rationale based on the data
- Suggested experimental approach
- Potential impact if validated
"""
        
        return prompt
    
    def _parse_research_hypotheses(self, response: str) -> List[Dict[str, Any]]:
        """Parse research hypotheses from response."""
        hypotheses = []
        
        # Simple parsing - in production would use more sophisticated NLP
        sections = response.split("\n\n")
        for section in sections:
            if "hypothesis" in section.lower():
                hypotheses.append({
                    "hypothesis": section,
                    "generated_at": datetime.utcnow().isoformat(),
                    "model": self.config.default_model.value
                })
        
        return hypotheses
    
    def _build_pgx_prompt(
        self,
        markers: List[Dict[str, Any]],
        medications: List[str],
        patient_factors: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for pharmacogenomic analysis."""
        prompt = f"""Analyze drug-gene interactions for the following:

Pharmacogenomic Markers:
"""
        for marker in markers:
            prompt += f"- {marker.get('gene')}: {marker.get('genotype')} ({marker.get('phenotype', 'Unknown')})\n"
        
        prompt += f"\nMedications to Analyze:\n"
        for med in medications:
            prompt += f"- {med}\n"
        
        if patient_factors:
            prompt += f"\nPatient Factors:\n"
            for factor, value in patient_factors.items():
                prompt += f"- {factor}: {value}\n"
        
        prompt += """
Provide:
1. Drug-gene interaction assessment for each medication
2. Dosing recommendations based on genotype
3. Alternative medications if interactions are severe
4. Relevant clinical guidelines (CPIC, DPWG, FDA)
5. Monitoring recommendations

Use evidence-based guidelines and indicate confidence levels."""
        
        return prompt
    
    def _parse_drug_interactions(self, response: str) -> Dict[str, Any]:
        """Parse drug interaction analysis."""
        return {
            "analysis": response,
            "timestamp": datetime.utcnow().isoformat(),
            "model": self.config.default_model.value,
            "disclaimer": "Consult prescribing information and clinical guidelines before making medication changes."
        }
    
    async def close(self):
        """Close client connections."""
        await self.client.aclose()


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int, max_concurrent: int):
        """Initialize rate limiter."""
        self.requests_per_minute = requests_per_minute
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_times: List[datetime] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self.semaphore:
            async with self.lock:
                now = datetime.utcnow()
                
                # Remove old requests outside the window
                self.request_times = [
                    t for t in self.request_times
                    if now - t < timedelta(minutes=1)
                ]
                
                # Check if we're at the limit
                if len(self.request_times) >= self.requests_per_minute:
                    # Wait until the oldest request expires
                    wait_time = 60 - (now - self.request_times[0]).total_seconds()
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                
                # Record this request
                self.request_times.append(now)


# Example usage functions
async def example_variant_analysis():
    """Example of variant analysis using Claude."""
    integration = AnthropicIntegration()
    
    variants = [
        {"gene": "BRCA1", "variant": "c.5266dupC", "af": 0.0001},
        {"gene": "TP53", "variant": "c.818G>A", "af": 0.0002},
        {"gene": "MTHFR", "variant": "c.677C>T", "af": 0.35}
    ]
    
    patient_context = {
        "age": 45,
        "sex": "F",
        "phenotypes": ["breast cancer", "family history"]
    }
    
    try:
        results = await integration.analyze_variants(
            variants, patient_context, "clinical"
        )
        
        # Generate report
        report = await integration.generate_clinical_report(
            results, "detailed", include_recommendations=True
        )
        
        print(report)
        
    finally:
        await integration.close()


async def example_streaming():
    """Example of streaming responses."""
    integration = AnthropicIntegration()
    
    try:
        prompt = "Explain the clinical significance of BRCA1 mutations."
        
        async for chunk in integration.stream_analysis(prompt):
            print(chunk, end="", flush=True)
            
    finally:
        await integration.close()