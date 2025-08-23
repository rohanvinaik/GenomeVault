"""GenomeVault Python SDK Client."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urljoin

import httpx

from .models import (
    GenomicVariant,
    EncodeRequest,
    EncodeResponse,
    PIRQueryRequest,
    PIRQueryResponse,
    ProofRequest,
    ProofResponse,
    ClinicalAnalysisRequest,
    ClinicalAnalysisResponse,
    HealthResponse,
)
from .exceptions import (
    GenomeVaultAPIError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
    ServiceUnavailableError,
)


logger = logging.getLogger(__name__)


class GenomeVaultClient:
    """Python client for the GenomeVault API."""

    def __init__(
        self,
        base_url: str = "https://api.genomevault.io",
        api_key: Optional[str] = None,
        oauth_token: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        """
        Initialize GenomeVault client.

        Args:
            base_url: Base URL for the GenomeVault API
            api_key: API key for authentication
            oauth_token: OAuth2 token for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_backoff: Backoff factor for retries
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Setup authentication headers
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "genomevault-python-sdk/1.0.0",
        }

        if api_key:
            self.headers["X-API-Key"] = api_key
        elif oauth_token:
            self.headers["Authorization"] = f"Bearer {oauth_token}"
        else:
            logger.warning("No authentication provided. Some endpoints may not be accessible.")

        # Create HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=self.headers,
            follow_redirects=True,
        )

    async def __aenter__(self) -> "GenomeVaultClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    def __enter__(self) -> "GenomeVaultClient":
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sync context manager exit."""
        asyncio.create_task(self.close())

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with error handling and retries.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            retry_count: Current retry count

        Returns:
            Response data

        Raises:
            GenomeVaultAPIError: On API errors
        """
        url = urljoin(self.base_url, endpoint)

        try:
            response = await self.client.request(
                method=method,
                url=url,
                json=data,
                params=params,
            )

            # Handle rate limiting
            if response.status_code == 429:
                if retry_count < self.max_retries:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                    return await self._make_request(method, endpoint, data, params, retry_count + 1)
                else:
                    raise RateLimitError("Rate limit exceeded", response=response)

            # Handle server errors with retry
            if response.status_code >= 500:
                if retry_count < self.max_retries:
                    wait_time = self.retry_backoff * (2**retry_count)
                    logger.warning(
                        f"Server error {response.status_code}. Retrying in {wait_time}s."
                    )
                    await asyncio.sleep(wait_time)
                    return await self._make_request(method, endpoint, data, params, retry_count + 1)
                else:
                    error_data = self._parse_error_response(response)
                    raise ServiceUnavailableError(
                        error_data.get("message", "Service unavailable"), response=response
                    )

            # Handle client errors
            if response.status_code >= 400:
                error_data = self._parse_error_response(response)
                self._raise_api_error(response.status_code, error_data, response)

            # Parse successful response
            return response.json()

        except httpx.RequestError as e:
            if retry_count < self.max_retries:
                wait_time = self.retry_backoff * (2**retry_count)
                logger.warning(f"Request error: {e}. Retrying in {wait_time}s.")
                await asyncio.sleep(wait_time)
                return await self._make_request(method, endpoint, data, params, retry_count + 1)
            else:
                raise GenomeVaultAPIError(f"Request failed: {e}")

    def _parse_error_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse error response body."""
        try:
            return response.json()
        except Exception:
            return {
                "type": "HTTPError",
                "code": f"GV_HTTP_{response.status_code}",
                "message": response.text or f"HTTP {response.status_code} error",
                "request_id": response.headers.get("X-Request-ID"),
            }

    def _raise_api_error(
        self, status_code: int, error_data: Dict[str, Any], response: httpx.Response
    ) -> None:
        """Raise appropriate API error based on status code and error data."""
        message = error_data.get("message", f"HTTP {status_code} error")

        if status_code == 401:
            raise AuthenticationError(message, response=response)
        elif status_code == 422:
            raise ValidationError(message, errors=error_data.get("errors"), response=response)
        elif status_code == 429:
            raise RateLimitError(message, response=response)
        elif status_code >= 500:
            raise ServiceUnavailableError(message, response=response)
        else:
            raise GenomeVaultAPIError(message, response=response)

    # Health endpoints
    async def health_check(self) -> HealthResponse:
        """Get system health status."""
        data = await self._make_request("GET", "/v1/health")
        return HealthResponse(**data)

    # Hypervector endpoints
    async def encode_variants(
        self,
        variants: List[Union[GenomicVariant, Dict[str, Any]]],
        dim: int = 8192,
        binary: bool = False,
    ) -> EncodeResponse:
        """
        Encode genomic variants into hypervectors.

        Args:
            variants: List of genomic variants
            dim: Hypervector dimension
            binary: Whether to return binary vectors

        Returns:
            Encoded hypervector response
        """
        # Convert variants to dict format if needed
        variant_dicts = []
        for variant in variants:
            if isinstance(variant, GenomicVariant):
                variant_dicts.append(variant.dict())
            else:
                variant_dicts.append(variant)

        request = EncodeRequest(variants=variant_dicts, dim=dim, binary=binary)

        data = await self._make_request("POST", "/v1/hv/encode", data=request.dict())
        return EncodeResponse(**data)

    async def encode_numeric(
        self,
        numeric: List[float],
        dim: int = 8192,
        binary: bool = False,
    ) -> EncodeResponse:
        """
        Encode numeric features into hypervectors.

        Args:
            numeric: List of numeric features
            dim: Hypervector dimension
            binary: Whether to return binary vectors

        Returns:
            Encoded hypervector response
        """
        request = EncodeRequest(numeric=numeric, dim=dim, binary=binary)

        data = await self._make_request("POST", "/v1/hv/encode", data=request.dict())
        return EncodeResponse(**data)

    # PIR endpoints
    async def pir_query(
        self,
        index: int,
        query_id: Optional[str] = None,
        timeout_seconds: int = 30,
    ) -> PIRQueryResponse:
        """
        Execute a Private Information Retrieval query.

        Args:
            index: Index to query
            query_id: Unique query identifier
            timeout_seconds: Query timeout

        Returns:
            PIR query response
        """
        request = PIRQueryRequest(index=index, query_id=query_id, timeout_seconds=timeout_seconds)

        data = await self._make_request("POST", "/v1/pir/query", data=request.dict())
        return PIRQueryResponse(**data)

    # Zero-knowledge proof endpoints
    async def generate_proof(
        self,
        proof_type: str,
        public_inputs: Dict[str, Any],
        private_inputs_hash: str,
        circuit_params: Optional[Dict[str, Any]] = None,
    ) -> ProofResponse:
        """
        Generate a zero-knowledge proof.

        Args:
            proof_type: Type of proof (genomic, clinical, research)
            public_inputs: Public inputs visible to verifiers
            private_inputs_hash: Hash of private inputs
            circuit_params: Additional circuit parameters

        Returns:
            Generated proof response
        """
        request = ProofRequest(
            proof_type=proof_type,
            public_inputs=public_inputs,
            private_inputs_hash=private_inputs_hash,
            circuit_params=circuit_params or {},
        )

        data = await self._make_request("POST", "/v1/zk/prove", data=request.dict())
        return ProofResponse(**data)

    # Clinical endpoints
    async def clinical_analysis(
        self,
        patient_id_hash: str,
        variants: List[Dict[str, Any]],
        analysis_type: str,
        population_reference: str = "gnomAD",
        consent_hash: Optional[str] = None,
    ) -> ClinicalAnalysisResponse:
        """
        Perform clinical genomic analysis.

        Args:
            patient_id_hash: Hash of patient identifier
            variants: Clinical variants for analysis
            analysis_type: Type of clinical analysis
            population_reference: Population reference database
            consent_hash: Hash of patient consent documentation

        Returns:
            Clinical analysis results
        """
        request = ClinicalAnalysisRequest(
            patient_id_hash=patient_id_hash,
            variants=variants,
            analysis_type=analysis_type,
            population_reference=population_reference,
            consent_hash=consent_hash,
        )

        data = await self._make_request("POST", "/v1/clinical/analyze", data=request.dict())
        return ClinicalAnalysisResponse(**data)

    # Convenience methods
    async def encode_vcf_variants(
        self,
        vcf_path: str,
        dim: int = 8192,
        binary: bool = False,
        max_variants: Optional[int] = None,
    ) -> EncodeResponse:
        """
        Encode variants from a VCF file.

        Args:
            vcf_path: Path to VCF file
            dim: Hypervector dimension
            binary: Whether to return binary vectors
            max_variants: Maximum number of variants to process

        Returns:
            Encoded hypervector response
        """
        try:
            import pysam
        except ImportError:
            raise ImportError(
                "pysam is required for VCF processing. Install with: pip install pysam"
            )

        variants = []
        with pysam.VariantFile(vcf_path) as vcf:
            for i, record in enumerate(vcf.fetch()):
                if max_variants and i >= max_variants:
                    break

                variant = GenomicVariant(
                    chrom=record.chrom,
                    pos=record.pos,
                    ref=record.ref,
                    alt=record.alts[0] if record.alts else ".",
                    quality=record.qual or 0.0,
                )
                variants.append(variant)

        return await self.encode_variants(variants, dim=dim, binary=binary)

    # Batch operations
    async def batch_encode(
        self,
        requests: List[Union[EncodeRequest, Dict[str, Any]]],
        max_concurrent: int = 10,
    ) -> List[EncodeResponse]:
        """
        Perform batch encoding with concurrency control.

        Args:
            requests: List of encoding requests
            max_concurrent: Maximum concurrent requests

        Returns:
            List of encoding responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def encode_single(
            request_data: Union[EncodeRequest, Dict[str, Any]],
        ) -> EncodeResponse:
            async with semaphore:
                if isinstance(request_data, EncodeRequest):
                    data = request_data.dict()
                else:
                    data = request_data

                response_data = await self._make_request("POST", "/v1/hv/encode", data=data)
                return EncodeResponse(**response_data)

        tasks = [encode_single(req) for req in requests]
        return await asyncio.gather(*tasks)


# Sync wrapper for backwards compatibility
class SyncGenomeVaultClient:
    """Synchronous wrapper for GenomeVaultClient."""

    def __init__(self, **kwargs):
        self._client = GenomeVaultClient(**kwargs)
        self._loop = None

    def _get_loop(self):
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop

    def _run_async(self, coro):
        """Run async coroutine synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def health_check(self) -> HealthResponse:
        """Sync version of health_check."""
        return self._run_async(self._client.health_check())

    def encode_variants(
        self, variants: List[Union[GenomicVariant, Dict[str, Any]]], **kwargs
    ) -> EncodeResponse:
        """Sync version of encode_variants."""
        return self._run_async(self._client.encode_variants(variants, **kwargs))

    def encode_numeric(self, numeric: List[float], **kwargs) -> EncodeResponse:
        """Sync version of encode_numeric."""
        return self._run_async(self._client.encode_numeric(numeric, **kwargs))

    def pir_query(self, index: int, **kwargs) -> PIRQueryResponse:
        """Sync version of pir_query."""
        return self._run_async(self._client.pir_query(index, **kwargs))

    def generate_proof(
        self, proof_type: str, public_inputs: Dict[str, Any], private_inputs_hash: str, **kwargs
    ) -> ProofResponse:
        """Sync version of generate_proof."""
        return self._run_async(
            self._client.generate_proof(proof_type, public_inputs, private_inputs_hash, **kwargs)
        )

    def clinical_analysis(
        self, patient_id_hash: str, variants: List[Dict[str, Any]], analysis_type: str, **kwargs
    ) -> ClinicalAnalysisResponse:
        """Sync version of clinical_analysis."""
        return self._run_async(
            self._client.clinical_analysis(patient_id_hash, variants, analysis_type, **kwargs)
        )

    def close(self):
        """Close the client."""
        self._run_async(self._client.close())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
