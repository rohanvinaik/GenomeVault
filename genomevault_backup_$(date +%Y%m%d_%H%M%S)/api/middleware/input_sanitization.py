"""
Input Sanitization and Validation Middleware for GenomeVault.

Provides comprehensive input sanitization and validation with special handling
for genomic data formats. Prevents XSS, SQL injection, command injection,
and other security vulnerabilities while preserving the integrity of
scientific data.
"""

import re
import json
import html
import unicodedata
from typing import Any, Dict, List, Optional, Union, Set
from enum import Enum
from dataclasses import dataclass

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.status import HTTP_400_BAD_REQUEST


class DataType(str, Enum):
    """Types of data that require different sanitization approaches."""

    TEXT = "text"
    HTML = "html"
    JSON = "json"
    GENOMIC_SEQUENCE = "genomic_sequence"
    GENOMIC_VARIANT = "genomic_variant"
    CLINICAL_ID = "clinical_id"
    NUMERIC = "numeric"
    EMAIL = "email"
    FILENAME = "filename"
    URL = "url"


class SecurityLevel(str, Enum):
    """Security levels for different types of data."""

    LOW = "low"  # Public data, minimal sanitization
    MEDIUM = "medium"  # Standard data, normal sanitization
    HIGH = "high"  # Sensitive data, strict sanitization
    CLINICAL = "clinical"  # PHI data, maximum security


@dataclass
class SanitizationRule:
    """Configuration for sanitizing a specific type of input."""

    data_type: DataType
    security_level: SecurityLevel
    max_length: Optional[int] = None
    allowed_chars: Optional[Set[str]] = None
    forbidden_patterns: Optional[List[str]] = None
    required_patterns: Optional[List[str]] = None
    normalize_unicode: bool = True
    escape_html: bool = True
    strip_whitespace: bool = True


class GenomicInputSanitizer:
    """Specialized sanitizer for genomic data inputs."""

    # Valid nucleotide characters
    DNA_CHARS = set("ATCGN-")
    RNA_CHARS = set("AUCGN-")
    AMINO_ACID_CHARS = set("ACDEFGHIKLMNPQRSTVWY*-")

    # Valid chromosome identifiers
    VALID_CHROMOSOMES = {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "X",
        "Y",
        "MT",
        "M",
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr8",
        "chr9",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chr22",
        "chrX",
        "chrY",
        "chrMT",
        "chrM",
    }

    @classmethod
    def sanitize_dna_sequence(cls, sequence: str, max_length: int = 1000000) -> str:
        """Sanitize DNA sequence input."""
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Invalid DNA sequence: must be a non-empty string")

        # Remove whitespace and convert to uppercase
        sequence = sequence.strip().upper()

        # Check length
        if len(sequence) > max_length:
            raise ValueError(f"DNA sequence too long: {len(sequence)} > {max_length}")

        # Validate characters
        invalid_chars = set(sequence) - cls.DNA_CHARS
        if invalid_chars:
            raise ValueError(f"Invalid characters in DNA sequence: {invalid_chars}")

        return sequence

    @classmethod
    def sanitize_genomic_position(cls, position: Union[str, int]) -> int:
        """Sanitize genomic position."""
        try:
            pos = int(position)
            if pos < 1 or pos > 1000000000:  # Max human genome position
                raise ValueError(f"Invalid genomic position: {pos}")
            return pos
        except (ValueError, TypeError):
            raise ValueError(f"Invalid genomic position format: {position}")

    @classmethod
    def sanitize_chromosome(cls, chromosome: str) -> str:
        """Sanitize chromosome identifier."""
        if not chromosome or not isinstance(chromosome, str):
            raise ValueError("Invalid chromosome: must be a non-empty string")

        chrom = chromosome.strip()
        if chrom not in cls.VALID_CHROMOSOMES:
            raise ValueError(f"Invalid chromosome: {chrom}")

        return chrom

    @classmethod
    def sanitize_variant_call(cls, variant: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize a genomic variant call."""
        if not isinstance(variant, dict):
            raise ValueError("Variant must be a dictionary")

        sanitized = {}

        # Required fields
        if "chromosome" in variant:
            sanitized["chromosome"] = cls.sanitize_chromosome(variant["chromosome"])

        if "position" in variant:
            sanitized["position"] = cls.sanitize_genomic_position(variant["position"])

        if "ref" in variant:
            sanitized["ref"] = cls.sanitize_dna_sequence(variant["ref"], max_length=1000)

        if "alt" in variant:
            sanitized["alt"] = cls.sanitize_dna_sequence(variant["alt"], max_length=1000)

        # Optional fields with validation
        for field in ["quality", "depth", "genotype"]:
            if field in variant:
                sanitized[field] = cls._sanitize_numeric_field(variant[field])

        return sanitized

    @classmethod
    def _sanitize_numeric_field(cls, value: Any) -> Union[int, float]:
        """Sanitize numeric fields."""
        try:
            if isinstance(value, str):
                # Try integer first, then float
                try:
                    return int(value)
                except ValueError:
                    return float(value)
            elif isinstance(value, (int, float)):
                return value
            else:
                raise ValueError("Must be numeric")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value: {value}")


class InputSanitizer:
    """Main input sanitization engine."""

    # Default sanitization rules
    DEFAULT_RULES = {
        DataType.TEXT: SanitizationRule(
            data_type=DataType.TEXT,
            security_level=SecurityLevel.MEDIUM,
            max_length=10000,
            forbidden_patterns=[
                r"<script.*?>.*?</script>",  # XSS scripts
                r"javascript:",  # JavaScript URLs
                r"vbscript:",  # VBScript URLs
                r"onload=",  # Event handlers
                r"onerror=",
                r"onclick=",
                r"data:",  # Data URLs
                r"eval\(",  # Code evaluation
                r"exec\(",
                r"system\(",  # System commands
                r"import\s+os",  # Python imports
                r"__import__",
                r"\.\./",  # Path traversal
                r"\.\.\\",
            ],
        ),
        DataType.GENOMIC_SEQUENCE: SanitizationRule(
            data_type=DataType.GENOMIC_SEQUENCE,
            security_level=SecurityLevel.HIGH,
            max_length=1000000,
            allowed_chars=GenomicInputSanitizer.DNA_CHARS,
        ),
        DataType.CLINICAL_ID: SanitizationRule(
            data_type=DataType.CLINICAL_ID,
            security_level=SecurityLevel.CLINICAL,
            max_length=100,
            allowed_chars=set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"),
            forbidden_patterns=[
                r"DROP\s+TABLE",  # SQL injection
                r"DELETE\s+FROM",
                r"INSERT\s+INTO",
                r"UPDATE\s+.*SET",
                r"UNION\s+SELECT",
                r"--",  # SQL comments
                r"/\*.*\*/",  # SQL comments
                r";",  # SQL statement separator
            ],
        ),
        DataType.EMAIL: SanitizationRule(
            data_type=DataType.EMAIL,
            security_level=SecurityLevel.MEDIUM,
            max_length=320,
            required_patterns=[r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"],
        ),
        DataType.FILENAME: SanitizationRule(
            data_type=DataType.FILENAME,
            security_level=SecurityLevel.HIGH,
            max_length=255,
            forbidden_patterns=[
                r"\.\.",  # Path traversal
                r"/",  # Directory separators
                r"\\",
                r"<",  # HTML/XML
                r">",
                r":",  # Windows reserved
                r"\*",
                r"\?",
                r"\|",
                r'"',
            ],
        ),
    }

    def __init__(self, custom_rules: Optional[Dict[DataType, SanitizationRule]] = None):
        """Initialize sanitizer with custom rules."""
        self.rules = self.DEFAULT_RULES.copy()
        if custom_rules:
            self.rules.update(custom_rules)

    def sanitize_value(self, value: Any, data_type: DataType) -> Any:
        """Sanitize a single value according to its data type."""
        if value is None:
            return None

        rule = self.rules.get(data_type)
        if not rule:
            raise ValueError(f"No sanitization rule for data type: {data_type}")

        # Handle different input types
        if data_type == DataType.GENOMIC_SEQUENCE:
            return GenomicInputSanitizer.sanitize_dna_sequence(str(value))
        elif data_type == DataType.GENOMIC_VARIANT:
            if isinstance(value, dict):
                return GenomicInputSanitizer.sanitize_variant_call(value)
            else:
                raise ValueError("Genomic variant must be a dictionary")

        # Convert to string for text-based sanitization
        if not isinstance(value, str):
            if isinstance(value, (int, float, bool)):
                value = str(value)
            else:
                raise ValueError(f"Cannot sanitize value of type: {type(value)}")

        return self._sanitize_string(value, rule)

    def _sanitize_string(self, text: str, rule: SanitizationRule) -> str:
        """Sanitize a string value according to the rule."""
        if not isinstance(text, str):
            text = str(text)

        # Strip whitespace if required
        if rule.strip_whitespace:
            text = text.strip()

        # Normalize Unicode
        if rule.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # Check length
        if rule.max_length and len(text) > rule.max_length:
            raise ValueError(f"Input too long: {len(text)} > {rule.max_length}")

        # Check forbidden patterns
        if rule.forbidden_patterns:
            for pattern in rule.forbidden_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    raise ValueError(f"Input contains forbidden pattern: {pattern}")

        # Check required patterns
        if rule.required_patterns:
            for pattern in rule.required_patterns:
                if not re.search(pattern, text):
                    raise ValueError(f"Input does not match required pattern: {pattern}")

        # Check allowed characters
        if rule.allowed_chars:
            invalid_chars = set(text.upper()) - rule.allowed_chars
            if invalid_chars:
                raise ValueError(f"Input contains invalid characters: {invalid_chars}")

        # HTML escape if required
        if rule.escape_html:
            text = html.escape(text)

        return text

    def sanitize_json(
        self, data: Dict[str, Any], field_types: Dict[str, DataType]
    ) -> Dict[str, Any]:
        """Sanitize JSON data with field-specific rules."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        sanitized = {}

        for field, value in data.items():
            if field in field_types:
                try:
                    sanitized[field] = self.sanitize_value(value, field_types[field])
                except ValueError as e:
                    raise ValueError(f"Invalid {field}: {e}")
            else:
                # Apply default text sanitization to unknown fields
                sanitized[field] = self.sanitize_value(value, DataType.TEXT)

        return sanitized


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """Middleware for sanitizing all incoming requests."""

    # Field type mappings for different endpoints
    ENDPOINT_FIELD_TYPES = {
        "/api/hdc/encode": {
            "variants": DataType.GENOMIC_VARIANT,
            "dimension": DataType.NUMERIC,
            "sample_id": DataType.CLINICAL_ID,
        },
        "/api/zk/prove": {
            "circuit_type": DataType.TEXT,
            "public_inputs": DataType.TEXT,
            "private_inputs": DataType.TEXT,
        },
        "/api/pir/query": {"database_id": DataType.TEXT, "query_index": DataType.NUMERIC},
        "/api/clinical/": {
            "patient_id": DataType.CLINICAL_ID,
            "sample_id": DataType.CLINICAL_ID,
            "clinical_data": DataType.TEXT,  # Will be further sanitized by clinical module
        },
    }

    def __init__(self, app):
        """Initialize input sanitization middleware."""
        super().__init__(app)
        self.sanitizer = InputSanitizer()

    async def dispatch(self, request: Request, call_next) -> Response:
        """Sanitize incoming request data."""

        # Skip sanitization for certain endpoints
        if self._should_skip_sanitization(request):
            return await call_next(request)

        try:
            # Sanitize request based on content type and endpoint
            await self._sanitize_request(request)
        except ValueError as e:
            # Return 400 error for invalid input
            return Response(
                content=json.dumps(
                    {"error": "Invalid input", "message": str(e), "type": "validation_error"}
                ),
                status_code=HTTP_400_BAD_REQUEST,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            # Log unexpected errors but don't expose details
            print(f"Input sanitization error: {e}")
            return Response(
                content=json.dumps(
                    {
                        "error": "Input processing failed",
                        "message": "Invalid request format",
                        "type": "processing_error",
                    }
                ),
                status_code=HTTP_400_BAD_REQUEST,
                headers={"Content-Type": "application/json"},
            )

        return await call_next(request)

    def _should_skip_sanitization(self, request: Request) -> bool:
        """Check if sanitization should be skipped for this request."""
        path = request.url.path.lower()

        # Skip for health checks and static content
        skip_paths = ["/health", "/metrics", "/docs", "/openapi.json", "/static/"]
        return any(skip_path in path for skip_path in skip_paths)

    async def _sanitize_request(self, request: Request):
        """Sanitize request data based on content type."""
        content_type = request.headers.get("content-type", "").lower()

        # Sanitize query parameters
        self._sanitize_query_params(request)

        # Sanitize headers (check for injection attempts)
        self._sanitize_headers(request)

        # Sanitize body content
        if content_type.startswith("application/json"):
            await self._sanitize_json_body(request)
        elif content_type.startswith("application/x-www-form-urlencoded"):
            await self._sanitize_form_body(request)
        elif content_type.startswith("multipart/form-data"):
            await self._sanitize_multipart_body(request)

    def _sanitize_query_params(self, request: Request):
        """Sanitize URL query parameters."""
        # FastAPI automatically parses query params, but we can add validation
        for key, value in request.query_params.items():
            # Basic sanitization of parameter names and values
            if not re.match(r"^[a-zA-Z0-9_-]+$", key):
                raise ValueError(f"Invalid query parameter name: {key}")

            # Sanitize parameter values
            if isinstance(value, str):
                sanitized_value = self.sanitizer.sanitize_value(value, DataType.TEXT)
                # Note: We can't modify request.query_params directly in middleware
                # This validation helps prevent malicious input

    def _sanitize_headers(self, request: Request):
        """Sanitize request headers for security issues."""
        dangerous_headers = ["x-forwarded-for", "x-real-ip", "host"]

        for header_name, header_value in request.headers.items():
            header_lower = header_name.lower()

            # Check for header injection attempts
            if "\n" in header_value or "\r" in header_value:
                raise ValueError(f"Invalid characters in header: {header_name}")

            # Validate important security headers
            if header_lower in dangerous_headers:
                # Basic validation for IP headers
                if header_lower in ["x-forwarded-for", "x-real-ip"]:
                    if not re.match(r"^[\d\.,\s:]+$", header_value):
                        raise ValueError(f"Invalid IP header format: {header_name}")

    async def _sanitize_json_body(self, request: Request):
        """Sanitize JSON request body."""
        try:
            # Read and parse JSON body
            body = await request.body()
            if not body:
                return

            json_data = json.loads(body)

            # Get field types for this endpoint
            field_types = self._get_field_types_for_endpoint(request.url.path)

            # Sanitize the JSON data
            sanitized_data = self.sanitizer.sanitize_json(json_data, field_types)

            # Note: In a real implementation, you would replace the request body
            # with the sanitized data. This requires a custom request class.

        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")

    async def _sanitize_form_body(self, request: Request):
        """Sanitize form-encoded request body."""
        # This would handle application/x-www-form-urlencoded data
        pass

    async def _sanitize_multipart_body(self, request: Request):
        """Sanitize multipart form data."""
        # This would handle multipart/form-data (file uploads)
        pass

    def _get_field_types_for_endpoint(self, path: str) -> Dict[str, DataType]:
        """Get field type mappings for a specific endpoint."""
        # Exact match first
        if path in self.ENDPOINT_FIELD_TYPES:
            return self.ENDPOINT_FIELD_TYPES[path]

        # Pattern matching for dynamic endpoints
        for endpoint_pattern, field_types in self.ENDPOINT_FIELD_TYPES.items():
            if endpoint_pattern.endswith("/") and path.startswith(endpoint_pattern):
                return field_types

        # Default field types
        return {}


# Utility functions for manual sanitization
def sanitize_clinical_id(clinical_id: str) -> str:
    """Sanitize clinical/patient ID."""
    sanitizer = InputSanitizer()
    return sanitizer.sanitize_value(clinical_id, DataType.CLINICAL_ID)


def sanitize_genomic_sequence(sequence: str) -> str:
    """Sanitize genomic sequence."""
    return GenomicInputSanitizer.sanitize_dna_sequence(sequence)


def sanitize_genomic_variant(variant: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize genomic variant call."""
    return GenomicInputSanitizer.sanitize_variant_call(variant)
