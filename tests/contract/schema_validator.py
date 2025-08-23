"""Schema validation middleware for API contract enforcement."""

from __future__ import annotations

import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

import jsonschema
import yaml
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from genomevault.api.v1.errors import ErrorCodes, create_error_response


logger = logging.getLogger(__name__)


class SchemaValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API requests and responses against OpenAPI schema."""

    def __init__(
        self,
        app,
        schema_path: Optional[str] = None,
        validate_requests: bool = True,
        validate_responses: bool = True,
        strict_mode: bool = False,
    ):
        """
        Initialize schema validation middleware.

        Args:
            app: FastAPI application
            schema_path: Path to OpenAPI schema file
            validate_requests: Whether to validate incoming requests
            validate_responses: Whether to validate outgoing responses
            strict_mode: Whether to fail on schema violations
        """
        super().__init__(app)
        self.validate_requests = validate_requests
        self.validate_responses = validate_responses
        self.strict_mode = strict_mode

        # Load OpenAPI schema
        if schema_path:
            self.schema = self._load_schema(schema_path)
        else:
            # Default schema path
            schema_path = Path(__file__).parent.parent.parent / "api" / "openapi.yaml"
            self.schema = self._load_schema(str(schema_path))

        # Create JSON schema validator
        self.validator = jsonschema.Draft7Validator(self.schema)

    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load OpenAPI schema from file."""
        try:
            with open(schema_path, "r") as f:
                if schema_path.endswith(".yaml") or schema_path.endswith(".yml"):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load OpenAPI schema from {schema_path}: {e}")
            return {}

    async def dispatch(self, request: Request, call_next):
        """Process request and response validation."""
        # Validate request if enabled
        if self.validate_requests:
            await self._validate_request(request)

        # Process the request
        response = await call_next(request)

        # Validate response if enabled
        if self.validate_responses:
            response = await self._validate_response(request, response)

        return response

    async def _validate_request(self, request: Request) -> None:
        """Validate incoming request against schema."""
        if not self.schema:
            return

        # Extract request details
        method = request.method.lower()
        path = request.url.path

        # Find matching path in OpenAPI schema
        path_item = self._find_path_item(path)
        if not path_item:
            logger.debug(f"No schema found for path: {path}")
            return

        # Get operation schema
        operation = path_item.get(method)
        if not operation:
            logger.debug(f"No schema found for {method.upper()} {path}")
            return

        try:
            # Validate request body if present
            if hasattr(request, "_body") or request.method in ["POST", "PUT", "PATCH"]:
                await self._validate_request_body(request, operation)

            # Validate query parameters
            await self._validate_query_parameters(request, operation)

            # Validate headers
            await self._validate_headers(request, operation)

        except ValidationError as e:
            logger.warning(f"Request validation failed for {method.upper()} {path}: {e}")
            if self.strict_mode:
                raise HTTPException(status_code=400, detail=f"Request validation failed: {str(e)}")

    async def _validate_response(self, request: Request, response: Response) -> Response:
        """Validate outgoing response against schema."""
        if not self.schema:
            return response

        # Extract request details
        method = request.method.lower()
        path = request.url.path
        status_code = str(response.status_code)

        # Find matching path in OpenAPI schema
        path_item = self._find_path_item(path)
        if not path_item:
            return response

        # Get operation schema
        operation = path_item.get(method)
        if not operation:
            return response

        # Get response schema
        responses = operation.get("responses", {})
        response_schema = responses.get(status_code) or responses.get("default")

        if not response_schema:
            logger.debug(f"No response schema for {status_code} in {method.upper()} {path}")
            return response

        try:
            # Validate response body
            await self._validate_response_body(response, response_schema)

            # Validate response headers
            await self._validate_response_headers(response, response_schema)

        except ValidationError as e:
            logger.error(f"Response validation failed for {method.upper()} {path}: {e}")
            if self.strict_mode:
                # Return error response instead of original
                return create_error_response(
                    error_type="ResponseValidationError",
                    error_code=ErrorCodes.INTERNAL_ERROR,
                    message="Response validation failed",
                    status_code=500,
                )

        return response

    def _find_path_item(self, path: str) -> Optional[Dict[str, Any]]:
        """Find matching path item in OpenAPI schema."""
        paths = self.schema.get("paths", {})

        # Direct match first
        if path in paths:
            return paths[path]

        # Pattern matching for parameterized paths
        for schema_path, path_item in paths.items():
            if self._path_matches(path, schema_path):
                return path_item

        return None

    def _path_matches(self, actual_path: str, schema_path: str) -> bool:
        """Check if actual path matches OpenAPI path pattern."""
        actual_parts = actual_path.strip("/").split("/")
        schema_parts = schema_path.strip("/").split("/")

        if len(actual_parts) != len(schema_parts):
            return False

        for actual, schema in zip(actual_parts, schema_parts):
            # Schema parameter: {param_name}
            if schema.startswith("{") and schema.endswith("}"):
                continue
            # Exact match required
            elif actual != schema:
                return False

        return True

    async def _validate_request_body(self, request: Request, operation: Dict[str, Any]) -> None:
        """Validate request body against schema."""
        request_body_spec = operation.get("requestBody")
        if not request_body_spec:
            return

        # Get content type
        content_type = request.headers.get("content-type", "").split(";")[0]
        content_spec = request_body_spec.get("content", {}).get(content_type)

        if not content_spec:
            return

        # Get request body
        body = await request.body()
        if not body:
            if request_body_spec.get("required", False):
                raise ValidationError("Request body is required")
            return

        try:
            json_body = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in request body: {e}")

        # Validate against schema
        schema = content_spec.get("schema")
        if schema:
            self._validate_json_schema(json_body, schema)

    async def _validate_query_parameters(self, request: Request, operation: Dict[str, Any]) -> None:
        """Validate query parameters against schema."""
        parameters = operation.get("parameters", [])
        query_params = dict(request.query_params)

        for param_spec in parameters:
            if param_spec.get("in") != "query":
                continue

            param_name = param_spec["name"]
            required = param_spec.get("required", False)

            if required and param_name not in query_params:
                raise ValidationError(f"Required query parameter '{param_name}' is missing")

            if param_name in query_params:
                # Validate parameter value
                param_schema = param_spec.get("schema")
                if param_schema:
                    param_value = query_params[param_name]
                    self._validate_parameter_value(param_value, param_schema, param_name)

    async def _validate_headers(self, request: Request, operation: Dict[str, Any]) -> None:
        """Validate request headers against schema."""
        parameters = operation.get("parameters", [])

        for param_spec in parameters:
            if param_spec.get("in") != "header":
                continue

            param_name = param_spec["name"]
            required = param_spec.get("required", False)

            if required and param_name not in request.headers:
                raise ValidationError(f"Required header '{param_name}' is missing")

    async def _validate_response_body(
        self, response: Response, response_schema: Dict[str, Any]
    ) -> None:
        """Validate response body against schema."""
        content_spec = response_schema.get("content", {})
        json_content = content_spec.get("application/json")

        if not json_content:
            return

        # Get response body
        if hasattr(response, "body"):
            body = response.body
        else:
            # For streaming responses, we can't validate the body
            return

        try:
            json_body = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return  # Skip validation for non-JSON responses

        # Validate against schema
        schema = json_content.get("schema")
        if schema:
            self._validate_json_schema(json_body, schema)

    async def _validate_response_headers(
        self, response: Response, response_schema: Dict[str, Any]
    ) -> None:
        """Validate response headers against schema."""
        headers_spec = response_schema.get("headers", {})

        for header_name, header_spec in headers_spec.items():
            if header_spec.get("required", False) and header_name not in response.headers:
                raise ValidationError(f"Required response header '{header_name}' is missing")

    def _validate_json_schema(self, data: Any, schema: Dict[str, Any]) -> None:
        """Validate data against JSON schema."""
        try:
            # Resolve schema references if needed
            resolved_schema = self._resolve_schema_refs(schema)
            jsonschema.validate(data, resolved_schema)
        except jsonschema.ValidationError as e:
            raise ValidationError(f"Schema validation failed: {e.message}")

    def _validate_parameter_value(
        self, value: str, schema: Dict[str, Any], param_name: str
    ) -> None:
        """Validate parameter value against schema."""
        param_type = schema.get("type", "string")

        try:
            # Type conversion and validation
            if param_type == "integer":
                int(value)
            elif param_type == "number":
                float(value)
            elif param_type == "boolean":
                if value.lower() not in ["true", "false", "1", "0"]:
                    raise ValidationError(
                        f"Invalid boolean value for parameter '{param_name}': {value}"
                    )

            # Additional validation (enum, pattern, etc.)
            enum_values = schema.get("enum")
            if enum_values and value not in enum_values:
                raise ValidationError(
                    f"Parameter '{param_name}' value '{value}' not in allowed values: {enum_values}"
                )

        except ValueError as e:
            raise ValidationError(
                f"Invalid {param_type} value for parameter '{param_name}': {value}"
            )

    def _resolve_schema_refs(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve schema references ($ref)."""
        # Simple reference resolution - can be enhanced for complex schemas
        if "$ref" in schema:
            ref_path = schema["$ref"]
            if ref_path.startswith("#/"):
                # Internal reference
                path_parts = ref_path[2:].split("/")
                resolved = self.schema
                for part in path_parts:
                    resolved = resolved[part]
                return resolved

        return schema


class ValidationError(Exception):
    """Schema validation error."""

    pass


# Backwards compatibility validation
class BackwardsCompatibilityChecker:
    """Check API backwards compatibility between schema versions."""

    def __init__(self, old_schema: Dict[str, Any], new_schema: Dict[str, Any]):
        self.old_schema = old_schema
        self.new_schema = new_schema
        self.breaking_changes = []
        self.warnings = []

    def check_compatibility(self) -> Dict[str, Any]:
        """Check backwards compatibility and return results."""
        self._check_paths()
        self._check_schemas()

        return {
            "breaking_changes": self.breaking_changes,
            "warnings": self.warnings,
            "compatible": len(self.breaking_changes) == 0,
        }

    def _check_paths(self) -> None:
        """Check for breaking changes in API paths."""
        old_paths = set(self.old_schema.get("paths", {}).keys())
        new_paths = set(self.new_schema.get("paths", {}).keys())

        # Removed paths are breaking changes
        removed_paths = old_paths - new_paths
        for path in removed_paths:
            self.breaking_changes.append(f"Removed path: {path}")

        # Added paths are non-breaking
        added_paths = new_paths - old_paths
        for path in added_paths:
            self.warnings.append(f"Added path: {path}")

        # Check existing paths for changes
        common_paths = old_paths & new_paths
        for path in common_paths:
            self._check_path_operations(path)

    def _check_path_operations(self, path: str) -> None:
        """Check for changes in path operations."""
        old_path_item = self.old_schema["paths"][path]
        new_path_item = self.new_schema["paths"][path]

        old_operations = set(old_path_item.keys())
        new_operations = set(new_path_item.keys())

        # Removed operations are breaking
        removed_ops = old_operations - new_operations
        for op in removed_ops:
            self.breaking_changes.append(f"Removed operation: {op.upper()} {path}")

        # Check parameter compatibility for existing operations
        common_ops = old_operations & new_operations
        for op in common_ops:
            self._check_operation_parameters(path, op, old_path_item[op], new_path_item[op])

    def _check_operation_parameters(
        self, path: str, operation: str, old_op: Dict, new_op: Dict
    ) -> None:
        """Check for breaking changes in operation parameters."""
        old_params = {p["name"]: p for p in old_op.get("parameters", [])}
        new_params = {p["name"]: p for p in new_op.get("parameters", [])}

        # New required parameters are breaking
        for name, param in new_params.items():
            if name not in old_params and param.get("required", False):
                self.breaking_changes.append(
                    f"New required parameter: {name} in {operation.upper()} {path}"
                )

        # Removed required parameters are breaking
        for name, param in old_params.items():
            if name not in new_params and param.get("required", False):
                self.breaking_changes.append(
                    f"Removed required parameter: {name} from {operation.upper()} {path}"
                )

    def _check_schemas(self) -> None:
        """Check for breaking changes in data schemas."""
        old_components = self.old_schema.get("components", {})
        new_components = self.new_schema.get("components", {})

        old_schemas = old_components.get("schemas", {})
        new_schemas = new_components.get("schemas", {})

        # Check existing schemas for breaking changes
        common_schemas = set(old_schemas.keys()) & set(new_schemas.keys())
        for schema_name in common_schemas:
            self._check_schema_compatibility(
                schema_name, old_schemas[schema_name], new_schemas[schema_name]
            )

    def _check_schema_compatibility(self, name: str, old_schema: Dict, new_schema: Dict) -> None:
        """Check compatibility of individual schema."""
        old_required = set(old_schema.get("required", []))
        new_required = set(new_schema.get("required", []))

        # New required fields are breaking
        new_required_fields = new_required - old_required
        for field in new_required_fields:
            self.breaking_changes.append(f"New required field: {field} in schema {name}")

        # Removed required fields are breaking (for request schemas)
        removed_required_fields = old_required - new_required
        for field in removed_required_fields:
            self.warnings.append(f"Removed required field: {field} from schema {name}")

        # Check property types
        old_properties = old_schema.get("properties", {})
        new_properties = new_schema.get("properties", {})

        for prop_name in old_properties:
            if prop_name in new_properties:
                old_type = old_properties[prop_name].get("type")
                new_type = new_properties[prop_name].get("type")

                if old_type != new_type:
                    self.breaking_changes.append(
                        f"Changed type of {prop_name} in schema {name}: {old_type} -> {new_type}"
                    )
