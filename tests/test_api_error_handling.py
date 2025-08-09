# tests/test_api_error_handling.py
import pytest
from fastapi.testclient import TestClient
from genomevault.api.main import app
from genomevault.exceptions import GVInputError


client = TestClient(app)


def test_root_endpoint():
    """Test that the root endpoint works."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_gv_input_error_handling():
    """Test that GVInputError returns consistent JSON via the error handler."""
    # Test empty data
    response = client.post("/api/v1/encode", json={"data": [], "seed": 42})
    assert response.status_code == 400

    error_data = response.json()
    assert error_data["type"] == "GVInputError"
    assert error_data["code"] == "GV_INPUT"
    assert "Input data cannot be empty" in error_data["message"]
    assert "details" in error_data
    assert error_data["details"]["field"] == "data"


def test_gv_input_error_inconsistent_row_lengths():
    """Test validation error for inconsistent row lengths."""
    # Test inconsistent row lengths
    response = client.post(
        "/api/v1/encode", json={"data": [[1.0, 2.0], [3.0, 4.0, 5.0]], "seed": 42}
    )
    assert response.status_code == 400

    error_data = response.json()
    assert error_data["type"] == "GVInputError"
    assert error_data["code"] == "GV_INPUT"
    assert "same length" in error_data["message"]
    assert "lengths" in error_data["details"]


def test_gv_input_error_empty_rows():
    """Test validation error for empty rows."""
    response = client.post(
        "/api/v1/encode", json={"data": [[], [1.0, 2.0]], "seed": 42}
    )
    assert response.status_code == 400

    error_data = response.json()
    assert error_data["type"] == "GVInputError"
    assert error_data["code"] == "GV_INPUT"
    assert "empty" in error_data["message"]


def test_health_endpoint():
    """Test that health endpoint works."""
    response = client.get("/health")
    assert response.status_code == 200


def test_exception_structure():
    """Test that our exceptions have the expected structure."""
    exc = GVInputError("Test message", details={"test": "value"})

    # Test to_dict method
    data = exc.to_dict()
    assert data["type"] == "GVInputError"
    assert data["code"] == "GV_INPUT"
    assert data["message"] == "Test message"
    assert data["details"]["test"] == "value"

    # Test HTTP status
    assert exc.http_status == 400
