"""
Tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)


def test_predict_endpoint_valid(client):
    """Test prediction endpoint with valid input."""
    # This test may fail if model is not loaded, which is expected in CI
    request_data = {
        "features": [-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252]
    }
    
    response = client.post("/predict", json=request_data)
    
    # Accept both success (200) and service unavailable (503)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], (int, float))
        assert "model_version" in data


def test_predict_endpoint_invalid_features(client):
    """Test prediction endpoint with invalid feature count."""
    request_data = {
        "features": [1.0, 2.0, 3.0]  # Wrong number of features
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422  # Validation error


def test_drift_endpoint(client):
    """Test drift detection endpoint."""
    response = client.get("/drift")
    
    # Accept both success (200) and service unavailable (503)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "drift_detected" in data
        assert "mean_shift" in data
        assert "std_shift" in data
        assert "sample_size" in data
        assert isinstance(data["drift_detected"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

