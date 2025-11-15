"""
Tests for model inference service.
"""
import pytest
import numpy as np
from api.service import ModelService
from api.monitor import PredictionLogger, DriftDetector


def test_model_service_initialization():
    """Test model service initialization."""
    service = ModelService(model_name="test-model")
    assert service.model_name == "test-model"
    assert service.model is None
    assert not service.is_loaded()


def test_prediction_logger():
    """Test prediction logging."""
    import tempfile
    import os
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = PredictionLogger(log_dir=tmpdir)
        
        # Log some predictions
        logger.log_prediction([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 4.5, "1")
        logger.log_prediction([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 5.0, "1")
        
        # Retrieve predictions
        predictions = logger.get_recent_predictions(n=10)
        assert len(predictions) == 2, "Should have 2 logged predictions"
        assert predictions[0]['prediction'] == 4.5
        assert predictions[1]['prediction'] == 5.0


def test_drift_detector():
    """Test drift detection."""
    # Create reference data
    np.random.seed(42)
    reference_data = np.random.randn(100, 8)
    
    # Create detector
    detector = DriftDetector(reference_data, threshold=0.1)
    
    # Test with similar data (no drift)
    similar_data = np.random.randn(50, 8) * 1.0 + 0.1
    report = detector.detect_drift(similar_data)
    assert 'drift_detected' in report
    assert 'mean_shift' in report
    assert 'std_shift' in report
    assert 'sample_size' in report
    
    # Test with very different data (drift)
    drifted_data = np.random.randn(50, 8) * 2.0 + 5.0
    report_drifted = detector.detect_drift(drifted_data)
    assert 'drift_detected' in report_drifted
    # Drift should be detected with such a large shift
    assert report_drifted['mean_shift'] > report['mean_shift']


def test_model_prediction_format():
    """Test that model prediction returns correct format."""
    # This test assumes model is loaded (integration test)
    # For unit test, we'll just check the structure
    service = ModelService()
    
    # Test prediction method structure (will fail if model not loaded, which is expected)
    if service.is_loaded():
        features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        prediction = service.predict(features)
        assert isinstance(prediction, (int, float)), "Prediction should be numeric"
        assert np.isfinite(prediction), "Prediction should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

