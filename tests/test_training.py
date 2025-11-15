"""
Tests for the training pipeline.
"""
import pytest
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def test_dataset_loading():
    """Test that dataset can be loaded."""
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    
    assert X.shape[0] > 0, "Dataset should have samples"
    assert X.shape[1] == 8, "Dataset should have 8 features"
    assert len(y) == X.shape[0], "Target should match number of samples"
    assert y.min() >= 0, "Target should be non-negative"


def test_preprocessing():
    """Test data preprocessing."""
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check scaling
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10), "Scaled train data should have zero mean"
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-10), "Scaled train data should have unit std"
    
    # Check shapes
    assert X_train_scaled.shape[0] == len(y_train), "Train shapes should match"
    assert X_test_scaled.shape[0] == len(y_test), "Test shapes should match"


def test_model_training():
    """Test model training."""
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Check predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    assert len(y_train_pred) == len(y_train), "Train predictions should match length"
    assert len(y_test_pred) == len(y_test), "Test predictions should match length"
    assert np.all(np.isfinite(y_train_pred)), "Train predictions should be finite"
    assert np.all(np.isfinite(y_test_pred)), "Test predictions should be finite"


def test_model_evaluation():
    """Test model evaluation metrics."""
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_test_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Basic sanity checks
    assert r2 > 0, "R² should be positive"
    assert r2 <= 1, "R² should be <= 1"
    assert rmse > 0, "RMSE should be positive"
    assert rmse < 10, "RMSE should be reasonable for this dataset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

