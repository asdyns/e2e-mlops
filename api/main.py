"""
FastAPI application for model inference service.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from api.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    HealthResponse,
    DriftReportResponse
)
from api.service import model_service
from api.monitor import prediction_logger, drift_detector, initialize_drift_detector
import numpy as np
from pathlib import Path


app = FastAPI(
    title="House Price Prediction API",
    description="ML inference service for house price prediction using Linear Regression",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    print("Loading model on startup...")
    success = model_service.load_model_from_registry()
    if not success:
        print("Warning: Model could not be loaded. Service may not function correctly.")
    else:
        print(f"Model loaded successfully. Version: {model_service.model_version}")
        
        # Initialize drift detector with reference data (dummy for now)
        # In production, this would load actual training data
        if model_service.scaler is not None:
            # Create dummy reference data (8 features, 100 samples)
            reference_data = np.random.randn(100, 8)
            initialize_drift_detector(reference_data, threshold=0.1)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "drift": "/drift",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded() else "unhealthy",
        model_loaded=model_service.is_loaded(),
        model_version=model_service.model_version
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction.
    
    Args:
        request: Prediction request with features
        
    Returns:
        Prediction response with predicted value
    """
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate feature count
        if len(request.features) != 8:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 8 features, got {len(request.features)}"
            )
        
        # Make prediction
        prediction = model_service.predict(request.features)
        
        # Log prediction
        prediction_logger.log_prediction(
            features=request.features,
            prediction=prediction,
            model_version=model_service.model_version
        )
        
        return PredictionResponse(
            prediction=prediction,
            model_version=model_service.model_version
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/drift", response_model=DriftReportResponse)
async def check_drift():
    """Check for data drift in recent predictions.
    
    Returns:
        Drift detection report
    """
    if drift_detector is None:
        raise HTTPException(
            status_code=503, 
            detail="Drift detector not initialized"
        )
    
    try:
        # Get recent predictions
        recent_predictions = prediction_logger.get_recent_predictions(n=100)
        
        if len(recent_predictions) == 0:
            return DriftReportResponse(
                drift_detected=False,
                mean_shift=0.0,
                std_shift=0.0,
                sample_size=0,
                details={"message": "No predictions logged yet"}
            )
        
        # Extract features from recent predictions
        features_array = np.array([pred['features'] for pred in recent_predictions])
        
        # Detect drift
        drift_report = drift_detector.detect_drift(features_array)
        
        return DriftReportResponse(**drift_report)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

