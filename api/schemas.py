"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    features: List[float] = Field(
        ..., 
        description="Feature values for prediction",
        min_length=8,
        max_length=8,
        example=[-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252]
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    prediction: float = Field(..., description="Predicted house price")
    model_version: Optional[str] = Field(None, description="Model version used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 4.526,
                "model_version": "1"
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Model version")


class DriftReportResponse(BaseModel):
    """Response schema for drift detection endpoint."""
    drift_detected: bool = Field(..., description="Whether drift was detected")
    mean_shift: float = Field(..., description="Mean shift in feature distribution")
    std_shift: float = Field(..., description="Standard deviation shift")
    sample_size: int = Field(..., description="Number of samples analyzed")
    details: dict = Field(..., description="Detailed drift metrics")

