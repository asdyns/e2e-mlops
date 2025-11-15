"""
Monitoring module for prediction logging and drift detection.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path


class PredictionLogger:
    """Logs predictions for monitoring purposes."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize prediction logger.
        
        Args:
            log_dir: Directory to store prediction logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "predictions.jsonl"
    
    def log_prediction(self, features: List[float], prediction: float, 
                      model_version: Optional[str] = None):
        """Log a prediction with timestamp.
        
        Args:
            features: Input features
            prediction: Model prediction
            model_version: Model version used
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "prediction": prediction,
            "model_version": model_version
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_recent_predictions(self, n: int = 1000) -> List[Dict]:
        """Get recent predictions from log.
        
        Args:
            n: Number of recent predictions to retrieve
            
        Returns:
            List of prediction dictionaries
        """
        if not self.log_file.exists():
            return []
        
        predictions = []
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-n:]:
                try:
                    predictions.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return predictions


class DriftDetector:
    """Simple drift detection based on feature distribution shifts."""
    
    def __init__(self, reference_data: np.ndarray, threshold: float = 0.1):
        """Initialize drift detector.
        
        Args:
            reference_data: Reference dataset (training data)
            threshold: Threshold for drift detection (relative change)
        """
        self.reference_mean = np.mean(reference_data, axis=0)
        self.reference_std = np.std(reference_data, axis=0)
        self.threshold = threshold
    
    def detect_drift(self, current_data: np.ndarray) -> Dict:
        """Detect drift in current data compared to reference.
        
        Args:
            current_data: Current data to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        if len(current_data) == 0:
            return {
                "drift_detected": False,
                "mean_shift": 0.0,
                "std_shift": 0.0,
                "sample_size": 0,
                "details": {}
            }
        
        current_mean = np.mean(current_data, axis=0)
        current_std = np.std(current_data, axis=0)
        
        # Calculate relative shifts
        mean_shift = np.abs((current_mean - self.reference_mean) / (self.reference_std + 1e-8))
        std_shift = np.abs((current_std - self.reference_std) / (self.reference_std + 1e-8))
        
        # Overall drift metrics
        mean_shift_avg = np.mean(mean_shift)
        std_shift_avg = np.mean(std_shift)
        
        # Detect drift if any feature exceeds threshold
        drift_detected = (mean_shift_avg > self.threshold) or (std_shift_avg > self.threshold)
        
        return {
            "drift_detected": bool(drift_detected),
            "mean_shift": float(mean_shift_avg),
            "std_shift": float(std_shift_avg),
            "sample_size": len(current_data),
            "details": {
                "mean_shift_by_feature": mean_shift.tolist(),
                "std_shift_by_feature": std_shift.tolist(),
                "threshold": self.threshold
            }
        }


# Global instances
prediction_logger = PredictionLogger()
drift_detector: Optional[DriftDetector] = None


def initialize_drift_detector(reference_data: np.ndarray, threshold: float = 0.1):
    """Initialize the global drift detector.
    
    Args:
        reference_data: Reference dataset
        threshold: Drift detection threshold
    """
    global drift_detector
    drift_detector = DriftDetector(reference_data, threshold)

