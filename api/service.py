"""
Model service for loading and serving predictions.
"""
import mlflow
import mlflow.sklearn
import pickle
import os
from typing import Optional, Tuple
import numpy as np
from pathlib import Path


class ModelService:
    """Service for managing model loading and inference."""
    
    def __init__(self, model_name: str = "linear-regression-house-price"):
        """Initialize model service.
        
        Args:
            model_name: Name of the model in MLflow registry
        """
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = None
        self.mlflow_client = None
        
    def load_model_from_registry(self) -> bool:
        """Load the latest model from MLflow model registry.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri("file:./mlflow/mlruns")
            self.mlflow_client = mlflow.tracking.MlflowClient()
            
            # Get latest model version
            try:
                latest_version = self.mlflow_client.get_latest_versions(
                    self.model_name, 
                    stages=["None", "Staging", "Production"]
                )
                
                if not latest_version:
                    print(f"No model found in registry: {self.model_name}")
                    return self._load_local_model()
                
                # Get the most recent version
                latest = max(latest_version, key=lambda x: int(x.version))
                model_version = latest.version
                model_uri = f"models:/{self.model_name}/{model_version}"
                
                print(f"Loading model from MLflow: {model_uri}")
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_version = model_version
                
                # Try to load scaler from artifacts
                self._load_scaler_from_artifacts(latest.run_id)
                
                return True
                
            except Exception as e:
                print(f"Error loading from registry: {e}")
                return self._load_local_model()
                
        except Exception as e:
            print(f"Error initializing MLflow client: {e}")
            return self._load_local_model()
    
    def _load_local_model(self) -> bool:
        """Load model from local file as fallback.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        # Try multiple possible paths
        possible_paths = [
            Path("model/model.pkl"),
            Path("../model/model.pkl"),
            Path(__file__).parent.parent / "model" / "model.pkl"
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path and model_path.exists():
            try:
                print(f"Loading model from local file: {model_path}")
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data.get('scaler')
                    self.feature_names = model_data.get('feature_names')
                    self.model_version = "local"
                return True
            except Exception as e:
                print(f"Error loading local model: {e}")
                return False
        else:
            print(f"Local model file not found: {model_path}")
            return False
    
    def _load_scaler_from_artifacts(self, run_id: str):
        """Load scaler from MLflow artifacts.
        
        Args:
            run_id: MLflow run ID
        """
        try:
            artifacts_path = Path(f"mlflow/mlruns/0/{run_id}/artifacts")
            scaler_path = artifacts_path / "scaler.pkl"
            
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Scaler loaded from artifacts")
        except Exception as e:
            print(f"Could not load scaler from artifacts: {e}")
    
    def predict(self, features: list) -> float:
        """Make a prediction.
        
        Args:
            features: List of feature values
            
        Returns:
            Prediction value
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_from_registry() first.")
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Scale if scaler is available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        return float(prediction)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None


# Global model service instance
model_service = ModelService()

