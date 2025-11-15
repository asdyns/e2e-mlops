"""
Metaflow training pipeline for house price prediction using Linear Regression.
Integrates with MLflow for experiment tracking and model registry.
"""
from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import mlflow
import mlflow.sklearn
import pickle
import os
from datetime import datetime
from pathlib import Path


class TrainFlow(FlowSpec):
    """
    Metaflow pipeline for training a Linear Regression model on California Housing dataset.
    """
    
    # Parameters
    test_size = Parameter('test_size', default=0.2, help='Test set size ratio')
    random_state = Parameter('random_state', default=42, help='Random seed')
    model_name = Parameter('model_name', default='linear-regression-house-price', 
                          help='MLflow model registry name')
    
    @step
    def start(self):
        """Initialize the flow."""
        print("Starting training pipeline...")
        self.next(self.load_dataset)
    
    @step
    def load_dataset(self):
        """Load California Housing dataset from sklearn."""
        print("Loading California Housing dataset...")
        housing = fetch_california_housing()
        
        self.X = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.y = pd.Series(housing.target, name='target')
        
        print(f"Dataset loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Target range: [{self.y.min():.2f}, {self.y.max():.2f}]")
        
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        """Preprocess data: train/test split and feature scaling."""
        print("Preprocessing data...")
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Train set: {self.X_train_scaled.shape[0]} samples")
        print(f"Test set: {self.X_test_scaled.shape[0]} samples")
        
        self.next(self.train)
    
    @step
    def train(self):
        """Train Linear Regression model."""
        print("Training Linear Regression model...")
        
        self.model = LinearRegression()
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print("Model training completed!")
        self.next(self.evaluate)
    
    @step
    def evaluate(self):
        """Evaluate model and log metrics to MLflow."""
        print("Evaluating model...")
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        self.metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        print(f"Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
        
        self.next(self.log_to_mlflow)
    
    @step
    def log_to_mlflow(self):
        """Log experiment to MLflow and register model."""
        print("Logging to MLflow...")
        
        # Set MLflow tracking URI (local)
        mlflow.set_tracking_uri("file:./mlflow/mlruns")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"train_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("n_features", self.X_train.shape[1])
            mlflow.log_param("n_train_samples", self.X_train.shape[0])
            mlflow.log_param("n_test_samples", self.X_test.shape[0])
            
            # Log metrics
            for metric_name, metric_value in self.metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                registered_model_name=self.model_name
            )
            
            # Log scaler as artifact
            scaler_path = "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            mlflow.log_artifact(scaler_path)
            
            # Save run ID for later use
            self.mlflow_run_id = mlflow.active_run().info.run_id
            print(f"MLflow run ID: {self.mlflow_run_id}")
        
        # Also save model locally for direct access
        model_dir = Path(__file__).parent.parent / "model"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.X_train.columns.tolist()
            }, f)
        print(f"Model saved locally to {model_path}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """End of the flow."""
        print("Training pipeline completed successfully!")
        print(f"Model registered in MLflow as: {self.model_name}")


if __name__ == '__main__':
    TrainFlow()

