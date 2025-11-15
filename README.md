# End-to-End ML System: House Price Prediction

A complete machine learning system for house price prediction using Linear Regression, featuring:

- **Metaflow** for training pipeline orchestration
- **MLflow** for experiment tracking and model registry
- **FastAPI** for model inference service
- **Docker** containerization
- **Model monitoring** with prediction logging and drift detection

## 📁 Project Structure

```
project/
├── data/
│   ├── raw/
│   └── processed/
├── flows/
│   └── train_flow.py          # Metaflow training pipeline
├── mlflow/
│   └── mlruns/                # MLflow tracking data
├── model/
│   └── model.pkl              # Saved model (auto-generated)
├── api/
│   ├── main.py                # FastAPI application
│   ├── service.py             # Model service
│   ├── schemas.py             # Pydantic schemas
│   └── monitor.py           # Monitoring utilities
├── tests/
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_api.py
├── Dockerfile
├── requirements.txt
├── README.md
└── Makefile
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd cursor-ml
   ```

2. **Install dependencies**:
   ```bash
   make install
   # or
   pip install -r requirements.txt
   ```

### Training Pipeline

Run the Metaflow training pipeline:

```bash
make train
# or
cd flows && python train_flow.py run
```

This will:
1. Load the California Housing dataset
2. Preprocess the data (train/test split, scaling)
3. Train a Linear Regression model
4. Evaluate the model (R², RMSE)
5. Log everything to MLflow
6. Register the model in MLflow Model Registry
7. Save the model locally

### Inference Service

Start the FastAPI inference service:

```bash
make serve
# or
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

Returns service status and model information.

#### 2. Prediction
```bash
POST /predict
Content-Type: application/json

{
  "features": [-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252]
}
```

Returns:
```json
{
  "prediction": 4.526,
  "model_version": "1"
}
```

**Feature Description** (California Housing Dataset):
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

#### 3. Drift Detection
```bash
GET /drift
```

Returns drift detection report based on recent predictions.

## 🐳 Docker Deployment

### Build Docker Image

```bash
make docker-build
# or
docker build -t house-price-api:latest .
```

### Run Docker Container

```bash
make docker-run
# or
docker run -p 8000:8000 \
  -v $(PWD)/mlflow:/app/mlflow \
  -v $(PWD)/model:/app/model \
  -v $(PWD)/logs:/app/logs \
  house-price-api:latest
```

The service will be available at http://localhost:8000

## 🧪 Testing

Run all tests:

```bash
make test
# or
pytest tests/ -v
```

Run specific test files:

```bash
pytest tests/test_training.py -v
pytest tests/test_inference.py -v
pytest tests/test_api.py -v
```

## 📊 MLflow Integration

### View Experiments

MLflow tracking data is stored locally in `mlflow/mlruns/`. To view experiments:

1. **Install MLflow UI** (if not already installed):
   ```bash
   pip install mlflow
   ```

2. **Start MLflow UI**:
   ```bash
   mlflow ui --backend-store-uri file:./mlflow/mlruns
   ```

3. **Open browser**: http://localhost:5000

### Model Registry

Models are automatically registered in MLflow Model Registry with the name:
- `linear-regression-house-price`

The inference service loads the latest version from the registry automatically.

## 📈 Monitoring

### Prediction Logging

All predictions are automatically logged to `logs/predictions.jsonl` with:
- Timestamp
- Input features
- Prediction value
- Model version

### Drift Detection

The drift detection endpoint (`/drift`) analyzes recent predictions and compares feature distributions to the training data. It reports:
- Whether drift was detected
- Mean shift in feature distributions
- Standard deviation shift
- Per-feature drift metrics

## 🔧 Configuration

### Model Parameters

Training parameters can be adjusted in `flows/train_flow.py`:

```python
test_size = Parameter('test_size', default=0.2)
random_state = Parameter('random_state', default=42)
model_name = Parameter('model_name', default='linear-regression-house-price')
```

### API Configuration

API settings can be modified in `api/main.py`:
- Host: `0.0.0.0`
- Port: `8000`

### Monitoring Configuration

Drift detection threshold can be adjusted in `api/monitor.py`:

```python
initialize_drift_detector(reference_data, threshold=0.1)
```

## 📝 Makefile Commands

```bash
make help          # Show available commands
make install       # Install dependencies
make train         # Run training pipeline
make serve         # Start inference service
make test          # Run tests
make docker-build  # Build Docker image
make docker-run    # Run Docker container
make clean         # Clean generated files
```

## 🛠️ Development

### Adding New Features

1. **Training Pipeline**: Modify `flows/train_flow.py`
2. **API Endpoints**: Add routes in `api/main.py`
3. **Model Service**: Extend `api/service.py`
4. **Monitoring**: Enhance `api/monitor.py`

### Code Structure

- **Flows**: Metaflow pipelines for training
- **API**: FastAPI application and services
- **Tests**: Unit and integration tests
- **Monitoring**: Prediction logging and drift detection

## 📦 Dependencies

Key dependencies:
- `scikit-learn`: Machine learning models
- `metaflow`: Workflow orchestration
- `mlflow`: Experiment tracking and model registry
- `fastapi`: Web framework for API
- `pydantic`: Data validation
- `pytest`: Testing framework

See `requirements.txt` for complete list.

## 🐛 Troubleshooting

### Model Not Loading

If the model fails to load:
1. Ensure training pipeline has been run: `make train`
2. Check that `model/model.pkl` exists
3. Verify MLflow registry has a registered model

### Port Already in Use

If port 8000 is already in use:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### MLflow Tracking Issues

If MLflow tracking fails:
1. Ensure `mlflow/mlruns/` directory exists
2. Check write permissions
3. Verify MLflow is installed: `pip install mlflow`

## 📄 License

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Add tests
4. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue in the repository.

