.PHONY: help install train serve test docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make train        - Run training pipeline"
	@echo "  make serve        - Start FastAPI inference service"
	@echo "  make test         - Run tests"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make clean        - Clean generated files"

install:
	pip install -r requirements.txt

train:
	cd flows && python train_flow.py run

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --cov=api --cov=flows

docker-build:
	docker build -t house-price-api:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/mlflow:/app/mlflow -v $(PWD)/model:/app/model -v $(PWD)/logs:/app/logs house-price-api:latest

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".pytest_cache" -delete
	rm -rf logs/*.jsonl
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

