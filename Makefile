.PHONY: help install train serve test docker-build docker-run clean \
	k8s-setup k8s-deploy k8s-delete k8s-status k8s-logs k8s-port-forward \
	kind-create kind-delete kind-load-image

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make train           - Run training pipeline"
	@echo "  make serve           - Start FastAPI inference service"
	@echo "  make mlflow-ui       - Start MLflow UI (http://localhost:5000)"
	@echo "  make test            - Run tests"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo ""
	@echo "Kubernetes (kind) commands:"
	@echo "  make kind-create     - Create kind cluster"
	@echo "  make kind-delete     - Delete kind cluster"
	@echo "  make kind-load-image - Load Docker image into kind"
	@echo "  make k8s-setup       - Setup Kubernetes namespace and resources"
	@echo "  make k8s-deploy      - Deploy application to Kubernetes"
	@echo "  make k8s-delete      - Delete Kubernetes deployment"
	@echo "  make k8s-status      - Check deployment status"
	@echo "  make k8s-logs        - View pod logs"
	@echo "  make k8s-port-forward - Port forward to service"
	@echo ""
	@echo "  make clean           - Clean generated files"

install:
	@if [ ! -d .venv ]; then python3 -m venv .venv; fi
	. .venv/bin/activate && pip install --upgrade pip setuptools wheel && pip install --only-binary :all: -r requirements.txt || pip install -r requirements.txt

train:
	cd flows && python train_flow.py run

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	@echo "Starting MLflow UI..."
	@echo "Access at: http://localhost:5000"
	@if [ -d "flows/mlflow/mlruns" ] && [ -n "$$(ls -A flows/mlflow/mlruns 2>/dev/null)" ]; then \
		echo "Using MLflow data from: flows/mlflow/mlruns"; \
		. .venv/bin/activate && mlflow ui --backend-store-uri file:./flows/mlflow/mlruns --host 0.0.0.0 --port 5000; \
	elif [ -d "mlflow/mlruns" ] && [ -n "$$(ls -A mlflow/mlruns 2>/dev/null)" ]; then \
		echo "Using MLflow data from: mlflow/mlruns"; \
		. .venv/bin/activate && mlflow ui --backend-store-uri file:./mlflow/mlruns --host 0.0.0.0 --port 5000; \
	else \
		echo "No MLflow data found. Run 'make train' first to generate experiments."; \
		. .venv/bin/activate && mlflow ui --backend-store-uri file:./mlflow/mlruns --host 0.0.0.0 --port 5000; \
	fi

test:
	pytest tests/ -v --cov=api --cov=flows

docker-build:
	docker build -t house-price-api:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/mlflow:/app/mlflow -v $(PWD)/model:/app/model -v $(PWD)/logs:/app/logs house-price-api:latest

# Kubernetes (kind) commands
kind-create:
	@echo "Creating kind cluster..."
	kind create cluster --name ml-system-cluster --config k8s/kind-config.yaml || echo "Cluster may already exist"
	@echo "Setting up storage class for kind..."
	@kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.24/deploy/local-path-storage.yaml || echo "Storage provisioner may already exist"
	@sleep 5
	@kubectl patch storageclass standard -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}' || echo "Storage class may already be configured"

kind-delete:
	@echo "Deleting kind cluster..."
	kind delete cluster --name ml-system-cluster || echo "Cluster may not exist"

kind-load-image: docker-build
	@echo "Loading Docker image into kind cluster..."
	kind load docker-image house-price-api:latest --name ml-system-cluster

k8s-setup:
	@echo "Setting up Kubernetes namespace and resources..."
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/pvc.yaml
	@echo "Waiting for PVCs to be bound..."
	kubectl wait --for=condition=Bound pvc/model-pvc pvc/mlflow-pvc pvc/logs-pvc -n ml-system --timeout=60s || true

k8s-deploy: kind-load-image k8s-setup
	@echo "Deploying application to Kubernetes..."
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	@echo "Waiting for deployment to be ready..."
	kubectl wait --for=condition=available --timeout=120s deployment/house-price-api -n ml-system
	@echo "Deployment complete!"
	@echo "Service available at: http://localhost:30080"
	@echo "Run 'make k8s-status' to check status"

k8s-delete:
	@echo "Deleting Kubernetes resources..."
	kubectl delete -f k8s/service.yaml || true
	kubectl delete -f k8s/deployment.yaml || true
	kubectl delete -f k8s/pvc.yaml || true
	kubectl delete -f k8s/configmap.yaml || true
	kubectl delete -f k8s/namespace.yaml || true

k8s-status:
	@echo "=== Deployment Status ==="
	kubectl get deployments -n ml-system
	@echo ""
	@echo "=== Pod Status ==="
	kubectl get pods -n ml-system
	@echo ""
	@echo "=== Service Status ==="
	kubectl get services -n ml-system
	@echo ""
	@echo "=== PVC Status ==="
	kubectl get pvc -n ml-system

k8s-logs:
	@echo "Fetching logs from pods..."
	kubectl logs -f -l app=house-price-api -n ml-system --tail=50

k8s-port-forward:
	@echo "Port forwarding to service (Ctrl+C to stop)..."
	kubectl port-forward -n ml-system service/house-price-api-service 8000:8000

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".pytest_cache" -delete
	rm -rf logs/*.jsonl
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

