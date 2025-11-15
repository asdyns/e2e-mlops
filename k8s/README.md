# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the ML inference service to a Kubernetes cluster.

## Prerequisites

- [kind](https://kind.sigs.k8s.io/) installed (for local development)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) installed
- Docker installed and running

## Quick Start with kind

### 1. Create kind Cluster

```bash
make kind-create
# or
kind create cluster --name ml-system-cluster --config k8s/kind-config.yaml
```

### 2. Build and Load Docker Image

```bash
make kind-load-image
# or
make docker-build
kind load docker-image house-price-api:latest --name ml-system-cluster
```

### 3. Deploy to Kubernetes

```bash
make k8s-deploy
```

This will:
- Create the `ml-system` namespace
- Create ConfigMaps and PVCs
- Deploy the application
- Create the service

### 4. Access the Service

The service is exposed via NodePort on port 30080:

```bash
# Check status
make k8s-status

# Access API
curl http://localhost:30080/health
curl http://localhost:30080/docs

# Or use port forwarding
make k8s-port-forward
# Then access at http://localhost:8000
```

## Manual Deployment

If you prefer to deploy manually:

```bash
# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Create ConfigMap
kubectl apply -f k8s/configmap.yaml

# 3. Create PersistentVolumeClaims
kubectl apply -f k8s/pvc.yaml

# 4. Deploy application
kubectl apply -f k8s/deployment.yaml

# 5. Create service
kubectl apply -f k8s/service.yaml
```

## Using Kustomize

You can also use Kustomize to deploy:

```bash
kubectl apply -k k8s/
```

## Resource Files

- **namespace.yaml**: Creates the `ml-system` namespace
- **configmap.yaml**: Configuration for the API service
- **pvc.yaml**: PersistentVolumeClaims for model, MLflow, and logs storage
- **deployment.yaml**: Application deployment with 2 replicas
- **service.yaml**: NodePort service exposing the API
- **kind-config.yaml**: kind cluster configuration

## Monitoring

### Check Pod Status

```bash
kubectl get pods -n ml-system
```

### View Logs

```bash
make k8s-logs
# or
kubectl logs -f -l app=house-price-api -n ml-system
```

### Check Service

```bash
kubectl get svc -n ml-system
```

### Describe Resources

```bash
kubectl describe deployment house-price-api -n ml-system
kubectl describe pod <pod-name> -n ml-system
```

## Scaling

Scale the deployment:

```bash
kubectl scale deployment house-price-api --replicas=3 -n ml-system
```

## Updating the Deployment

After making changes to the code:

```bash
# 1. Rebuild Docker image
make docker-build

# 2. Load into kind
make kind-load-image

# 3. Restart deployment
kubectl rollout restart deployment/house-price-api -n ml-system

# 4. Watch rollout
kubectl rollout status deployment/house-price-api -n ml-system
```

## Cleanup

Delete all resources:

```bash
make k8s-delete
# or
kubectl delete -k k8s/
```

Delete kind cluster:

```bash
make kind-delete
# or
kind delete cluster --name ml-system-cluster
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n ml-system

# Check pod logs
kubectl logs <pod-name> -n ml-system
```

### PVC Not Binding

For kind, you may need to configure a default storage class:

```bash
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.24/deploy/local-path-storage.yaml
kubectl patch storageclass standard -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

### Image Pull Errors

Ensure the image is loaded into kind:

```bash
kind load docker-image house-price-api:latest --name ml-system-cluster
```

### Service Not Accessible

Check if the service is running:

```bash
kubectl get svc -n ml-system
kubectl get endpoints -n ml-system
```

## Production Considerations

For production deployments, consider:

1. **Ingress Controller**: Use Ingress instead of NodePort
2. **Secrets**: Store sensitive data in Kubernetes Secrets
3. **Resource Limits**: Adjust based on actual usage
4. **Autoscaling**: Add HorizontalPodAutoscaler
5. **Monitoring**: Integrate with Prometheus/Grafana
6. **Logging**: Use centralized logging (e.g., ELK stack)
7. **TLS/SSL**: Enable HTTPS with certificates
8. **Network Policies**: Add network policies for security

