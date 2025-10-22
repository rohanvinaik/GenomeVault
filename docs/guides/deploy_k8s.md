# Kubernetes deployment

## Build & load image
```bash
# Option 1: use a registry (recommended)
docker build -t <REGISTRY>/genomevault/api:TAG .
docker push <REGISTRY>/genomevault/api:TAG
# Edit image in deploy/k8s/api-deployment.yaml

# Option 2: kind/minikube local loads
kind load docker-image genomevault/api:local
```

## Apply manifests
```bash
kubectl apply -f deploy/k8s/api-deployment.yaml
kubectl apply -f deploy/k8s/api-service.yaml
kubectl rollout status deploy/genomevault-api
kubectl get svc genomevault-api
```
