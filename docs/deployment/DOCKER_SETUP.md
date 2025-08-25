# GenomeVault Docker Setup

This guide provides instructions for setting up GenomeVault in a containerized environment.

## Prerequisites

1. **Docker Desktop** (macOS/Windows) or **Docker Engine** (Linux)
   - Version 20.10 or higher
   - At least 4GB RAM allocated to Docker
   - At least 10GB available disk space

2. **Docker Compose**
   - Version 2.0 or higher (comes with Docker Desktop)

## Quick Start

### 1. Verify Docker Installation

```bash
# Check Docker is running
docker --version
docker ps

# Check Docker Compose
docker compose version
```

### 2. Choose Your Stack

GenomeVault provides multiple Docker Compose configurations:

- `docker-compose.yml` - Full production stack
- `docker-compose.dev.yml` - Development environment  
- `docker-compose.demo.yml` - Demo/testing environment
- `docker-compose.obsv.yml` - Observability stack (Prometheus/Grafana)

### 3. Development Environment (Recommended for testing)

```bash
# Start development services
docker compose -f docker-compose.dev.yml up -d

# Check service status
docker compose -f docker-compose.dev.yml ps

# View logs
docker compose -f docker-compose.dev.yml logs -f api
```

### 4. Demo Environment (For demonstrations)

```bash
# Start demo stack
docker compose -f docker-compose.demo.yml up -d

# Test API endpoint
curl http://localhost:8000/health
```

## Available Services

### Core Services
- **API Server** (`genomevault-api`) - Port 8000
  - FastAPI application
  - Authentication endpoints
  - HDC encoding endpoints
  - PIR query endpoints

- **Database** (`postgres`) - Port 5432
  - PostgreSQL with genomic data tables
  - Automated migrations

- **Cache** (`redis`) - Port 6379
  - Redis for session management
  - PIR query caching

### ZK Proof Services
- **ZK Prover** (`zk-prover`) - Port 8001
  - Circom/SnarkJS proof generation
  - Trusted setup management
  - Circuit compilation

### PIR Services
- **PIR Server** (`pir-server`) - Port 8002
  - Private information retrieval
  - Multi-server PIR protocol
  - XOR-based aggregation

### Monitoring (Optional)
- **Prometheus** - Port 9090
  - Metrics collection
  - Performance monitoring

- **Grafana** - Port 3000
  - Dashboards and visualization
  - Default login: admin/admin

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# Database
DATABASE_URL=postgresql://genomevault:secure_password@postgres:5432/genomevault

# Security
JWT_SECRET_KEY=your-super-secret-jwt-key-here
API_KEY_SECRET=your-api-key-secret-here

# Features
ENABLE_ZK_PROOFS=true
ENABLE_PIR=true
ENABLE_METAL_ACCELERATION=false  # Set to true on Apple Silicon

# Logging
LOG_LEVEL=INFO
ENABLE_AUDIT_LOGGING=true

# Development
DEBUG=false
RELOAD=true
```

### Production Environment Variables

```bash
# Additional production settings
HIPAA_COMPLIANCE=true
AUDIT_LOG_RETENTION_DAYS=2557  # 7 years
BACKUP_ENABLED=true
MONITORING_ENABLED=true
HSM_ENABLED=false  # Set to true when HSM is configured
```

## Service Health Checks

Each service includes health checks:

```bash
# Check all service health
docker compose ps

# Individual service health
curl http://localhost:8000/health      # API
curl http://localhost:8001/health      # ZK Prover  
curl http://localhost:8002/health      # PIR Server
```

## Development Workflow

### 1. Local Development with Docker

```bash
# Start services in background
docker compose -f docker-compose.dev.yml up -d postgres redis

# Run API locally (with hot reload)
uvicorn genomevault.api.main:app --reload --port 8000

# Run tests against containerized services
pytest tests/integration/
```

### 2. Full Containerized Development

```bash
# Build and start all services
docker compose -f docker-compose.dev.yml up --build

# Rebuild specific service
docker compose -f docker-compose.dev.yml up --build api

# Attach to running container for debugging
docker compose -f docker-compose.dev.yml exec api bash
```

## Data Persistence

### Volumes

The following data is persisted across container restarts:

- **Database data**: `genomevault_postgres_data`
- **Redis data**: `genomevault_redis_data` 
- **ZK circuits**: `genomevault_zk_data`
- **Logs**: `genomevault_logs`

### Backups

```bash
# Manual database backup
docker compose exec postgres pg_dump -U genomevault genomevault > backup.sql

# Restore from backup  
docker compose exec -T postgres psql -U genomevault genomevault < backup.sql
```

## Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Use different ports
   docker compose -f docker-compose.dev.yml -p genomevault-dev up
   ```

2. **Out of memory**
   ```bash
   # Increase Docker memory limit to 8GB minimum
   # Docker Desktop → Settings → Resources → Memory
   
   # Monitor memory usage
   docker stats
   ```

3. **Permission issues**
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER data/
   
   # Reset volumes
   docker compose down -v
   docker compose up -d
   ```

4. **Service won't start**
   ```bash
   # Check logs
   docker compose logs service-name
   
   # Restart specific service
   docker compose restart service-name
   
   # Rebuild from scratch
   docker compose down
   docker compose up --build
   ```

### Performance Optimization

```bash
# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Use multi-stage builds (already implemented in Dockerfiles)
docker compose build --parallel

# Prune unused resources
docker system prune -a
```

## Production Deployment

For production deployment, see:
- `docs/deploy_k8s.md` - Kubernetes deployment
- `deployment/kubernetes/` - K8s manifests
- `deployment/helm/` - Helm charts

## Security Considerations

### Development
- Default passwords are insecure - change before production
- API runs in debug mode - disable for production
- No TLS termination - use reverse proxy in production

### Production Checklist
- [ ] Change all default passwords
- [ ] Enable TLS/HTTPS
- [ ] Configure proper firewall rules
- [ ] Enable audit logging
- [ ] Set up monitoring and alerting
- [ ] Configure backup strategy
- [ ] Review and test disaster recovery

## Integration Testing

```bash
# Run full integration test suite
docker compose -f docker-compose.test.yml up --build --exit-code-from test

# Run specific test categories
docker compose exec api pytest tests/integration/test_api_integration.py -v
docker compose exec api pytest tests/e2e/ -v
```

This completes the Docker setup for GenomeVault. The system is now ready for containerized development and testing.