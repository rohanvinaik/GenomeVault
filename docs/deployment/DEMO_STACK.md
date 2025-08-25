# GenomeVault Demo Stack

Complete Docker Compose demo environment for GenomeVault.

## Quick Start

```bash
# Start the full demo stack
./demo_stack.sh up

# Run demo sequence
./demo_stack.sh demo

# View logs
./demo_stack.sh logs

# Stop and cleanup
./demo_stack.sh clean
```

## Components

### Core Services
- **API Server**: FastAPI application on port 8000
- **PostgreSQL**: Database with persistent storage
- **Redis**: Caching and task queue
- **ZK Prover**: Zero-knowledge proof service

### Monitoring
- **Prometheus**: Metrics collection on port 9090
- **Grafana**: Dashboards and visualization on port 3000

### Demo Runner
- **Automated Demo**: Tests all API endpoints
- **Performance Metrics**: Benchmarks HDC, ZK, PIR operations

## Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | - |
| API Docs | http://localhost:8000/docs | - |
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | - |

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │  API Server │    │  PostgreSQL │
│             │◄──►│   (FastAPI) │◄──►│  Database   │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
                           ▲
                           │
                   ┌─────────────┐    ┌─────────────┐
                   │    Redis    │    │ ZK Prover   │
                   │   (Cache)   │    │  Service    │
                   │             │    │             │
                   └─────────────┘    └─────────────┘
                           ▲
                           │
                   ┌─────────────┐    ┌─────────────┐
                   │ Prometheus  │    │   Grafana   │
                   │ (Metrics)   │◄──►│ (Dashboard) │
                   │             │    │             │
                   └─────────────┘    └─────────────┘
```

## Demo Flow

The automated demo tests the complete GenomeVault pipeline:

1. **HDC Compression**
   - Encodes genomic variants using hyperdimensional computing
   - Demonstrates extreme compression ratios (>1000×)

2. **Zero-Knowledge Proofs**
   - Generates proofs for variant presence
   - Verifies proofs without revealing data

3. **Private Information Retrieval**
   - Queries genomic database privately
   - Returns results without server learning query

4. **Performance Metrics**
   - Collects latency and throughput data
   - Displays results in Grafana dashboards

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GENOMEVAULT_ENV` | demo | Runtime environment |
| `DATABASE_URL` | postgres://... | Database connection |
| `REDIS_URL` | redis://redis:6379 | Redis connection |
| `SECRET_KEY` | demo_secret... | JWT secret (demo only) |
| `PROVER_MODE` | mock | ZK prover mode (mock/real) |

### Persistent Storage

- **PostgreSQL Data**: `postgres_data` volume
- **Redis Data**: `redis_data` volume  
- **Prometheus Data**: `prometheus_data` volume
- **Grafana Data**: `grafana_data` volume

## Development

### Building Images

```bash
# Build API image
docker compose -f docker-compose.demo.yml build api

# Build prover image
docker compose -f docker-compose.demo.yml build prover
```

### Running Individual Services

```bash
# Start just the database
docker compose -f docker-compose.demo.yml up postgres redis

# Start API only
docker compose -f docker-compose.demo.yml up api

# Run demo without full stack
docker compose -f docker-compose.demo.yml run --rm demo
```

### Debugging

```bash
# View service logs
docker compose -f docker-compose.demo.yml logs api
docker compose -f docker-compose.demo.yml logs prover

# Access database
docker compose -f docker-compose.demo.yml exec postgres psql -U genomevault

# Access Redis
docker compose -f docker-compose.demo.yml exec redis redis-cli
```

## Requirements

- Docker Desktop with Compose v2
- 4GB+ RAM available for containers
- Ports 3000, 8000, 9090 available

## Security Notes

⚠️ **This is a demo environment only**

- Uses weak passwords (`genomevault_demo_2024`)
- Exposes services on localhost
- No SSL/TLS encryption
- Mock authentication tokens

**Do not use in production**

## Troubleshooting

### Common Issues

1. **Port conflicts**: Stop services using ports 3000, 8000, 9090
2. **Memory issues**: Increase Docker memory limit to 4GB+
3. **Slow startup**: Wait for health checks to pass (~30s)

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check database connection  
docker compose -f docker-compose.demo.yml exec api python -c "
from genomevault.core.database import engine
print('DB connected' if engine else 'DB failed')
"
```

### Performance Tuning

For better demo performance:

```bash
# Allocate more memory to services
docker compose -f docker-compose.demo.yml up --scale api=2

# Use real ZK provers (requires Circom)
export PROVER_MODE=real
docker compose -f docker-compose.demo.yml up
```

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Report bugs in project issues
- **API Reference**: http://localhost:8000/docs when running