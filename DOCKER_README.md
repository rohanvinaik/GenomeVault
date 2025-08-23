# GenomeVault Docker Setup

This guide explains how to run GenomeVault using Docker Compose with PostgreSQL and Redis services.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB of available RAM
- 10GB of free disk space

## Quick Start

### 1. Production Setup

```bash
# Clone the repository
git clone <repository-url>
cd genomevault

# Copy environment variables
cp .env.example .env

# Edit .env file with your configuration
nano .env

# Start all services
docker-compose up -d

# Check service health
docker-compose ps
```

### 2. Development Setup

```bash
# Start with development overrides
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f api

# Access development tools:
# - API: http://localhost:8000
# - PostgreSQL Admin: http://localhost:8081 (admin@genomevault.dev / genomevault_pgadmin_dev)
# - Redis Commander: http://localhost:8082 (admin / genomevault_redis_commander_dev)
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin / genomevault)
```

## Services Overview

### Core Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| API | genomevault_api | 8000 | Main GenomeVault API service |
| PostgreSQL | genomevault_postgres | 5432 | Primary database |
| Redis | genomevault_redis | 6379 | Cache and session store |

### Development Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| pgAdmin | genomevault_pgadmin | 8081 | PostgreSQL web interface |
| Redis Commander | genomevault_redis_commander | 8082 | Redis web interface |
| Prometheus | prometheus | 9090 | Metrics collection |
| Grafana | grafana | 3000 | Monitoring dashboards |

## Environment Variables

Key environment variables in `.env` file:

```bash
# Database Configuration
POSTGRES_DB=genomevault
POSTGRES_USER=genomevault
POSTGRES_PASSWORD=your_secure_password

# Redis Configuration
REDIS_PASSWORD=your_redis_password

# API Configuration
API_PORT=8000
GENOMEVAULT_SECRET_KEY=your_secret_key
GENOMEVAULT_ENV=production
```

## Health Checks

All services include health checks. Check service status:

```bash
# View all service status
docker-compose ps

# Check specific service health
docker-compose exec api curl -f http://localhost:8000/v1/health
```

## Database Management

### Initial Setup

The database is automatically initialized with:
- Required extensions (uuid-ossp, pgcrypto, pg_stat_statements)
- Basic schema (genomevault, audit, metrics)
- Default admin user (development only)

### Development Data

In development mode, the database is seeded with:
- Sample users with different roles
- Test hypervector encodings
- Sample PIR queries and ZK proofs
- Mock API metrics and audit logs

### Access Database

```bash
# Using psql in container
docker-compose exec postgres psql -U genomevault -d genomevault

# Using pgAdmin (development)
# Navigate to http://localhost:8081
# Login: admin@genomevault.dev / genomevault_pgadmin_dev
```

## Volume Management

### Data Persistence

- `postgres_data`: PostgreSQL database files
- `redis_data`: Redis persistence files
- Application logs: `./logs` directory

### Backup Strategy

```bash
# Backup database
docker-compose exec postgres pg_dump -U genomevault -d genomevault > backup.sql

# Backup Redis
docker-compose exec redis redis-cli BGSAVE
docker-compose cp redis:/data/dump.rdb ./redis-backup.rdb
```

## Development Workflow

### Hot Reloading

The development setup includes:
- Source code mounted as volume
- Auto-reload on Python file changes
- Python debugger on port 5678

### Debugging

```bash
# Attach debugger (VS Code)
# Add configuration to launch.json:
{
    "name": "Attach to Docker",
    "type": "python",
    "request": "attach",
    "connect": {
        "host": "localhost",
        "port": 5678
    },
    "pathMappings": [
        {
            "localRoot": "${workspaceFolder}",
            "remoteRoot": "/app"
        }
    ]
}
```

### Testing

```bash
# Run tests in container
docker-compose exec api pytest

# Run specific test file
docker-compose exec api pytest tests/test_hypervector.py

# Run with coverage
docker-compose exec api pytest --cov=genomevault
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 5432, 6379, 8000 are available
2. **Memory issues**: Increase Docker memory limit to 4GB+
3. **Permission errors**: Check file ownership and Docker permissions

### Service Logs

```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f api
docker-compose logs -f postgres
docker-compose logs -f redis

# View last 100 lines
docker-compose logs --tail=100 api
```

### Reset Environment

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: This deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Clean up everything
docker system prune -a
```

## Security Considerations

### Production Deployment

1. **Environment Variables**: Use Docker secrets or external secret management
2. **Network Security**: Configure firewall rules and use reverse proxy
3. **SSL/TLS**: Enable HTTPS with proper certificates
4. **Database Security**: Use strong passwords and restrict access
5. **Container Security**: Keep images updated and scan for vulnerabilities

### Development Security

- Default passwords are used for convenience
- Services are exposed on all interfaces
- Debug mode is enabled
- Use only in trusted development environments

## Performance Tuning

### PostgreSQL

```sql
-- Monitor performance
SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;

-- Check connection stats
SELECT * FROM pg_stat_database WHERE datname = 'genomevault';
```

### Redis

```bash
# Monitor Redis performance
docker-compose exec redis redis-cli INFO stats
docker-compose exec redis redis-cli MONITOR
```

### API Performance

- Monitor response times at http://localhost:9091/metrics (dev)
- Check logs for performance warnings
- Use profiling tools in development mode

## Scaling

### Horizontal Scaling

```yaml
# In docker-compose.override.yml
services:
  api:
    deploy:
      replicas: 3
```

### Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## Support

For issues and questions:
- Check service logs first
- Review health check status
- Consult application documentation
- Submit issues with complete log output
