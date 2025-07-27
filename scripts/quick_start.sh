#!/bin/bash
# Quick-start script for GenomeVault development environment

set -e

echo "🧬 GenomeVault Quick Start 🧬"
echo "=============================="

# Check for required tools
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is not installed. Please install it first."
        exit 1
    fi
}

echo "Checking prerequisites..."
check_command docker
check_command docker-compose
echo "✅ All prerequisites installed"

# Function to wait for service
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=0

    echo -n "Waiting for $service_name..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -f -s $url > /dev/null 2>&1; then
            echo " ✅"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo " ❌ (timeout)"
    return 1
}

# Start services
echo ""
echo "Starting GenomeVault services..."
docker-compose -f docker/docker-compose.dev.yml up -d

# Wait for services to be ready
echo ""
echo "Waiting for services to initialize..."
wait_for_service "PostgreSQL" "http://localhost:5432" || true
wait_for_service "Redis" "http://localhost:6379" || true
wait_for_service "Local Blockchain" "http://localhost:8545" || true
wait_for_service "FastAPI" "http://localhost:8000/health" || true
wait_for_service "PIR Server 1" "http://localhost:9001/health" || true
wait_for_service "Prometheus" "http://localhost:9090" || true
wait_for_service "Grafana" "http://localhost:3000" || true

# Deploy smart contracts
echo ""
echo "Deploying smart contracts..."
docker-compose -f docker/docker-compose.dev.yml run --rm deploy-contracts

# Show status
echo ""
echo "=================================="
echo "✅ GenomeVault is ready!"
echo "=================================="
echo ""
echo "Services available at:"
echo "  • Main API:        http://localhost:8000"
echo "  • Documentation:   http://localhost:8000/docs"
echo "  • PIR Servers:     http://localhost:9001-9004"
echo "  • Blockchain:      http://localhost:8545"
echo "  • Prometheus:      http://localhost:9090"
echo "  • Grafana:         http://localhost:3000 (admin/genomevault)"
echo ""
echo "To view logs:"
echo "  docker-compose -f docker/docker-compose.dev.yml logs -f [service_name]"
echo ""
echo "To stop all services:"
echo "  docker-compose -f docker/docker-compose.dev.yml down"
echo ""
echo "Happy coding! 🚀"
