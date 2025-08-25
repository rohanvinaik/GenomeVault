#!/bin/bash
set -e

echo "🧬 GenomeVault One-Command Demo Stack"
echo "====================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    echo "   Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker compose version &> /dev/null 2>&1; then
    echo "❌ Docker Compose v2 is required"
    echo "   Update Docker Desktop or install docker-compose-plugin"
    exit 1
fi

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Command
COMMAND=${1:-up}

case $COMMAND in
    up|start)
        echo -e "${BLUE}Starting GenomeVault demo stack...${NC}"
        
        # Build images if needed
        docker compose -f docker-compose.demo.yml build
        
        # Start stack
        docker compose -f docker-compose.demo.yml up -d
        
        echo -e "\n${GREEN}✅ Stack started!${NC}"
        echo ""
        echo "Waiting for services to be ready..."
        sleep 10
        
        # Check health
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ API is healthy${NC}"
        else
            echo -e "${YELLOW}⚠️  API still starting...${NC}"
        fi
        
        echo ""
        echo "Access points:"
        echo "  API:        http://localhost:8000"
        echo "  API Docs:   http://localhost:8000/docs"
        echo "  Grafana:    http://localhost:3000 (admin/admin)"
        echo "  Prometheus: http://localhost:9090"
        echo ""
        echo "View logs:"
        echo "  docker compose -f docker-compose.demo.yml logs -f"
        echo ""
        echo "Run demo:"
        echo "  docker compose -f docker-compose.demo.yml run demo"
        ;;
        
    down|stop)
        echo -e "${BLUE}Stopping GenomeVault demo stack...${NC}"
        docker compose -f docker-compose.demo.yml down
        echo -e "${GREEN}✅ Stack stopped${NC}"
        ;;
        
    logs)
        docker compose -f docker-compose.demo.yml logs -f
        ;;
        
    demo)
        echo -e "${BLUE}Running demo sequence...${NC}"
        docker compose -f docker-compose.demo.yml run --rm demo
        ;;
        
    clean)
        echo -e "${YELLOW}Cleaning up all resources...${NC}"
        docker compose -f docker-compose.demo.yml down -v
        echo -e "${GREEN}✅ Cleanup complete${NC}"
        ;;
        
    *)
        echo "Usage: $0 {up|down|logs|demo|clean}"
        echo ""
        echo "Commands:"
        echo "  up    - Start the demo stack"
        echo "  down  - Stop the demo stack"
        echo "  logs  - View logs"
        echo "  demo  - Run demo sequence"
        echo "  clean - Stop and remove all data"
        exit 1
        ;;
esac