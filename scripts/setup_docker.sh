#!/bin/bash
# GenomeVault Docker Setup Script
# Generated automatically by validate_docker_setup.py

set -e

echo '🐳 Setting up GenomeVault Docker environment...'

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo '❌ Docker not found. Please install Docker Desktop.'
    echo 'Visit: https://docs.docker.com/get-docker/'
    exit 1
fi

# Check Docker Compose
if ! docker compose version &> /dev/null && ! docker-compose --version &> /dev/null; then
    echo '❌ Docker Compose not found. Please install Docker Compose.'
    exit 1
fi

# Create data directories
mkdir -p data/{cache,encrypted,input,output,processed,raw}
mkdir -p logs
mkdir -p keys
mkdir -p config

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo '📝 Creating .env file...'
    cat > .env << 'EOF'
# GenomeVault Environment Variables
DATABASE_URL=postgresql://genomevault:secure_password@postgres:5432/genomevault
JWT_SECRET_KEY=change-this-secret-key-in-production
API_KEY_SECRET=change-this-api-key-secret
ENABLE_ZK_PROOFS=true
ENABLE_PIR=true
LOG_LEVEL=INFO
DEBUG=false
EOF
    echo '✅ Created .env file'
else
    echo '✅ .env file already exists'
fi

# Pull required images
echo '📦 Pulling Docker images...'
docker compose -f docker-compose.dev.yml pull

# Build custom images
echo '🔨 Building GenomeVault images...'
docker compose -f docker-compose.dev.yml build

echo '✅ Setup complete!'
echo ''
echo 'To start services:'
echo '  docker compose -f docker-compose.dev.yml up -d'
echo ''
echo 'To check status:'
echo '  docker compose -f docker-compose.dev.yml ps'
echo ''
echo 'To view logs:'
echo '  docker compose -f docker-compose.dev.yml logs -f api'