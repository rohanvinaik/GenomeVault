#!/bin/bash
# GenomeVault Docker Fix Script
# Generated automatically by docker_debug.py

set -e

echo '🔧 Fixing GenomeVault Docker issues...'

# Create missing data directories
mkdir -p data/{cache,encrypted,input,output,processed,raw}
mkdir -p logs keys config

# Fix directory permissions
chmod 755 data logs keys config
chmod -R 755 data/*

# Create .env file if missing
if [ ! -f .env ]; then
  ./scripts/docker_compose_wrapper.sh setup
fi

# Start Docker Desktop if not running
if ! docker info >/dev/null 2>&1; then
  echo 'Starting Docker Desktop...'
  open -a Docker
  sleep 10
fi


echo '✅ Fix script complete!'
echo 'You can now try:'
echo '  ./scripts/docker_compose_wrapper.sh dev'
echo '  ./scripts/docker_compose_wrapper.sh status'