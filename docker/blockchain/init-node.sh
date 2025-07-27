#!/bin/sh
# Initialize blockchain node

set -e

echo "Initializing GenomeVault blockchain node..."

# Check if already initialized
if [ -f "/data/blockchain/config/config.toml" ]; then
    echo "Node already initialized, skipping..."
else
    echo "First time initialization..."

    # Initialize Tendermint (placeholder)
    tendermint init --home /data/blockchain

    # Copy custom genesis if provided
    if [ -f "/config/genesis.json" ]; then
        cp /config/genesis.json /data/blockchain/config/genesis.json
        echo "Custom genesis file copied"
    fi

    # Configure node based on environment variables
    if [ "$NODE_CLASS" = "full" ]; then
        echo "Configuring as full node..."
        # Additional full node configuration
    elif [ "$NODE_CLASS" = "light" ]; then
        echo "Configuring as light node..."
        # Light node configuration
    fi

    if [ "$SIGNATORY_STATUS" = "trusted" ]; then
        echo "Configuring as trusted signatory..."
        # Trusted signatory configuration
    fi
fi

echo "Node initialization complete"

# Execute the main command
exec "$@"
