# GenomeVault Blockchain Node Dockerfile
FROM golang:1.20-alpine as builder

# Install build dependencies
RUN apk add --no-cache make gcc musl-dev linux-headers git

# Set working directory
WORKDIR /build

# Copy go mod files (if using Go for blockchain)
# COPY go.mod go.sum ./
# RUN go mod download

# For now, we'll use a placeholder blockchain implementation
# In production, this would build the actual blockchain node
RUN echo "#!/bin/sh" > /build/genomevault-node && \
    echo "echo 'GenomeVault blockchain node starting...'" >> /build/genomevault-node && \
    echo "exec /usr/local/bin/tendermint node" >> /build/genomevault-node && \
    chmod +x /build/genomevault-node

# Production stage
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache ca-certificates jq curl bash

# Install Tendermint (placeholder for actual blockchain)
RUN wget https://github.com/tendermint/tendermint/releases/download/v0.35.9/tendermint_0.35.9_linux_amd64.tar.gz \
    && tar -xzf tendermint_0.35.9_linux_amd64.tar.gz \
    && mv tendermint /usr/local/bin/ \
    && rm tendermint_0.35.9_linux_amd64.tar.gz

# Copy blockchain binary
COPY --from=builder /build/genomevault-node /usr/local/bin/

# Create blockchain user
RUN adduser -D -u 1000 blockchain

# Create data directories
RUN mkdir -p /data/blockchain /config && \
    chown -R blockchain:blockchain /data /config

# Copy initialization scripts
COPY docker/blockchain/init-node.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/init-node.sh

# Switch to non-root user
USER blockchain

# Expose ports
EXPOSE 26656 26657 26660

# Volume for blockchain data
VOLUME ["/data/blockchain"]

# Initialize and run node
ENTRYPOINT ["/usr/local/bin/init-node.sh"]
CMD ["/usr/local/bin/genomevault-node"]
