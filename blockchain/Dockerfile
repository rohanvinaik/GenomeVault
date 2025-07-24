# Dockerfile for Local Blockchain
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy contract files
COPY contracts ./contracts
COPY scripts ./scripts
COPY test ./test
COPY hardhat.config.ts ./
COPY tsconfig.json ./

# Compile contracts
RUN npx hardhat compile

# Create deployments directory
RUN mkdir -p /app/deployments

# Expose Hardhat node port
EXPOSE 8545

# Default command is to run the node
CMD ["npx", "hardhat", "node", "--hostname", "0.0.0.0"]
