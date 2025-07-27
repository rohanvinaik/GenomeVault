# Dockerfile for PIR Server
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy PIR server code
COPY genomevault/pir /app/genomevault/pir
COPY genomevault/utils /app/genomevault/utils

# Create data directory
RUN mkdir -p /data

# Expose PIR server port
EXPOSE 9001

# Run PIR server
CMD ["python", "-m", "genomevault.pir.server"]
