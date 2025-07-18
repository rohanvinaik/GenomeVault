# GenomeVault PIR Server Dockerfile
FROM python:3.10-slim-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Production stage
FROM python:3.10-slim-bullseye

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user
RUN useradd -m -u 1000 pirserver

# Create necessary directories
RUN mkdir -p /app /data/pir /data/reference /logs && \
    chown -R pirserver:pirserver /app /data /logs

# Copy application code
WORKDIR /app
COPY --chown=pirserver:pirserver . .

# Install the package
RUN pip install -e .

# Switch to non-root user
USER pirserver

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run PIR server
CMD ["python", "-m", "genomevault.pir.server.enhanced_pir_server", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--data-dir", "/data/pir", \
     "--reference-dir", "/data/reference"]
