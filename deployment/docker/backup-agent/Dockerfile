# Multi-stage build for GenomeVault backup agent
# Provides encrypted backup/restore capabilities with cloud storage support

FROM alpine:3.18 AS builder

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    postgresql14-dev \
    python3-dev \
    py3-pip \
    openssl-dev \
    libffi-dev

# Install Python dependencies
COPY requirements-backup.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements-backup.txt

FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache \
    postgresql14-client \
    python3 \
    py3-pip \
    bash \
    curl \
    openssl \
    gnupg \
    aws-cli \
    gzip \
    bzip2 \
    xz \
    ca-certificates \
    tzdata \
    && rm -rf /var/cache/apk/*

# Install Google Cloud SDK
RUN curl -sSL https://sdk.cloud.google.com | bash && \
    mv /root/google-cloud-sdk /opt/google-cloud-sdk && \
    /opt/google-cloud-sdk/bin/gcloud components install gsutil --quiet

# Copy Python packages from builder
COPY --from=builder /usr/lib/python3.11/site-packages /usr/lib/python3.11/site-packages

# Create backup user and directories
RUN addgroup -g 1000 backup && \
    adduser -D -u 1000 -G backup backup && \
    mkdir -p /backups /scripts /var/log /etc/encryption && \
    chown -R backup:backup /backups /var/log

# Copy backup scripts
COPY scripts/backup_restore.sh /scripts/
COPY scripts/backup_monitor.py /scripts/
COPY scripts/dr_test.sh /scripts/
COPY scripts/encryption_utils.sh /scripts/

# Make scripts executable
RUN chmod +x /scripts/*.sh && \
    chmod +x /scripts/*.py

# Set environment variables
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    BACKUP_DIR=/backups \
    LOG_DIR=/var/log

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD [ -f "/var/log/backup.log" ] || exit 1

# Switch to backup user
USER backup

# Set working directory
WORKDIR /backups

# Entry point
ENTRYPOINT ["/bin/bash"]
CMD ["/scripts/backup.sh"]