# syntax=docker/dockerfile:1

# ---- base runtime ----
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System deps for numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cache-friendly)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY . /app

# Expose port
EXPOSE 8000

# Healthcheck (simple HTTP GET)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \ 
  CMD python - <<'PY' \ 
import urllib.request, sys \ 
try: \ 
    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as r: \ 
        sys.exit(0 if r.status == 200 else 1) \ 
except Exception: \ 
    sys.exit(1) \ 
PY

# Default command: uvicorn
CMD ["uvicorn", "genomevault.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
