# inference_service/Dockerfile
# Base : python:3.10.19-slim  (Debian 13 trixie)
# pip  : 26.0.1
# Verified environment snapshot: 2026-02-23

FROM python:3.10.19-slim

WORKDIR /app

# Upgrade pip to the exact version used when requirements were captured
RUN pip install --no-cache-dir pip==26.0.1

# Copy and install all pinned dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code is bind-mounted at runtime via docker-compose.yaml volumes.
# No COPY of source here — allows live reload with uvicorn --reload in dev.
