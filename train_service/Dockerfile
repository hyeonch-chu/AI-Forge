# train_service/Dockerfile
# Base : python:3.10.19-slim  (Debian 13 trixie)
# pip  : 26.0.1
# Verified environment snapshot: 2026-02-23

FROM python:3.10.19-slim

# Install system libraries required by OpenCV (cv2) and Ultralytics.
# python:*-slim omits most shared libs; these are the ones cv2 needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxcb1 \
        libx11-6 \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip to the exact version used when requirements were captured
RUN pip install --no-cache-dir pip==26.0.1

# Copy and install all pinned dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code is bind-mounted at runtime via docker-compose.yaml volumes.
# No COPY of source here — allows live editing of train.py in dev.
