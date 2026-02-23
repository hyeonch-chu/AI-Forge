# mlflow/Dockerfile
# Base : python:3.10.19-slim  (Debian 13 trixie)
# pip  : 26.0.1
# Verified environment snapshot: 2026-02-23

FROM python:3.10.19-slim

# Upgrade pip to the exact version used when requirements were captured
RUN pip install --no-cache-dir pip==26.0.1

# Copy requirements to /tmp so the install is not shadowed by the
# ./mlflow:/mlflow_code bind-mount that docker-compose applies at runtime.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Runtime working directory — bind-mounted from ./mlflow at runtime.
WORKDIR /mlflow_code
