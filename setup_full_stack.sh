#!/usr/bin/env bash

set -e
echo
echo "=========================================="
echo "  🚀 On Premise MLOps Auto Setup"
echo "=========================================="
echo

######
# 1) SYSTEM UPDATE + DOCKER INSTALL + COMPOSE
######
echo "🔹 System update & prerequisites..."
sudo apt update -y
sudo apt install -y \
    curl git wget apt-transport-https ca-certificates \
    software-properties-common

# Docker Install
if ! command -v docker >/dev/null 2>&1; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
fi

# Docker Compose Plugin
if ! docker compose version >/dev/null 2>&1; then
    echo "📦 Installing Docker Compose plugin..."
    sudo apt install -y docker-compose-plugin
fi

echo "✅ Docker installed"
echo

######
# 2) ENSURE DOCKER SERVICE
######
echo "🔹 Starting Docker..."
if ! sudo systemctl is-active --quiet docker; then
    sudo systemctl start docker
fi

echo "🐳 Docker is running!"
echo

######
# 3) CREATE PROJECT STRUCTURE
######
echo "📁 Creating directories..."
mkdir -p postgres_data minio_data mlflow train_service inference_service ui

echo "📁 Done"
echo

######
# 4) WRITE .env
######
cat > .env <<EOF
# -----------------------------
# 온프레미스 MLOps Stack
# -----------------------------

POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow123
POSTGRES_DB=mlflow
POSTGRES_PORT=5432

MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

MLFLOW_PORT=5000
MLFLOW_ARTIFACT_ROOT=s3://mlflow/

TRAIN_PORT=6006

INFER_PORT=8000

UI_PORT=3000
EOF

echo "📄 Created .env"
echo

######
# 5) CREATE mlflow Dockerfile
######
cat > mlflow/Dockerfile <<'EOF'
FROM python:3.10-slim

# MLflow + MinIO S3 deps
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir mlflow boto3 psycopg2-binary

WORKDIR /mlflow_code
EOF

echo "📄 mlflow Dockerfile done"
echo

######
# 6) TRAIN/INFERENCE DOCKERFILES
######
cat > train_service/Dockerfile <<'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
EOF

cat > inference_service/Dockerfile <<'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
EOF

echo "📄 Train & Inference Dockerfiles done"
echo

######
# 7) DEFAULT REQUIREMENTS
######
cat > train_service/requirements.txt <<'EOF'
mlflow
torch
torchvision
EOF

cat > inference_service/requirements.txt <<'EOF'
fastapi
uvicorn[standard]
torch
torchvision
mlflow
EOF

echo "📄 Requirements files created"
echo

######
# 8) BASIC INFERENCE APP
######
cat > inference_service/app.py <<'EOF'
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    return {"prediction": "TODO"}
EOF

echo "📄 Starter inference app created"
echo

######
# 9) DOCKER-COMPOSE (온프레미스 수정 버전)
######
cat > docker-compose.yaml <<'EOF'
version: "3.9"
services:

  postgres:
    image: postgres:16
    container_name: postgres
    working_dir: /var/lib/postgresql/data
    environment:
      POSTGRES_USER: \${POSTGRES_USER}
      POSTGRES_PASSWORD: \${POSTGRES_PASSWORD}
      POSTGRES_DB: \${POSTGRES_DB}
    volumes:
      - ./postgres_data:/var/lib/postgresql/data
    ports:
      - "\${POSTGRES_PORT}:5432"

  minio:
    image: minio/minio:latest
    container_name: minio
    working_dir: /data
    command: server /data --console-address \":9001\"
    environment:
      MINIO_ROOT_USER: \${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: \${MINIO_ROOT_PASSWORD}
    volumes:
      - ./minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  mlflow:
    build:
      context: ./mlflow
      dockerfile: Dockerfile
    container_name: mlflow
    working_dir: /mlflow_code
    ports:
      - "\${MLFLOW_PORT}:5000"
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: \${MINIO_ROOT_USER}
      AWS_SECRET_ACCESS_KEY: \${MINIO_ROOT_PASSWORD}
      MLFLOW_DEFAULT_ARTIFACT_ROOT: \${MLFLOW_ARTIFACT_ROOT}
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - postgres
      - minio
    volumes:
      - ./mlflow:/mlflow_code

  trainer:
    build:
      context: ./train_service
      dockerfile: Dockerfile
    container_name: trainer
    working_dir: /app
    command: bash -c "cd /app && tail -f /dev/null"
    volumes:
      - ./train_service:/app
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      AWS_ACCESS_KEY_ID: \${MINIO_ROOT_USER}
      AWS_SECRET_ACCESS_KEY: \${MINIO_ROOT_PASSWORD}
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    depends_on:
      - mlflow
      - minio
      - postgres

  inference:
    build:
      context: ./inference_service
      dockerfile: Dockerfile
    container_name: inference
    working_dir: /app
    command: bash -c "cd /app && uvicorn app:app --host 0.0.0.0 --port 8000"
    volumes:
      - ./inference_service:/app
    environment:
      MODEL_STORE: /app/models
    depends_on:
      - mlflow
      - minio

  ui:
    image: node:18
    container_name: ui
    working_dir: /app
    command: bash -c "cd /app && npm install && npm run dev"
    volumes:
      - ./ui:/app
    ports:
      - "\${UI_PORT}:3000"
EOF

echo "📄 docker-compose.yaml created"
echo

######
# 10) LAUNCH STACK
######
echo "🚀 Launching full on-premise stack..."
docker compose up -d --build

echo
echo "✅ STACK READY!"
echo " - MLflow: http://localhost:${MLFLOW_PORT}"
echo " - MinIO: http://localhost:9001"
echo " - Inference: http://localhost:${INFER_PORT}/health"
echo " - UI: http://localhost:${UI_PORT}"
echo

echo "📦 To see logs: docker compose logs -f"
echo