# AI-Forge — On-Premise Production Deployment Guide

This guide covers moving AI-Forge from a developer laptop to an on-premise server
that runs continuously in production. It assumes a single-host deployment using
Docker Compose; multi-host (Swarm / Kubernetes) patterns are noted where applicable.

---

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [Host OS Setup](#2-host-os-setup)
3. [Repository & Environment](#3-repository--environment)
4. [Securing Credentials](#4-securing-credentials)
5. [API Authentication](#5-api-authentication)
6. [Persistent Storage & Backups](#6-persistent-storage--backups)
7. [Nginx Reverse Proxy](#7-nginx-reverse-proxy)
8. [TLS / HTTPS](#8-tls--https)
9. [GPU Support](#9-gpu-support)
10. [MinIO Bucket Initialization](#10-minio-bucket-initialization)
11. [Artifact Lifecycle Policy](#11-artifact-lifecycle-policy)
12. [Firewall Rules](#12-firewall-rules)
13. [Starting & Monitoring the Stack](#13-starting--monitoring-the-stack)
14. [Upgrading](#14-upgrading)

---

## 1. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU       | 4 cores | 8+ cores    |
| RAM       | 16 GB   | 32+ GB      |
| Disk      | 200 GB SSD | 1 TB NVMe |
| GPU       | Optional | NVIDIA RTX 3090 / A6000 (CUDA 12) |

---

## 2. Host OS Setup

Ubuntu 22.04 LTS is recommended.

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker Engine (official method)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker Compose v2 is available
docker compose version  # should print "Docker Compose version v2.x.x"
```

For GPU training, install the NVIDIA Container Toolkit:

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access inside Docker
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

---

## 3. Repository & Environment

```bash
# Clone to a stable location — avoid home directories on shared systems
sudo mkdir -p /opt/ai-forge
sudo chown $USER:$USER /opt/ai-forge
git clone <repo-url> /opt/ai-forge
cd /opt/ai-forge

# Copy the environment template
cp .env.example .env
```

---

## 4. Securing Credentials

Edit `/opt/ai-forge/.env` and replace every `changeme` placeholder with a strong,
randomly generated value. Never commit the `.env` file.

```bash
# Generate random passwords (example using openssl)
openssl rand -hex 24   # → use for POSTGRES_PASSWORD
openssl rand -hex 24   # → use for MINIO_ROOT_PASSWORD
openssl rand -hex 24   # → use for GRAFANA_PASSWORD
```

Restrict file permissions so only the service user can read it:

```bash
chmod 600 /opt/ai-forge/.env
```

Key variables to set in production:

```dotenv
POSTGRES_PASSWORD=<strong-random-password>
MINIO_ROOT_PASSWORD=<strong-random-password>
GRAFANA_PASSWORD=<strong-random-password>

# API authentication (see §5)
INFERENCE_ADMIN_KEY=<strong-random-key>
INFERENCE_VIEWER_KEY=<strong-random-key>
```

---

## 5. API Authentication

The inference service supports optional API key authentication controlled by two
environment variables:

| Variable               | Role   | Grants access to            |
|------------------------|--------|-----------------------------|
| `INFERENCE_ADMIN_KEY`  | Admin  | `POST /api/v1/detect`       |
| `INFERENCE_VIEWER_KEY` | Viewer | Future read-only endpoints  |

When these variables are set in `.env`, clients must include the key in the
`X-API-Key` request header:

```bash
# Call the detection endpoint with authentication
curl -s -X POST http://your-server:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${INFERENCE_ADMIN_KEY}" \
  -d '{"image_base64": "'"$(base64 -w0 image.jpg)"'"}' | jq .
```

The `/health` endpoint is always accessible without a key (required for Docker
healthchecks and load balancer probes).

---

## 6. Persistent Storage & Backups

Docker Compose mounts the following host directories for stateful data:

| Directory        | Contents                          |
|------------------|-----------------------------------|
| `./postgres_data`| MLflow experiment metadata        |
| `./minio_data`   | Model artifacts (weights, exports)|

**Important**: Back up both directories regularly.

```bash
# Example: nightly rsync to a backup server
0 3 * * * rsync -az --delete /opt/ai-forge/postgres_data/ backup-host:/backups/ai-forge/postgres/
0 3 * * * rsync -az --delete /opt/ai-forge/minio_data/   backup-host:/backups/ai-forge/minio/
```

For PostgreSQL, prefer a proper dump over raw directory backup:

```bash
# Add to cron (runs inside the postgres container)
0 2 * * * docker exec postgres pg_dump -U mlflow mlflow | gzip > /backups/mlflow-$(date +%F).sql.gz
```

---

## 7. Nginx Reverse Proxy

Use Nginx to expose all services under a single domain with clean paths, and to
terminate TLS (see §8).

Install Nginx on the host:

```bash
sudo apt-get install -y nginx
```

Create `/etc/nginx/sites-available/ai-forge`:

```nginx
upstream mlflow_upstream    { server 127.0.0.1:5000; }
upstream inference_upstream { server 127.0.0.1:8000; }
upstream grafana_upstream   { server 127.0.0.1:3001; }
upstream ui_upstream        { server 127.0.0.1:3000; }

server {
    listen 80;
    server_name ai-forge.yourdomain.com;

    # Redirect all HTTP to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ai-forge.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/ai-forge.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ai-forge.yourdomain.com/privkey.pem;
    include             /etc/letsencrypt/options-ssl-nginx.conf;

    # Frontend UI
    location / {
        proxy_pass         http://ui_upstream;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }

    # MLflow tracking server
    location /mlflow/ {
        proxy_pass         http://mlflow_upstream/;
        proxy_set_header   Host              $host;
    }

    # Inference API (includes /docs for Swagger UI)
    location /api/ {
        proxy_pass         http://inference_upstream/api/;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        client_max_body_size 20M;  # allow large base64 image payloads
    }

    # Grafana dashboards
    location /grafana/ {
        proxy_pass         http://grafana_upstream/;
        proxy_set_header   Host              $host;
    }
}
```

Enable and test the configuration:

```bash
sudo ln -s /etc/nginx/sites-available/ai-forge /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 8. TLS / HTTPS

Use Let's Encrypt (Certbot) to obtain a free TLS certificate:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d ai-forge.yourdomain.com
# Follow the prompts — Certbot will also auto-configure renewal
```

Certbot installs a cron job for auto-renewal. Verify it:

```bash
sudo certbot renew --dry-run
```

For internal / air-gapped networks without Let's Encrypt, generate a self-signed
certificate or use your organisation's CA:

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
  -keyout /etc/ssl/private/ai-forge.key \
  -out /etc/ssl/certs/ai-forge.crt \
  -subj "/CN=ai-forge.internal"
```

---

## 9. GPU Support

To enable GPU access in the trainer container, add a `deploy` section to
`docker-compose.yaml`:

```yaml
# In the 'trainer' service definition:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Then pass `--device cuda:0` (or the GPU index) when running `train.py`:

```bash
docker exec -it trainer python train.py \
  --model yolov8n.pt --data /data/dataset.yaml \
  --device 0       # GPU 0
```

For multi-GPU training with YOLO:

```bash
python train.py --device 0,1   # use GPU 0 and 1
```

---

## 10. MinIO Bucket Initialization

The first time the stack starts, MLflow automatically creates the `mlflow` bucket
if it does not exist. To create it manually and configure the lifecycle policy in
one step, run the provided setup script after the stack is healthy:

```bash
# Start the stack
docker compose up -d --build

# Wait for MinIO to be ready
docker compose ps --filter "status=running" minio

# Initialize bucket and apply lifecycle policy
python scripts/setup_minio_lifecycle.py
```

---

## 11. Artifact Lifecycle Policy

The `scripts/setup_minio_lifecycle.py` script sets an S3-compatible expiration
rule that automatically deletes MLflow artifacts older than `ARTIFACT_RETENTION_DAYS`
days (default: 90).

```bash
# Use a custom retention window
ARTIFACT_RETENTION_DAYS=180 python scripts/setup_minio_lifecycle.py
```

After the policy is applied, you can verify it in the MinIO Console at
`http://your-server:9001` → **Buckets** → **mlflow** → **Lifecycle**.

---

## 12. Firewall Rules

Only expose the ports that external users need. Everything else should be blocked
at the host firewall:

```bash
# Using ufw (Uncomplicated Firewall)
sudo ufw default deny incoming
sudo ufw allow ssh        # keep SSH access
sudo ufw allow 80/tcp     # HTTP (redirects to HTTPS via nginx)
sudo ufw allow 443/tcp    # HTTPS
sudo ufw enable

# Verify — the following ports should NOT be reachable from outside the host:
# 5432 (Postgres), 9000/9001 (MinIO), 5000 (MLflow), 8000 (Inference), 3000 (UI), 3001 (Grafana)
# All internal traffic flows through Docker's bridge networks and the nginx proxy.
```

---

## 13. Starting & Monitoring the Stack

```bash
# Pull latest images and rebuild all services
docker compose pull
docker compose up -d --build

# View real-time logs from all services
docker compose logs -f

# Check container health status
docker compose ps

# Access Grafana dashboards at https://ai-forge.yourdomain.com/grafana/
# Default login: admin / <GRAFANA_PASSWORD from .env>
```

### Systemd service (auto-start on boot)

Create `/etc/systemd/system/ai-forge.service`:

```ini
[Unit]
Description=AI-Forge MLOps Stack
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ai-forge
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-forge
sudo systemctl start ai-forge
```

---

## 14. Upgrading

```bash
cd /opt/ai-forge

# Pull new code
git pull origin main

# Rebuild images with latest changes and restart
docker compose up -d --build

# Check logs for errors
docker compose logs -f --tail=100
```

For breaking schema changes in PostgreSQL (rare), you may need to run a migration
or back up data before the upgrade.

---

*Document maintained alongside the project. Update when infrastructure decisions change.*
