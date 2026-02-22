# AI-Forge

> On-premise Vision MLOps Platform — End-to-End YOLO Training, Experiment Tracking & Inference Serving

---

## English

### Overview

AI-Forge is a fully self-hosted MLOps platform designed for Vision / VLM workloads. It connects a YOLO training pipeline, MLflow experiment tracking, MinIO artifact storage, a FastAPI inference API, and a Next.js UI — all orchestrated with Docker Compose and running entirely on your own hardware.

```
┌──────────────┐    train     ┌──────────────┐    register   ┌─────────────────┐
│ train_service│ ──────────▶  │    MLflow    │ ────────────▶ │  Model Registry │
│  (YOLO)      │              │  + MinIO     │               │  (best.pt)      │
└──────────────┘              └──────────────┘               └────────┬────────┘
                                                                      │ load
┌──────────────┐    POST      ┌──────────────┐                        ▼
│   UI / API   │ ──────────▶  │  inference   │ ◀─────── pyfunc model ─┘
│  (Next.js)   │              │  (FastAPI)   │
└──────────────┘              └──────────────┘
```

### Architecture & Services

| Container   | Image / Build         | Port | Role                                     |
|-------------|-----------------------|------|------------------------------------------|
| `postgres`  | `postgres:16`         | 5432 | MLflow metadata store                    |
| `minio`     | `minio/minio`         | 9000 | S3-compatible artifact store             |
| `mlflow`    | `./mlflow`            | 5000 | Experiment tracking + model registry     |
| `trainer`   | `./train_service`     | —    | YOLO training (attach & run manually)    |
| `inference` | `./inference_service` | 8000 | FastAPI detection API (`/api/v1/detect`) |
| `ui`        | `node:18`             | 3000 | Next.js frontend (dev mode)              |

### Prerequisites

- Docker ≥ 24 and Docker Compose v2
- NVIDIA drivers + `nvidia-container-toolkit` (optional, for GPU training)
- Git

### Quick Start

**1. Clone the repository**

```bash
git clone <repo-url> AI-Forge
cd AI-Forge
```

**2. Configure environment**

```bash
cp .env.example .env
# Edit .env — set secure passwords before first run
```

**3. Start the full stack**

```bash
docker compose up -d --build
```

**4. Verify services are healthy**

```bash
docker compose ps                    # all containers should show "healthy"
curl http://localhost:8000/health    # → {"status": "ok"}
curl http://localhost:5000/health    # → MLflow server reachable
```

**5. Access UIs**

| Service       | URL                        |
|---------------|----------------------------|
| MLflow UI     | http://localhost:5000      |
| MinIO Console | http://localhost:9001      |
| Inference API | http://localhost:8000/docs |
| Frontend      | http://localhost:3000      |

### Training a Model

Attach to the trainer container and run `train.py`:

```bash
docker exec -it trainer bash

# Inside the container:
python train.py \
  --model yolov8n.pt \
  --data /data/dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --experiment "my_experiment" \
  --run-name "baseline_v1" \
  --register yolo_detector
```

This logs parameters, metrics, and the best checkpoint to MLflow and registers the model under `yolo_detector/Production`. The inference service loads it automatically on the next request.

### Running Inference

```bash
# Base64-encode an image and call the detection API
IMAGE_B64=$(base64 -w 0 /path/to/image.jpg)

curl -s -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"${IMAGE_B64}\", \"options\": {}}" | jq .
```

Example response:

```json
{
  "success": true,
  "predictions": [
    {"label": "cat", "confidence": 0.94, "bbox": [10.0, 20.0, 300.0, 400.0]}
  ],
  "metrics": {
    "latency_ms": 42.1,
    "image_width": 640,
    "image_height": 480,
    "num_predictions": 1
  }
}
```

### Running Tests

Tests run **inside the Docker containers** per project convention:

```bash
# inference_service — 15 tests
docker exec inference pytest tests/ -v

# train_service — 10 tests
docker exec trainer pytest tests/ -v

# Both suites via the orchestrator script
bash scripts/test_runner.sh
```

### Project Structure

```
AI-Forge/
├── docker-compose.yaml        # Full stack orchestration
├── .env.example               # Environment template (copy → .env)
├── scripts/
│   └── test_runner.sh         # Test orchestrator
├── mlflow/
│   └── Dockerfile
├── train_service/
│   ├── train.py               # YOLO training + MLflow tracking
│   ├── requirements.txt
│   └── tests/
├── inference_service/
│   ├── app.py                 # FastAPI detection API
│   ├── requirements.txt
│   └── tests/
└── ui/
    └── package.json           # Next.js frontend
```

### Stopping the Stack

```bash
docker compose down      # stop containers (keep volumes)
docker compose down -v   # stop + remove all volumes
```

---

## 한국어

### 개요

AI-Forge는 Vision / VLM 워크로드를 위한 완전 자체 호스팅 MLOps 플랫폼입니다. YOLO 학습 파이프라인, MLflow 실험 추적, MinIO 아티팩트 저장소, FastAPI 추론 API, Next.js UI를 Docker Compose로 통합하여 자체 서버에서 완전히 운영할 수 있도록 설계되었습니다.

```
┌──────────────┐    학습      ┌──────────────┐    등록        ┌─────────────────┐
│ train_service│ ──────────▶  │    MLflow    │ ─────────────▶ │  모델 레지스트리 │
│  (YOLO)      │              │  + MinIO     │                │  (best.pt)      │
└──────────────┘              └──────────────┘                └────────┬────────┘
                                                                       │ 로드
┌──────────────┐    POST      ┌──────────────┐                         ▼
│   UI / API   │ ──────────▶  │  inference   │ ◀──────── pyfunc 모델 ──┘
│  (Next.js)   │              │  (FastAPI)   │
└──────────────┘              └──────────────┘
```

### 아키텍처 및 서비스

| 컨테이너    | 이미지 / 빌드         | 포트 | 역할                                        |
|-------------|-----------------------|------|---------------------------------------------|
| `postgres`  | `postgres:16`         | 5432 | MLflow 메타데이터 저장소                    |
| `minio`     | `minio/minio`         | 9000 | S3 호환 아티팩트 저장소                     |
| `mlflow`    | `./mlflow`            | 5000 | 실험 추적 및 모델 레지스트리                |
| `trainer`   | `./train_service`     | —    | YOLO 학습 (컨테이너 접속 후 수동 실행)      |
| `inference` | `./inference_service` | 8000 | FastAPI 탐지 API (`/api/v1/detect`)         |
| `ui`        | `node:18`             | 3000 | Next.js 프론트엔드 (개발 모드)              |

### 사전 요구사항

- Docker ≥ 24 및 Docker Compose v2
- NVIDIA 드라이버 + `nvidia-container-toolkit` (GPU 학습 시 선택 사항)
- Git

### 빠른 시작

**1. 저장소 클론**

```bash
git clone <repo-url> AI-Forge
cd AI-Forge
```

**2. 환경 설정**

```bash
cp .env.example .env
# .env 파일을 열어 비밀번호를 안전한 값으로 변경하세요
```

**3. 전체 스택 시작**

```bash
docker compose up -d --build
```

**4. 서비스 상태 확인**

```bash
docker compose ps                    # 모든 컨테이너가 "healthy" 상태인지 확인
curl http://localhost:8000/health    # → {"status": "ok"}
curl http://localhost:5000/health    # → MLflow 서버 응답 확인
```

**5. 웹 UI 접속**

| 서비스         | 주소                       |
|----------------|----------------------------|
| MLflow UI      | http://localhost:5000      |
| MinIO 콘솔     | http://localhost:9001      |
| 추론 API 문서  | http://localhost:8000/docs |
| 프론트엔드     | http://localhost:3000      |

### 모델 학습

trainer 컨테이너에 접속한 후 `train.py`를 실행합니다:

```bash
docker exec -it trainer bash

# 컨테이너 내부에서:
python train.py \
  --model yolov8n.pt \
  --data /data/dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --experiment "my_experiment" \
  --run-name "baseline_v1" \
  --register yolo_detector
```

학습이 완료되면 파라미터, 메트릭, best.pt 체크포인트가 MLflow에 기록되고, 모델이 `yolo_detector/Production`으로 레지스트리에 등록됩니다. inference 서비스는 다음 요청 시 이 모델을 자동으로 로드합니다.

### 추론 API 호출

```bash
# 이미지를 base64로 인코딩한 후 탐지 API 호출
IMAGE_B64=$(base64 -w 0 /path/to/image.jpg)

curl -s -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"${IMAGE_B64}\", \"options\": {}}" | jq .
```

응답 예시:

```json
{
  "success": true,
  "predictions": [
    {"label": "cat", "confidence": 0.94, "bbox": [10.0, 20.0, 300.0, 400.0]}
  ],
  "metrics": {
    "latency_ms": 42.1,
    "image_width": 640,
    "image_height": 480,
    "num_predictions": 1
  }
}
```

### 테스트 실행

테스트는 프로젝트 컨벤션에 따라 **Docker 컨테이너 내부에서** 실행합니다:

```bash
# inference_service — 15개 테스트
docker exec inference pytest tests/ -v

# train_service — 10개 테스트
docker exec trainer pytest tests/ -v

# 전체 테스트 일괄 실행 (스크립트 사용)
bash scripts/test_runner.sh
```

### 프로젝트 구조

```
AI-Forge/
├── docker-compose.yaml        # 전체 스택 오케스트레이션
├── .env.example               # 환경 변수 템플릿 (.env로 복사 후 사용)
├── scripts/
│   └── test_runner.sh         # 테스트 오케스트레이터
├── mlflow/
│   └── Dockerfile
├── train_service/
│   ├── train.py               # YOLO 학습 + MLflow 추적
│   ├── requirements.txt
│   └── tests/
├── inference_service/
│   ├── app.py                 # FastAPI 탐지 API
│   ├── requirements.txt
│   └── tests/
└── ui/
    └── package.json           # Next.js 프론트엔드
```

### 스택 종료

```bash
docker compose down      # 컨테이너 중지 (볼륨 유지)
docker compose down -v   # 컨테이너 중지 + 볼륨 전체 삭제
```
