# AI-Forge

> On-premise Vision MLOps Platform — End-to-End YOLO Training, Experiment Tracking & Inference Serving

---

## English

### Overview

AI-Forge is a fully self-hosted MLOps platform designed for Vision / VLM workloads.
It connects a YOLO training pipeline, MLflow experiment tracking, MinIO artifact storage,
a FastAPI inference API (with optional Claude VLM backend), and a Next.js UI —
all orchestrated with Docker Compose and running entirely on your own hardware.

```
┌──────────────┐    train     ┌──────────────┐    register   ┌─────────────────┐
│ train_service│ ──────────▶  │    MLflow    │ ────────────▶ │  Model Registry │
│  (YOLO)      │              │  + MinIO     │               │  (best.pt)      │
└──────────────┘              └──────────────┘               └────────┬────────┘
                                                                      │ load
┌──────────────┐    POST      ┌──────────────┐                        ▼
│   UI / API   │ ──────────▶  │  inference   │ ◀─────── pyfunc model ─┘
│  (Next.js)   │              │  (FastAPI)   │   or Claude VLM API
└──────────────┘              └──────────────┘

                 ┌──────────────────────────────────────────┐
                 │  Grafana ← Loki ← Promtail ← Docker logs │
                 └──────────────────────────────────────────┘
```

### Architecture & Services

| Container   | Image / Build         | Port(s)   | Role                                       |
|-------------|-----------------------|-----------|--------------------------------------------|
| `postgres`  | `postgres:16`         | 5432      | MLflow metadata store (backend network only) |
| `minio`     | `minio/minio`         | 9000/9001 | S3-compatible artifact store (backend only)  |
| `mlflow`    | `./mlflow`            | 5000      | Experiment tracking + model registry        |
| `trainer`   | `./train_service`     | —         | YOLO training (attach & run manually)       |
| `inference` | `./inference_service` | 8000      | FastAPI detection API + Claude VLM backend  |
| `ui`        | `node:18`             | 3000      | Next.js frontend (App Router)               |
| `loki`      | `grafana/loki:2.9`    | 3100      | Log aggregation backend                     |
| `promtail`  | `grafana/promtail:2.9`| —         | Collects Docker container logs → Loki       |
| `grafana`   | `grafana/grafana:10.2`| 3001      | Dashboards & log exploration                |

**Network isolation:** `postgres` and `minio` are on the `backend` network only —
they are not reachable from the `ui` or external requests.

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
# Edit .env — set secure passwords and (optionally) API keys before first run
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

**5. (Optional) Apply MinIO artifact lifecycle policy**

```bash
python scripts/setup_minio_lifecycle.py   # expires objects after 90 days by default
```

**6. Access UIs**

| Service        | URL                        |
|----------------|----------------------------|
| MLflow UI      | http://localhost:5000      |
| MinIO Console  | http://localhost:9001      |
| Inference API  | http://localhost:8000/docs |
| Frontend       | http://localhost:3000      |
| Grafana        | http://localhost:3001      |

### Training a Model

Attach to the trainer container and run `train.py`:

```bash
docker exec -it trainer bash

# Inside the container — train with YOLO (v8 / v10 / v11 variants supported):
python train.py \
  --model yolov8n.pt \
  --data /data/dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --experiment "my_experiment" \
  --run-name "baseline_v1" \
  --register yolo_detector \
  --export onnx            # optional: export to ONNX after training
```

This logs parameters, metrics, and the best checkpoint to MLflow and registers the
model under `yolo_detector/Production`. The inference service loads it automatically
on the next request.

### Running Inference

#### YOLO backend (default)

```bash
IMAGE_B64=$(base64 -w 0 /path/to/image.jpg)

curl -s -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"${IMAGE_B64}\", \"options\": {}}" | jq .
```

#### Claude VLM backend (optional)

Set `BACKEND=claude` and `ANTHROPIC_API_KEY=<your-key>` in `.env`, then restart
the inference service. The same API endpoint is used — no client changes required.

```bash
# With API authentication enabled:
curl -s -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${INFERENCE_ADMIN_KEY}" \
  -d "{\"image_base64\": \"${IMAGE_B64}\"}" | jq .
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
    "num_predictions": 1,
    "backend": "mlflow"
  }
}
```

### API Authentication

The inference service supports optional API key authentication.
Set the following in `.env` to enable it (leave empty to disable for local dev):

```dotenv
INFERENCE_ADMIN_KEY=<strong-random-key>   # required to call /api/v1/detect
INFERENCE_VIEWER_KEY=<strong-random-key>  # reserved for future read-only endpoints
```

Clients supply the key in the `X-API-Key` request header.
`GET /health` is always open (required for Docker healthchecks).

### Monitoring

Logs from all containers are collected by Promtail → Loki and visualised in Grafana.

- **Grafana:** http://localhost:3001 (login: `admin` / `GRAFANA_PASSWORD` from `.env`)
- JSON-structured logs from `inference_service` and `train_service` are auto-parsed
  and indexed by `level`, `logger`, and `service` labels for quick filtering.

### Running Tests

Tests run **inside the Docker containers** per project convention:

```bash
# inference_service — 29 tests (API, Claude backend, auth)
docker exec inference pytest tests/ -v

# train_service — YOLO/MLflow training tests
docker exec trainer pytest tests/ -v

# UI — Jest component tests (MetricsChart, Sidebar)
cd ui && npm run test:ci

# Both backend suites via the orchestrator script
bash scripts/test_runner.sh
```

### Project Structure

```
AI-Forge/
├── docker-compose.yaml        # Full stack orchestration (9 services)
├── .env.example               # Environment template (copy → .env)
├── docs/
│   └── deployment.md          # On-premise production deployment guide
├── scripts/
│   ├── setup_minio_lifecycle.py  # MinIO artifact lifecycle policy
│   ├── teardown.sh            # Clean stack removal
│   └── test_runner.sh         # Test orchestrator
├── mlflow/
│   └── Dockerfile
├── train_service/
│   ├── train.py               # YOLO training + MLflow tracking
│   ├── requirements.txt
│   └── tests/
├── inference_service/
│   ├── app.py                 # FastAPI detection API (auth + backend routing)
│   ├── backends/
│   │   └── claude_backend.py  # Optional Claude VLM backend
│   ├── requirements.txt
│   └── tests/
├── ui/
│   ├── package.json           # Next.js 13.4 App Router frontend
│   ├── jest.config.js         # Jest test configuration
│   └── src/
│       ├── app/               # Pages (/, /experiments, /models, /inference)
│       ├── components/        # Sidebar, MetricsChart
│       └── __tests__/         # Jest component tests
└── monitoring/
    ├── loki-config.yaml
    ├── promtail-config.yaml
    └── grafana/provisioning/  # Auto-provisioned Grafana datasources
```

### Stopping the Stack

```bash
docker compose down      # stop containers (keep volumes)
docker compose down -v   # stop + remove all volumes (destructive)
bash scripts/teardown.sh # alias for docker compose down -v --remove-orphans
```

---

## 한국어

### 개요

AI-Forge는 Vision / VLM 워크로드를 위한 완전 자체 호스팅 MLOps 플랫폼입니다.
YOLO 학습 파이프라인, MLflow 실험 추적, MinIO 아티팩트 저장소,
FastAPI 추론 API(Claude VLM 백엔드 선택 가능), Next.js UI를
Docker Compose로 통합하여 자체 서버에서 완전히 운영할 수 있도록 설계되었습니다.

```
┌──────────────┐    학습      ┌──────────────┐    등록        ┌─────────────────┐
│ train_service│ ──────────▶  │    MLflow    │ ─────────────▶ │  모델 레지스트리 │
│  (YOLO)      │              │  + MinIO     │                │  (best.pt)      │
└──────────────┘              └──────────────┘                └────────┬────────┘
                                                                       │ 로드
┌──────────────┐    POST      ┌──────────────┐                         ▼
│   UI / API   │ ──────────▶  │  inference   │ ◀──────── pyfunc 모델 ──┘
│  (Next.js)   │              │  (FastAPI)   │   또는 Claude VLM API
└──────────────┘              └──────────────┘

                 ┌──────────────────────────────────────────────────┐
                 │  Grafana ← Loki ← Promtail ← Docker 컨테이너 로그 │
                 └──────────────────────────────────────────────────┘
```

### 아키텍처 및 서비스

| 컨테이너    | 이미지 / 빌드         | 포트      | 역할                                        |
|-------------|-----------------------|-----------|---------------------------------------------|
| `postgres`  | `postgres:16`         | 5432      | MLflow 메타데이터 저장소 (backend 네트워크 전용) |
| `minio`     | `minio/minio`         | 9000/9001 | S3 호환 아티팩트 저장소 (backend 전용)          |
| `mlflow`    | `./mlflow`            | 5000      | 실험 추적 및 모델 레지스트리                  |
| `trainer`   | `./train_service`     | —         | YOLO 학습 (컨테이너 접속 후 수동 실행)          |
| `inference` | `./inference_service` | 8000      | FastAPI 탐지 API + Claude VLM 백엔드          |
| `ui`        | `node:18`             | 3000      | Next.js 프론트엔드 (App Router)              |
| `loki`      | `grafana/loki:2.9`    | 3100      | 로그 집계 백엔드                              |
| `promtail`  | `grafana/promtail:2.9`| —         | Docker 컨테이너 로그 수집 → Loki              |
| `grafana`   | `grafana/grafana:10.2`| 3001      | 대시보드 및 로그 탐색                          |

**네트워크 격리:** `postgres`와 `minio`는 `backend` 네트워크에만 연결되어 있어
`ui` 또는 외부에서 직접 접근할 수 없습니다.

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
# .env 파일을 열어 비밀번호와 API 키를 안전한 값으로 변경하세요
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

**5. (선택 사항) MinIO 아티팩트 수명 주기 정책 적용**

```bash
python scripts/setup_minio_lifecycle.py   # 기본값: 90일 후 객체 자동 삭제
```

**6. 웹 UI 접속**

| 서비스         | 주소                       |
|----------------|----------------------------|
| MLflow UI      | http://localhost:5000      |
| MinIO 콘솔     | http://localhost:9001      |
| 추론 API 문서  | http://localhost:8000/docs |
| 프론트엔드     | http://localhost:3000      |
| Grafana        | http://localhost:3001      |

### 모델 학습

trainer 컨테이너에 접속한 후 `train.py`를 실행합니다:

```bash
docker exec -it trainer bash

# 컨테이너 내부에서 (YOLOv8/v10/v11 다양한 변형 지원):
python train.py \
  --model yolov8n.pt \
  --data /data/dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --experiment "my_experiment" \
  --run-name "baseline_v1" \
  --register yolo_detector \
  --export onnx            # 선택 사항: 학습 후 ONNX 형식으로 내보내기
```

학습이 완료되면 파라미터, 메트릭, best.pt 체크포인트가 MLflow에 기록되고,
모델이 `yolo_detector/Production`으로 레지스트리에 등록됩니다.
inference 서비스는 다음 요청 시 이 모델을 자동으로 로드합니다.

### 추론 API 호출

#### YOLO 백엔드 (기본값)

```bash
IMAGE_B64=$(base64 -w 0 /path/to/image.jpg)

curl -s -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"${IMAGE_B64}\", \"options\": {}}" | jq .
```

#### Claude VLM 백엔드 (선택 사항)

`.env`에서 `BACKEND=claude`와 `ANTHROPIC_API_KEY=<키>`를 설정하고
inference 서비스를 재시작하면 동일한 API 엔드포인트를 통해 Claude를 사용할 수 있습니다.

```bash
# API 인증이 활성화된 경우:
curl -s -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${INFERENCE_ADMIN_KEY}" \
  -d "{\"image_base64\": \"${IMAGE_B64}\"}" | jq .
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
    "num_predictions": 1,
    "backend": "mlflow"
  }
}
```

### API 인증

inference 서비스는 선택적 API 키 인증을 지원합니다.
`.env`에서 아래 값을 설정하면 활성화됩니다 (로컬 개발 시 빈 문자열로 두면 비활성화):

```dotenv
INFERENCE_ADMIN_KEY=<강력한-랜덤-키>   # POST /api/v1/detect 호출 시 필요
INFERENCE_VIEWER_KEY=<강력한-랜덤-키>  # 향후 읽기 전용 엔드포인트용
```

클라이언트는 `X-API-Key` 요청 헤더에 키를 포함해야 합니다.
`GET /health`는 항상 인증 없이 접근 가능합니다 (Docker 헬스체크 용도).

### 모니터링

모든 컨테이너의 로그는 Promtail → Loki로 수집되어 Grafana에서 시각화됩니다.

- **Grafana:** http://localhost:3001 (로그인: `admin` / `.env`의 `GRAFANA_PASSWORD`)
- `inference_service`와 `train_service`의 JSON 구조화 로그는 `level`, `logger`,
  `service` 레이블로 자동 파싱되어 빠른 필터링이 가능합니다.

### 테스트 실행

테스트는 프로젝트 컨벤션에 따라 **Docker 컨테이너 내부에서** 실행합니다:

```bash
# inference_service — 29개 테스트 (API, Claude 백엔드, 인증)
docker exec inference pytest tests/ -v

# train_service — YOLO/MLflow 학습 테스트
docker exec trainer pytest tests/ -v

# UI — Jest 컴포넌트 테스트 (MetricsChart, Sidebar)
cd ui && npm run test:ci

# 전체 백엔드 테스트 일괄 실행 (스크립트 사용)
bash scripts/test_runner.sh
```

### 프로젝트 구조

```
AI-Forge/
├── docker-compose.yaml           # 전체 스택 오케스트레이션 (9개 서비스)
├── .env.example                  # 환경 변수 템플릿 (.env로 복사 후 사용)
├── docs/
│   └── deployment.md             # 온프레미스 프로덕션 배포 가이드
├── scripts/
│   ├── setup_minio_lifecycle.py  # MinIO 아티팩트 수명 주기 정책 설정
│   ├── teardown.sh               # 스택 완전 삭제 스크립트
│   └── test_runner.sh            # 테스트 오케스트레이터
├── mlflow/
│   └── Dockerfile
├── train_service/
│   ├── train.py                  # YOLO 학습 + MLflow 추적
│   ├── requirements.txt
│   └── tests/
├── inference_service/
│   ├── app.py                    # FastAPI 탐지 API (인증 + 백엔드 라우팅)
│   ├── backends/
│   │   └── claude_backend.py     # 선택적 Claude VLM 백엔드
│   ├── requirements.txt
│   └── tests/
├── ui/
│   ├── package.json              # Next.js 13.4 App Router 프론트엔드
│   ├── jest.config.js            # Jest 테스트 구성
│   └── src/
│       ├── app/                  # 페이지 (/, /experiments, /models, /inference)
│       ├── components/           # Sidebar, MetricsChart
│       └── __tests__/            # Jest 컴포넌트 테스트
└── monitoring/
    ├── loki-config.yaml
    ├── promtail-config.yaml
    └── grafana/provisioning/     # Grafana 데이터소스 자동 프로비저닝
```

### 스택 종료

```bash
docker compose down      # 컨테이너 중지 (볼륨 유지)
docker compose down -v   # 컨테이너 중지 + 볼륨 전체 삭제 (데이터 파괴)
bash scripts/teardown.sh # docker compose down -v --remove-orphans 단축 명령
```
