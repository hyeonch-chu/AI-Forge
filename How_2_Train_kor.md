# AI-Forge UI에서 모델 학습하기

이 가이드는 데이터셋을 준비하고, 스택을 설정하고, AI-Forge 웹 UI에서 YOLO 객체 검출 모델을 학습하는 전체 과정을 단계별로 안내합니다.

---

## 목차

1. [사전 준비](#1-사전-준비)
2. [데이터셋 준비](#2-데이터셋-준비)
3. [공개 데이터셋으로 빠른 시작 — Pascal VOC](#3-공개-데이터셋으로-빠른-시작--pascal-voc)
4. [학습 컨테이너에 데이터셋 연결하기](#4-학습-컨테이너에-데이터셋-연결하기)
5. [스택 시작](#5-스택-시작)
6. [학습 페이지 열기](#6-학습-페이지-열기)
7. [학습 폼 작성](#7-학습-폼-작성)
8. [실시간 로그 모니터링](#8-실시간-로그-모니터링)
9. [MLflow에서 결과 확인](#9-mlflow에서-결과-확인)
10. [학습된 모델을 추론에 활용하기](#10-학습된-모델을-추론에-활용하기)
11. [모델 내보내기(Export) 옵션](#11-모델-내보내기export-옵션)
12. [GPU 학습](#12-gpu-학습)
13. [문제 해결](#13-문제-해결)

---

## 1. 사전 준비

학습을 시작하기 전에 아래 항목을 모두 확인하세요.

| 항목 | 확인 방법 |
|---|---|
| Docker + Docker Compose 설치됨 | `docker compose version` |
| AI-Forge 전체 스택 빌드 및 실행 중 | `docker compose ps` — 모든 서비스 `healthy` 상태 |
| Ultralytics YAML 형식의 데이터셋 준비 | [2번 섹션](#2-데이터셋-준비) 참고 |
| 학습 컨테이너에서 데이터셋 접근 가능 | [3번 섹션](#3-학습-컨테이너에-데이터셋-연결하기) 참고 |

학습 과정에서 사용하는 서비스:

| 서비스 | URL | 역할 |
|---|---|---|
| **AI-Forge UI** | `http://localhost:3000` | 학습 폼 및 로그 뷰어 |
| **MLflow UI** | `http://localhost:5000` | 실험 및 메트릭 추적 |

---

## 2. 데이터셋 준비

학습 파이프라인은 **Ultralytics 데이터셋 YAML 형식**을 사용합니다.

### 디렉토리 구조

호스트 머신에서 아래와 같이 데이터셋을 구성하세요:

```
data/
├── dataset.yaml          ← UI에서 지정할 설정 파일
├── images/
│   ├── train/            ← 학습용 이미지 (.jpg / .png)
│   └── val/              ← 검증용 이미지
└── labels/
    ├── train/            ← YOLO 라벨 파일 (.txt, 이미지당 1개)
    └── val/
```

### dataset.yaml 형식

```yaml
# dataset.yaml — Ultralytics 형식
path: /data            # 학습 컨테이너 내부의 루트 경로 (절대 경로)
train: images/train    # 'path' 기준 상대 경로
val:   images/val

nc: 3                  # 클래스 수
names:
  0: 고양이
  1: 강아지
  2: 사람
```

> **핵심 규칙:** `path` 필드는 **학습 컨테이너 내부의 절대 경로**여야 합니다 — 호스트 경로가 아닙니다.
> 호스트의 `./data` 폴더를 컨테이너의 `/data`로 마운트했다면 (3번 섹션 참고), `path: /data`로 설정합니다.

### 라벨 형식 (이미지당 .txt 파일 1개)

각 줄 = 객체 1개:

```
<클래스_ID> <중심_x> <중심_y> <너비> <높이>
```

모든 좌표는 이미지 크기 기준 **정규화된 값** (0.0 – 1.0)입니다. 예시:

```
0 0.512 0.374 0.210 0.305
1 0.128 0.680 0.095 0.140
```

---

## 3. 공개 데이터셋으로 빠른 시작 — Pascal VOC

아직 데이터셋이 없다면, 레포지토리에 **Pascal VOC** 용 YAML 파일이 준비되어 있습니다 — 20개 클래스를 포함하는 고전적인 객체 검출 벤치마크 데이터셋입니다 (~2.8 GB).

> **참고 문서:** [Ultralytics VOC Dataset Docs](https://docs.ultralytics.com/datasets/detect/voc/)

### Pascal VOC 데이터셋이란?

| 항목 | 내용 |
|---|---|
| 클래스 수 | 20개 (비행기, 자전거, 새, 보트, 병, 버스, 자동차, 고양이, 의자, 소, 식탁, 개, 말, 오토바이, 사람, 화분, 양, 소파, 기차, TV) |
| 학습 이미지 | 16,551장 (VOC2007 + VOC2012 train/val 합산) |
| 검증 이미지 | 4,952장 (VOC2007 test 분할) |
| 어노테이션 형식 | Pascal VOC XML → YOLO .txt 형식으로 자동 변환 |
| 다운로드 크기 | ~2.8 GB (Ultralytics GitHub assets에서 ZIP 3개) |

### 파일 위치

```
data/VOC.yaml          ← 레포지토리 내 (호스트)
/data/VOC.yaml         ← 학습 컨테이너 내부 (볼륨 마운트 후)
```

### 사용 방법

**1. `docker-compose.yaml`에 데이터 볼륨 마운트가 이미 설정되어 있습니다:**

```yaml
trainer:
  volumes:
    - "./train_service:/app"
    - "./data:/data"          # 이미 설정됨 — data/VOC.yaml이 /data/VOC.yaml로 노출됨
```

**2. 트레이너 서비스 재빌드:**

```bash
docker compose up -d --build trainer
```

**3. YAML 파일이 보이는지 확인:**

```bash
docker exec trainer cat /data/VOC.yaml | head -5
```

**4. 학습 페이지** (`http://localhost:3000/training`)를 열고 아래 값을 입력합니다:

| 항목 | 값 |
|---|---|
| **Dataset YAML** | `/data/VOC.yaml` |
| **Model** | `yolov8n.pt` (첫 실행에 가장 빠름) |
| **Epochs** | `10` (파이프라인 동작 확인에 충분) |
| **Experiment Name** | `voc_quickstart` |

**5. Start Training 클릭.**

Ultralytics가 `/data/VOC` 디렉토리가 없는 것을 감지하고, 내장된 다운로드 + 변환 스크립트를 자동으로 실행한 뒤 학습을 시작합니다. 다운로드 진행 상황이 실시간 로그 뷰어에 표시됩니다.

> 첫 번째 실행 시 아래와 같은 로그가 나타납니다:
> ```
> Downloading VOCtrainval_06-Nov-2007.zip to /data/VOC/tmp ...
> VOC2012/train: 100%|████| 8218/8218
> VOC2007/train: 100%|████| 2501/2501
> ```
> 이후 실행에서는 다운로드를 건너뜁니다.

---

## 4. 학습 컨테이너에 데이터셋 연결하기

학습 컨테이너는 내부의 `/data` 경로에서 데이터셋을 읽습니다. 이 경로는 `docker-compose.yaml`에서 호스트의 `./data/` 폴더로 이미 연결되어 있습니다:

```yaml
trainer:
  volumes:
    - "./train_service:/app"
    - "./data:/data"        # ← 이미 설정됨
```

**자신의 데이터셋을 추가하려면**, 호스트의 `./data/` 폴더에 데이터셋을 넣으면 됩니다:

```
AI-Forge/
└── data/
    ├── VOC.yaml              ← 레포지토리에 포함됨
    ├── my_dataset.yaml       ← 여기에 추가
    └── my_dataset/
        ├── images/
        └── labels/
```

컨테이너 경로 `/data`는 `dataset.yaml`의 `path:` 필드와 일치해야 합니다.
새 데이터셋 추가 후 한 번 재빌드하면 변경 사항이 반영됩니다:

```bash
docker compose up -d --build trainer
```

### 방법 B — 실행 중인 컨테이너에 데이터 복사 (빠른 테스트용)

`docker-compose.yaml`을 수정하지 않고 파일을 직접 복사할 수도 있습니다:

```bash
docker cp ./data/. trainer:/data/
```

> 이 방법은 임시입니다 — 컨테이너가 재생성되면 데이터가 사라집니다. 반복 학습에는 방법 A를 사용하세요.

### 데이터 확인

```bash
docker exec trainer ls /data
# 예상 출력: dataset.yaml  images/  labels/
```

---

## 5. 스택 시작

스택이 아직 실행되지 않은 경우:

```bash
docker compose up -d --build
```

모든 서비스가 정상 상태인지 확인:

```bash
docker compose ps
```

모든 서비스가 `(healthy)` 또는 `Up` 상태여야 합니다. 트레이너 서비스는 uvicorn 초기화에 15–20초 정도 소요될 수 있습니다.

---

## 6. 학습 페이지 열기

1. 브라우저에서 `http://localhost:3000`으로 접속합니다
2. 왼쪽 사이드바에서 **Training**을 클릭합니다 (또는 대시보드의 **Training** 카드 클릭)
3. 왼쪽에는 학습 설정 폼, 오른쪽에는 작업 상태 패널이 표시됩니다

---

## 7. 학습 폼 작성

### 각 항목 설명

#### Dataset YAML * (필수)

```
/data/dataset.yaml
```

**학습 컨테이너 내부에서 보이는 dataset.yaml의 절대 경로**를 입력합니다.
가장 중요한 항목으로, 경로가 잘못되면 학습이 즉시 실패합니다.

> 팁: 제출 전에 `docker exec trainer ls /data`를 실행하여 경로를 확인하세요.

---

#### Model (모델)

드롭다운에서 YOLO 프리셋을 선택합니다. Ultralytics가 첫 사용 시 가중치를 자동으로 다운로드합니다.

| 크기 | 프리셋 | 설명 |
|------|--------|------|
| Nano | `yolov8n.pt` | 가장 빠름, 정확도 낮음 — 빠른 실험에 적합 |
| Small | `yolov8s.pt` | 엣지 디바이스에 적합한 균형 |
| Medium | `yolov8m.pt` | 대부분의 사용 사례에 권장 |
| Large | `yolov8l.pt` | 높은 정확도, 더 많은 VRAM 필요 |
| XLarge | `yolov8x.pt` | 최고 정확도, 가장 느림 |

YOLOv10 (`yolov10n/s/m/b/l/x.pt`) 및 YOLO11 (`yolo11n/s/m/l/x.pt`) 변형도 사용 가능합니다.

> 첫 실행 시에는 **`yolov8n.pt`** (nano)로 데이터셋이 올바르게 로드되는지 확인한 후, 긴 학습 작업을 진행하세요.

---

#### Epochs (에포크)

학습 데이터셋을 몇 번 완전히 반복할지 설정합니다.

| 용도 | 권장 값 |
|---|---|
| 빠른 검증 | 3–5 |
| 일반 학습 | 50–100 |
| 특정 도메인 파인튜닝 | 100–300 |

기본값: `50`

---

#### Image Size (이미지 크기)

모델이 학습 및 추론할 입력 해상도 (정사각형, 픽셀). 32의 배수여야 합니다.

| 값 | 용도 |
|---|---|
| `320` | 빠른 실험, 리소스 제한 환경 |
| `640` | 표준 — 속도와 정확도의 최적 균형 |
| `1280` | 고해상도 데이터셋 (더 많은 VRAM 필요) |

기본값: `640`

---

#### Batch Size (배치 크기)

한 번의 그래디언트 업데이트에 처리할 이미지 수. 배치가 클수록 빠르지만 더 많은 메모리가 필요합니다.

| 디바이스 | 실용적인 범위 |
|---|---|
| CPU | 4–16 |
| GPU (8 GB VRAM) | 16–32 |
| GPU (24 GB VRAM) | 64–128 |

로그에서 **메모리 부족(OOM) 오류**가 발생하면 배치 크기를 줄이세요.
기본값: `16`

---

#### Device (디바이스)

| 값 | 의미 |
|---|---|
| `cpu` | CPU 사용 (느리지만 모든 환경에서 동작) |
| `0` | GPU 0번 사용 (첫 번째 GPU) |
| `1` | GPU 1번 사용 (두 번째 GPU) |
| `0,1` | GPU 0, 1번 멀티 GPU 학습 |

기본값: `cpu`

> GPU 설정 방법은 [11번 섹션](#11-gpu-학습)을 참고하세요.

---

#### Export Format (내보내기 형식)

학습 후 선택적으로 모델을 다른 형식으로 변환합니다. 기본 `.pt` 체크포인트는 항상 저장되며, 이 옵션은 추가적인 배포 최적화 아티팩트를 생성합니다.

| 형식 | 용도 | 필요 조건 |
|---|---|---|
| `— none —` | 내보내기 없음 (기본값) | — |
| `onnx` | 범용 서빙 (ONNX Runtime, OpenVINO) | 없음 |
| `engine` | NVIDIA TensorRT (가장 빠른 GPU 추론) | NVIDIA GPU + TensorRT |
| `torchscript` | 이식성 있는 PyTorch 형식 | 없음 |
| `coreml` | Apple 디바이스 (iOS, macOS) | macOS |
| `saved_model` | TensorFlow / TFLite 서빙 | 없음 |

---

#### Experiment Name (실험 이름)

MLflow는 학습 실행(run)을 실험(experiment) 단위로 그룹화합니다. 나중에 필터링할 수 있도록 설명적인 이름을 사용하세요.

예시: `coco_기준선`, `커스텀_탐지기_v1`, `yolov10_비교`

기본값: `yolo_training`

---

#### Run Name (실행 이름, 선택사항)

실험 내에서 이 특정 실행에 붙이는 사람이 읽기 쉬운 레이블입니다.

예시: `배치32-lr0.01`, `증강-강화`, `사전학습-yolov8m`

비워두면 MLflow가 자동으로 이름을 생성합니다.

---

#### Register Model As (모델 등록 이름, 선택사항)

입력하면 학습 완료 후 최적 체크포인트(`best.pt`)가 **MLflow 모델 레지스트리**에 이 이름으로 등록됩니다. 추론 서비스가 이 이름으로 모델을 자동 로드할 수 있습니다.

예시: `yolo_detector`

등록을 건너뛰려면 비워두세요.

---

### 예시: 최소한의 첫 번째 실행

| 항목 | 값 |
|---|---|
| Dataset YAML | `/data/dataset.yaml` |
| Model | `yolov8n.pt` |
| Epochs | `5` |
| Image Size | `640` |
| Batch Size | `8` |
| Device | `cpu` |
| Export Format | `— none —` |
| Experiment Name | `첫번째_테스트` |
| Run Name | `검증-실행` |
| Register Model As | _(비워두기)_ |

---

### 작업 제출

**Start Training** 버튼을 클릭합니다.

- 버튼이 **Training in progress…**로 바뀌고 비활성화됩니다
- 오른쪽에 파란색 `RUNNING` 배지와 함께 작업 카드가 나타납니다
- 몇 초 내에 로그 뷰어에서 출력이 시작됩니다

---

## 8. 실시간 로그 모니터링

작업이 실행되는 동안 오른쪽 로그 패널에 학습 프로세스의 최근 50줄이 2초마다 갱신되어 표시됩니다.

### 확인할 내용

```
{"time": "...", "level": "INFO", ..., "message": "Starting training: model=yolov8n.pt ..."}
```

Ultralytics는 에포크마다 학습 진행 표를 출력합니다:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/50      0.00G      1.842      2.193      1.321         12        640
  2/50      0.00G      1.721      2.047      1.289         15        640
  ...
```

학습 완료 시 검증 메트릭이 출력됩니다:

```
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
  all        100        312      0.723      0.681      0.741      0.512
```

### 상태 배지

| 배지 | 의미 |
|---|---|
| 🔵 `RUNNING` (깜빡임) | 작업 진행 중 |
| 🟢 `DONE` | 학습 성공적으로 완료 |
| 🔴 `FAILED` | 오류 발생 — 로그 확인 필요 |

`DONE` 상태가 되면 **Experiments** 페이지로 이동하는 링크가 포함된 초록색 배너가 표시됩니다.

---

## 9. MLflow에서 결과 확인

학습 완료 후:

1. 초록색 완료 배너의 **View Experiments →**를 클릭하거나,
2. `http://localhost:5000`으로 직접 접속합니다

MLflow UI에서 확인 가능한 항목:

- **Parameters (파라미터)** — 기록된 모든 하이퍼파라미터 (모델, 에포크, imgsz, 배치, 디바이스)
- **Metrics (메트릭)** — 최종 검증 메트릭 (mAP50, mAP50-95, 정밀도, 재현율, box/cls/dfl 손실)
- **Artifacts (아티팩트)** — `weights/best.pt` (최적 체크포인트) 및 선택적으로 `exports/` (변환된 형식)

AI-Forge UI의 **Experiments** 페이지 (`http://localhost:3000/experiments`)에서도 최신 메트릭 요약을 볼 수 있습니다.

---

## 10. 학습된 모델을 추론에 활용하기

### 방법 A — 모델 레지스트리 사용 (권장)

학습 시 **Register Model As**를 입력했다면:

1. `http://localhost:5000` → **Models** 탭으로 이동
2. 등록된 모델 (예: `yolo_detector`)을 찾습니다
3. 새 버전을 **Production** 단계로 승격합니다 (버전 클릭 → Stage → Production)
4. 추론 서비스를 재시작하여 새 가중치를 로드합니다:

```bash
docker compose restart inference
```

추론 서비스 (`http://localhost:8000`)가 MLflow에서 Production 단계의 가중치를 자동으로 로드합니다.

### 방법 B — 체크포인트 직접 다운로드

```bash
# 학습 컨테이너에서 best.pt 경로 확인
docker exec trainer find /app/runs -name "best.pt" | head -1
# 예시 출력: /app/runs/train/exp/weights/best.pt

# 호스트로 복사
docker cp trainer:/app/runs/train/exp/weights/best.pt ./내_best.pt
```

---

## 11. 모델 내보내기(Export) 옵션

내보내기 형식을 선택한 경우, 변환된 파일은 MLflow의 해당 실행 `exports/` 아티팩트 디렉토리와 학습 컨테이너 디스크 모두에 저장됩니다.

### ONNX

PyTorch 없이 프로덕션 서빙에 가장 적합합니다. ONNX Runtime, OpenVINO, TensorRT (trtexec 경유), 대부분의 ML 서빙 프레임워크와 호환됩니다.

```bash
# 학습 컨테이너에서 ONNX 파일 다운로드
docker exec trainer find /app/runs -name "*.onnx"
docker cp trainer:/app/runs/train/exp/weights/best.onnx ./model.onnx
```

### TensorRT (`engine`)

NVIDIA GPU와 컨테이너 내 TensorRT가 설치되어 있어야 합니다. 내보내기 단계가 느리지만 (수 분 소요) 결과 `.engine` 파일은 가장 빠른 GPU 추론 속도를 제공합니다.

> TensorRT 오류로 내보내기가 실패해도 학습 결과 (`.pt` 체크포인트 및 메트릭)는 저장됩니다. 내보내기 실패는 치명적이지 않습니다.

### TorchScript

Ultralytics 라이브러리 없이 로드할 수 있는 이식 가능한 PyTorch 형식:

```python
import torch
model = torch.jit.load("best.torchscript")
```

---

## 12. GPU 학습

### 요구사항

- 호스트에 CUDA 11.8+ 또는 CUDA 12.x 드라이버가 설치된 NVIDIA GPU
- NVIDIA Container Toolkit 설치: `nvidia-ctk --version`
- GPU 접근을 위한 Docker 데몬 설정 완료

### docker-compose.yaml에서 GPU 활성화

`trainer` 서비스에 `deploy` 블록을 추가합니다:

```yaml
trainer:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

재빌드:

```bash
docker compose up -d --build trainer
```

컨테이너 내에서 GPU가 보이는지 확인:

```bash
docker exec trainer nvidia-smi
```

### UI의 Device 필드 값

| UI Device 값 | 의미 |
|---|---|
| `0` | 첫 번째 GPU |
| `1` | 두 번째 GPU |
| `0,1` | 두 GPU에서 데이터 병렬 학습 |

---

## 13. 문제 해결

### "Dataset YAML을 찾을 수 없음" 또는 즉각적인 실패

- 컨테이너 내부에서 경로가 올바른지 확인: `docker exec trainer ls /data`
- `docker-compose.yaml`에 데이터 디렉토리가 볼륨 마운트되어 있는지 확인
- 경로는 반드시 `/`로 시작하는 절대 경로여야 합니다 — 상대 경로는 지원되지 않습니다

### 로그 출력 없이 `RUNNING` 상태가 계속됨

- Ultralytics는 첫 사용 시 인터넷에서 모델 가중치를 다운로드합니다. 연결 속도에 따라 1–5분이 소요될 수 있습니다.
- 로그 뷰어에서 `Downloading yolov8n.pt from https://...`와 같은 줄을 확인하세요.

### 제출 직후 `FAILED` 배지

- 로그 출력에서 Python 예외 메시지를 확인하세요
- 주요 원인:
  - `dataset.yaml`의 `path:` 경로 오류
  - 라벨 파일 누락 또는 이름 불일치
  - `batch` 크기가 사용 가능한 RAM/VRAM을 초과 → 배치 크기 감소
  - 잘못된 모델 이름 오타

### 메모리 부족(OOM) 오류

폼에서 배치 크기 또는 이미지 크기를 줄이세요. CPU에서 `imgsz=640`으로 배치 크기 16 이상을 사용하면 수 GB의 RAM이 필요할 수 있습니다.

### 트레이너 서비스 연결 불가 (UI에 "Train service unreachable" 표시)

```bash
# 트레이너 상태 확인
docker compose ps trainer
curl http://localhost:6006/health
# 예상 응답: {"status":"ok"}

# 비정상 상태 시 재시작
docker compose restart trainer
```

### MLflow에 메트릭이 기록되지 않음

- `MLFLOW_TRACKING_URI=http://mlflow:5000`이 설정되어 있는지 확인 (`docker-compose.yaml`에 이미 설정됨)
- MLflow 서비스가 정상 상태인지 확인: `docker compose ps mlflow`
- MLflow는 MinIO와 Postgres가 먼저 정상 상태여야 합니다

### 트레이너 재시작 후 이전 작업이 보이지 않음

인메모리 작업 저장소는 **컨테이너 재시작 시 초기화**됩니다. 작업 기록은 재시작 간에 유지되지 않습니다. 학습 결과 (메트릭, 아티팩트)는 MLflow에 영구적으로 저장되며 — UI에 표시되는 작업 상태 및 로그 항목만 사라집니다.
