# TASKS.md — AI-Forge Project Task Registry

> Reference: [CLAUDE.md](CLAUDE.md) for conventions, API specs, and architecture.
> Last updated: 2026-02-23 (session 6)

---

## High Priority

| ID    | Task                                                                 | Status      | Completed At     |
|-------|----------------------------------------------------------------------|-------------|------------------|
| [H1]  | Implement `POST /api/v1/detect` endpoint in inference_service        | ✅ Done     | 2026-02-21 20:41 |
| [H2]  | Add base64 image decoding and input validation to detect endpoint    | ✅ Done     | 2026-02-21 20:41 |
| [H3]  | Implement model loading from MLflow artifact registry                | ✅ Done     | 2026-02-21 20:41 |
| [H4]  | Return structured `predictions` and `metrics` in detect response     | ✅ Done     | 2026-02-21 20:41 |
| [H5]  | Add structured error handling and correct HTTP status codes to API   | ✅ Done     | 2026-02-21 20:41 |
| [H6]  | Implement `train.py` with YOLO training pipeline and MLflow tracking | ✅ Done     | 2026-02-21 20:41 |
| [H7]  | Write pytest test suite for inference_service (`/health`, `/detect`) | ✅ Done     | 2026-02-21 20:41 |
| [H8]  | Write pytest test suite for train_service                            | ✅ Done     | 2026-02-21 20:41 |
| [H9]  | Create `scripts/test_runner.sh` to orchestrate tests                 | ✅ Done     | 2026-02-21 20:41 |
| [H10] | Add GitHub Actions CI workflow (build, lint, test on push/PR)        | ✅ Done     | 2026-02-21 20:41 |
| [H11] | Enforce Docker Compose stack rebuild and restart reliability         | ✅ Done     | 2026-02-21 20:41 |

---

## Medium Priority

| ID    | Task                                                                      | Status      | Completed At |
|-------|---------------------------------------------------------------------------|-------------|--------------|
| [M1]  | Scaffold Next.js `src/` directory with pages and component structure      | ✅ Done     | 2026-02-23 09:30 |
| [M2]  | Build experiment tracking dashboard UI (list/filter MLflow experiments)   | ✅ Done     | 2026-02-23 09:30 |
| [M3]  | Build model management UI (view, register, compare models from registry)  | ✅ Done     | 2026-02-23 09:30 |
| [M4]  | Build inference UI (upload image, call `/api/v1/detect`, render results)  | ✅ Done     | 2026-02-23 09:30 |
| [M5]  | Add MLflow metrics visualization charts to UI dashboard                   | ✅ Done     | 2026-02-23 10:15 |
| [M6]  | Support YOLO model variants (YOLOv8, YOLOv10, YOLOv11) in training       | ✅ Done     | 2026-02-23 10:15 |
| [M7]  | Add ONNX/TensorRT export step after training                              | ✅ Done     | 2026-02-23 10:15 |
| [M8]  | Integrate Claude VLM as an optional inference backend                     | ✅ Done     | 2026-02-23 11:30 |
| [M9]  | Add Grafana + Loki logging aggregation service to Docker Compose          | ✅ Done     | 2026-02-23 10:15 |
| [M10] | Add structured JSON logging across inference and train services           | ✅ Done     | 2026-02-21 20:41 |
| [M11] | Add Jest test suite for UI components                                     | ✅ Done     | 2026-02-23 11:30 |
| [M12] | Add ESLint and Prettier config and enforce in CI                          | ✅ Done     | 2026-02-23 09:30 |
| [M13] | Add pylint/black linting to Python services and enforce in CI             | ✅ Done     | 2026-02-23 09:30 |

---

## Low Priority

| ID   | Task                                                                        | Status      | Completed At |
|------|-----------------------------------------------------------------------------|-------------|--------------|
| [L1] | Implement API authentication (JWT or API key middleware)                    | ✅ Done     | 2026-02-23 12:15 |
| [L2] | Implement role-based access control (RBAC) for UI and API                  | ✅ Done     | 2026-02-23 12:15 |
| [L3] | Create `.env.example` template for onboarding                               | ✅ Done     | 2026-02-21 20:41 |
| [L4] | Expand `README.md` with setup, usage, and architecture overview             | ✅ Done     | 2026-02-23 12:17 |
| [L5] | Add OpenAPI/Swagger auto-docs for inference service (`/docs`)               | ✅ Done     | 2026-02-23 10:15 |
| [L6] | Write deployment guide for on-premise production setup                      | ✅ Done     | 2026-02-23 12:16 |
| [L7] | Add MinIO bucket lifecycle policies for artifact retention                  | ✅ Done     | 2026-02-23 12:16 |
| [L8] | Restrict internal service networking (Postgres, MinIO not publicly exposed) | ✅ Done     | 2026-02-23 10:15 |
| [L9] | Add `teardown.sh` script for clean stack removal                            | ✅ Done     | 2026-02-23 10:15 |

---

## Session 6 — Training UI

| ID   | Task                                                                               | Status      | Completed At |
|------|------------------------------------------------------------------------------------|-------------|--------------|
| [T1] | Add `train_service/app.py` FastAPI server (POST /train, GET /train/status, /jobs) | ✅ Done     | 2026-02-23   |
| [T2] | Add Next.js API proxy routes (`/api/train`, `/api/train/status/[jobId]`)           | ✅ Done     | 2026-02-23   |
| [T3] | Build `/training` page — form, live log viewer, 2 s polling, status badges        | ✅ Done     | 2026-02-23   |
| [T4] | Update `docker-compose.yaml` — trainer runs uvicorn, TRAIN_PORT, healthcheck      | ✅ Done     | 2026-02-23   |
| [T5] | Add Training link to `Sidebar.tsx` and Training card to `page.tsx` dashboard      | ✅ Done     | 2026-02-23   |

---

## Status Key

| Status      | Meaning                                  |
|-------------|------------------------------------------|
| Not Started | Work has not begun                       |
| In Progress | Actively being developed                 |
| ✅ Done     | Implemented and tests passing            |
| Blocked     | Waiting on a dependency or decision      |
