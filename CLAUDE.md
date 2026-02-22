# CLAUDE.md — AI Agent Development Guide for AI-Forge

## 1) Project Overview  
AI-Forge is an on-premise Vision(VLM) MLOps platform designed to support seamless development, training, experiment tracking, optimized inference serving, and unified UIs. The platform uses MLflow with a Postgres metadata store and MinIO artifact storage, modular training using PyTorch/YOLO families, scalable FastAPI inference APIs, and modern frontend tooling, all orchestrated via Docker Compose with local development volume mounts to enable real-time code updates and development workflows.

## 2) Project Structure (Monorepo)
The repository is organized as:  
AI-Forge/  
├── .gitignore  
├── .env  
├── setup_full_stack.sh  
├── docker-compose.yaml  
├── CLAUDE.md  
├── README.md  
├── mlflow/  
│ └── Dockerfile  
├── train_service/  
│ ├── Dockerfile  
│ ├── requirements.txt  
│ ├── train.py  
│ └── tests/  
├── inference_service/  
│ ├── Dockerfile  
│ ├── requirements.txt  
│ ├── app.py  
│ └── tests/  
├── ui/  
│ ├── package.json  
│ ├── src/  
│ └── tests/  
├── scripts/  
│ ├── setup_full_stack.sh  
│ ├── teardown.sh  
│ └── test_runner.sh  
├── postgres_data/  
└── minio_data/  

## 3) Environment & Setup  
All configuration must be defined via a `.env` file and referenced by services. Do not hard-code secrets; always use environment variables. Mount code volumes in `docker-compose.yaml` so that changes in the host filesystem are reflected in containers. Use VS Code Remote Containers or SSH + VS Code with full workspace mounts. Enable live reload for FastAPI and frontend (Next.js) during development. Maintain `.env.example` as a template for onboarding. When developing or testing any component (e.g., mlflow, train_service, inference_service), attach directly to the corresponding Docker container and run tests inside it.

## 4) Coding Standards & Conventions  
**Python:** PEP8, type hints, docstrings for public APIs.  
**JavaScript/TypeScript:** Follow ESLint/Prettier conventions.  
**Logging:** Structured and contextual logging across services.  
**Security:** Secrets must never appear in code. Use environment variables and secure CI secrets store.

## 5) API Specifications (Inference Service)  
**Health Check**

GET /health
Response: { "status": "ok" }

**Detection Endpoint**

POST /api/v1/detect
Content-Type: application/json
Body:
{
"image_base64": "...",
"options": {...}
}

Expected JSON Response:

{
"success": true,
"predictions": [...],
"metrics": {...}
}

Use appropriate HTTP status codes and provide clear error messages.

## 6) MLOps Integration (MLflow & MinIO)
MLflow is the central experiment and model registry. Use:
- Tracking server: `http://mlflow:5000`
- MinIO artifact store: `http://minio:9000`

**Python Example**

import mlflow

mlflow.set_tracking_uri("http://mlflow:5000
")
mlflow.log_param("lr", 0.001)
mlflow.log_metric("accuracy", 0.95)


## 7) Testing & Continuous Integration
**Backend:** Use `pytest`.  
**Frontend:** Use Jest or equivalent.  
Tests must run locally and in CI. A test runner script (`scripts/test_runner.sh`) should orchestrate service startup and test execution:

#!/usr/bin/env bash
docker compose up -d --build
pytest -q


**CI Workflow Example**

jobs:
test:
runs-on: ubuntu-latest
steps:
- uses: actions/checkout@v3
- uses: docker/setup-buildx-action@v2
- run: docker compose up -d --build
- run: pytest -q


CI must build all services, run tests, enforce linting for Python and JS, and fail on any errors.

## 8) Security Best Practices
Never commit plaintext credentials. Use `.env.example` templates for onboarding and configure CI/CD to use secure secret management. Restrict networking exposure so internal services like Postgres or MinIO are not publicly accessible.

## 9) Priority Tasks (with IDs)
**High Priority**
- [H1] Implement detection API (`/api/v1/detect`)
- [H2] Build backend automated test suites
- [H3] Add robust CI workflows (GitHub Actions)
- [H4] Ensure Docker Compose stack rebuild and restart reliability

**Medium Priority**
- [M1] Add metric dashboards to UI
- [M2] Support multi-model workflows (Claude VLM, ONNX/TensorRT)
- [M3] Logging aggregation (e.g., Grafana + Loki/ELK)

**Low Priority**
- [L1] Implement authentication and RBAC
- [L2] Enhance documentation (API docs, user guides)

## 10) Output Format for AI Agents
Agents must produce outputs in one of the following two formats only:

**I) Patch (Unified Diff) Format**

=== PATCH START ===
diff --git a/... b/...
...
=== PATCH END ===


**II) JSON Automation Format**

{
"task": "...",
"files_modified": [...],
"tests_passed": true,
"notes": "..."
}


Include:
- Description of changes
- List of affected files
- Test outcomes

## 11) Rules for AI Agents
1. **Read This Guide First:** Agents must fully read this entire document and understand the architecture, conventions, and tasks before acting.  
2. **Deliver Complete Work:** When completing a task, produce working code, tests, documentation updates if needed, and CI updates if applicable.  
3. **Validation:** All changes must pass the provided test runner (`scripts/test_runner.sh`).  
4. **Transparency:** Clearly list affected files and test results in the specified output format.

## 12) Developer Tips
For local development, use `uvicorn --reload` for backend development and `npm run dev` for frontend hot reload. Keep `.env.example` updated and commit it to version control for onboarding.

## 13) Goals Summary
The ultimate goal of AI-Forge is to enable a scalable, extensible, on-premise Vision MLOps environment that supports autonomous AI agent coding, robust testing, and unified developer workflows. Always update this CLAUDE.md when conventions or workflows evolve.


