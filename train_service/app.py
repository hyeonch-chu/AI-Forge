"""FastAPI HTTP server for the AI-Forge training service.

Exposes a minimal REST API that lets the UI (or any HTTP client) launch
YOLO training jobs and poll their progress without shelling into the
container manually.

Endpoints
---------
GET  /health
    Liveness probe — always returns {"status": "ok"}.

POST /api/v1/train
    Submit a new training job.  Spawns ``train.py`` as a subprocess in the
    background and returns a job_id immediately.

GET  /api/v1/train/status/{job_id}
    Poll the status of a running or finished job.  Returns the last 200 log
    lines so the UI can stream progress without a WebSocket.

GET  /api/v1/train/jobs
    List all job IDs and their statuses (most-recent-first).

Usage (inside the trainer container):
    uvicorn app:app --reload --host 0.0.0.0 --port 6006
"""

import os
import re
import subprocess
import threading
import uuid
from typing import List, Optional

# ---------------------------------------------------------------------------
# ANSI / terminal-escape helpers
# ---------------------------------------------------------------------------

# Matches common ANSI CSI sequences (colours, cursor moves, erase-line, etc.)
_ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[mKHFJGABCDSTsu]|\x1b\[K|\r')


def _strip_ansi(s: str) -> str:
    """Remove ANSI escape codes and bare carriage-returns from a string."""
    return _ANSI_RE.sub('', s).strip()


def _is_download_progress(line: str) -> bool:
    """Ultralytics file-download tqdm line (thick ━ bar).

    Example:
        Downloading VOCtrainval_11-May-2012.zip: 82% ━━━━━━━━ 1.5/1.8GB
    Collapse all updates into one slot — user only needs the latest %.
    """
    return bool(re.search(r'(Downloading|Unzipping)', line) and '━' in line)


def _is_batch_progress(line: str) -> bool:
    """YOLO per-epoch tqdm training progress line.

    Format: '{epoch}/{total}   {gpu_mem}   {box_loss} ... {img_size}: {pct}%'
    Example:
        1/20     0.326G      1.161      4.444      1.366         32        640:  25%|
    One slot is kept per epoch: progress updates overwrite in place;
    the final 100% line stays as the epoch's permanent summary row.
    """
    return bool(re.match(r'^\d+/\d+\s+\d', line))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    """Parameters forwarded directly to train.py CLI arguments."""

    data: str = Field(..., description="Path to dataset YAML inside the container, e.g. /data/coco.yaml")
    model: str = Field("yolov8n.pt", description="YOLO preset name or absolute path to a custom .pt file")
    epochs: int = Field(50, ge=1, le=1000, description="Number of training epochs")
    imgsz: int = Field(640, ge=32, le=1280, description="Input image size (square side in pixels)")
    batch: int = Field(16, ge=1, le=512, description="Batch size")
    experiment: str = Field("yolo_training", description="MLflow experiment name")
    run_name: Optional[str] = Field(None, description="Optional MLflow run display name")
    device: str = Field("cpu", description="Training device: 'cpu', '0' (GPU 0), '0,1' (multi-GPU)")
    register_name: Optional[str] = Field(
        None, description="Register the best checkpoint in the MLflow Model Registry under this name"
    )
    export_format: Optional[str] = Field(
        None,
        description=(
            "Export trained model after training. "
            "Choices: onnx | engine | torchscript | coreml | saved_model. "
            "'engine' requires NVIDIA TensorRT."
        ),
    )


class TrainStatus(BaseModel):
    """Response schema for job submission and status polling."""

    job_id: str
    status: str = Field(description="One of: running | done | failed")
    exit_code: Optional[int] = None
    log_tail: List[str] = Field(default_factory=list, description="Last 50 output lines from train.py")


class JobSummary(BaseModel):
    """Lightweight job record for the jobs list endpoint."""

    job_id: str
    status: str
    exit_code: Optional[int] = None


# ---------------------------------------------------------------------------
# In-memory job store
# Keys  : 8-char hex job IDs
# Values: {"status": str, "proc": Popen, "log_lines": list[str], "exit_code": int|None}
# NOTE  : Jobs are lost on container restart; this is intentional for a dev
#         workflow.  For persistence, replace with a DB or file-backed store.
# ---------------------------------------------------------------------------
_jobs: dict = {}

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-Forge Train Service",
    description="REST API for launching and monitoring YOLO + MLflow training jobs.",
    version="1.0.0",
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe — used by Docker healthcheck and load-balancer probes."""
    return {"status": "ok"}


@app.post("/api/v1/train", response_model=TrainStatus, tags=["training"])
def start_train(req: TrainRequest) -> TrainStatus:
    """Launch a training job asynchronously.

    Builds the train.py command from the request body, spawns it as a
    background subprocess, and returns the ``job_id`` immediately so the
    caller can start polling ``/api/v1/train/status/{job_id}``.
    """
    job_id = uuid.uuid4().hex[:8]

    # Build the CLI command that mirrors the train.py argparser
    cmd: List[str] = [
        "python", "train.py",
        "--data", req.data,
        "--model", req.model,
        "--epochs", str(req.epochs),
        "--imgsz", str(req.imgsz),
        "--batch", str(req.batch),
        "--experiment", req.experiment,
        "--device", req.device,
    ]
    if req.run_name:
        cmd += ["--run-name", req.run_name]
    if req.register_name:
        cmd += ["--register", req.register_name]
    if req.export_format:
        cmd += ["--export", req.export_format]

    # Spawn the subprocess — stdout+stderr merged so the log viewer gets everything
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/app",
        env=os.environ.copy(),
    )

    entry: dict = {
        "status": "running",
        "proc": proc,
        "log_lines": [],
        "exit_code": None,
    }
    _jobs[job_id] = entry

    # Background thread that drains the pipe — prevents the child process from
    # blocking when its stdout buffer is full.
    def _drain() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = _strip_ansi(raw)
            if not line:
                continue

            logs = entry["log_lines"]

            if _is_download_progress(line):
                # ── Download progress ─────────────────────────────────────
                # Collapse all updates into the last download slot so the
                # buffer isn't flooded with hundreds of "82% ━━━" lines.
                if logs and _is_download_progress(logs[-1]):
                    logs[-1] = line
                else:
                    logs.append(line)

            elif _is_batch_progress(line):
                # ── YOLO epoch/batch progress ─────────────────────────────
                # Keep exactly ONE slot per epoch so:
                #   • Intra-epoch batch updates overwrite in place (live %)
                #   • The final 100% summary is preserved as the epoch row
                #   • The next epoch appends a NEW row (epoch history kept)
                epoch_key = line.split()[0]  # e.g. "1/20"
                replaced = False
                for i in range(len(logs) - 1, max(len(logs) - 200, -1), -1):
                    if re.match(r'^\d+/\d+\s', logs[i]) and logs[i].split()[0] == epoch_key:
                        logs[i] = line
                        replaced = True
                        break
                if not replaced:
                    logs.append(line)

            elif logs and line == logs[-1]:
                # ── Consecutive duplicate ─────────────────────────────────
                # Drop repeated identical lines (e.g. recurring shm errors)
                # to prevent them from filling the buffer with noise.
                pass

            else:
                logs.append(line)

        proc.wait()
        entry["exit_code"] = proc.returncode
        entry["status"] = "done" if proc.returncode == 0 else "failed"

    threading.Thread(target=_drain, daemon=True).start()

    return TrainStatus(job_id=job_id, status="running")


@app.get("/api/v1/train/status/{job_id}", response_model=TrainStatus, tags=["training"])
def get_status(job_id: str) -> TrainStatus:
    """Return the current status and the last 50 log lines for a job.

    Poll this endpoint every few seconds from the UI to display live progress.
    Once ``status`` is ``"done"`` or ``"failed"`` the job will not change further.
    """
    entry = _jobs.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return TrainStatus(
        job_id=job_id,
        status=entry["status"],
        exit_code=entry["exit_code"],
        log_tail=entry["log_lines"][-200:],
    )


@app.get("/api/v1/train/jobs", response_model=List[JobSummary], tags=["training"])
def list_jobs() -> List[JobSummary]:
    """Return a summary of all submitted jobs, most-recent-first."""
    return [
        JobSummary(job_id=jid, status=e["status"], exit_code=e["exit_code"])
        for jid, e in reversed(list(_jobs.items()))
    ]
