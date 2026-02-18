from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .prompt_parser import parse_prompt_to_spec
from .pde_spec import PDESpec
from pathlib import Path
from .solvers.dispatch import solve
from .pde_spec import PDESpec
import mimetypes
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

ARTIFACT_ROOT = Path("artifacts")

from .schemas import (
    SimRequest,
    CreateJobResponse,
    JobDetail,
    JobMeta,
    ListJobsResponse,
)

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

app = FastAPI(title="Physics NLP Simulation API", version="0.2.0")

# Dev CORS (lock down in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# In-memory job store + queue
# ----------------------------

@dataclass
class JobRecord:
    job_id: str
    status: str
    domain: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = 0.0
    message: str = ""
    request: Optional[Dict[str, Any]] = None
    parsed_spec: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cancel_requested: bool = False

JOBS: Dict[str, JobRecord] = {}
JOB_QUEUE: "asyncio.Queue[str]" = asyncio.Queue()
WORKERS_STARTED = False
MAX_JOBS_TO_KEEP = 200  # prevent unbounded growth for prototype


# ----------------------------
# Utilities
# ----------------------------

def check_cancel(job: JobRecord):
    if job.cancel_requested:
        job.status = "cancelled"
        job.finished_at = time.time()
        job.progress = 1.0
        job.message = "Cancelled by user."
        raise asyncio.CancelledError()

def trim_jobs_if_needed():
    # Keep only the most recent jobs (by created_at)
    if len(JOBS) <= MAX_JOBS_TO_KEEP:
        return
    items = sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)
    keep = {j.job_id for j in items[:MAX_JOBS_TO_KEEP]}
    for jid in list(JOBS.keys()):
        if jid not in keep:
            del JOBS[jid]


# ----------------------------
# Worker loop
# ----------------------------

async def simulate_job(job_id: str):
    job = JOBS[job_id]
    job.status = "running"
    job.started_at = time.time()
    job.message = "Parsing prompt..."
    job.progress = 0.05

    req = job.request or {}                              # < ----- HERE
    prompt = req.get("prompt", "")
    domain = req.get("domain", "other")
    parameters = req.get("parameters", {})
    check_cancel(job)
    # -----------------------
    # 2) Parse prompt -> spec
    # -----------------------
    job.message = "Parsing prompt..."
    job.progress = 0.05
    await asyncio.sleep(0.05)
    out_dir = ARTIFACT_ROOT
    spec_dir = out_dir / "spec"
    spec_key = "latest"  # or use a session_id later (recommended)
    base = load_spec_from_artifacts(spec_dir, spec_key)
    spec = parse_prompt_to_spec(prompt, domain_hint=domain, parameters=base or None)
    job.parsed_spec = spec.model_dump()
    save_spec_to_artifacts(spec_dir, spec_key, spec.model_dump())

    # -----------------------
    # 3) Solve with FEniCSx
    # -----------------------
    job.message = "Solving PDE with FEniCSx..."
    job.progress = 0.35
    check_cancel(job)
    try:
        out_dir = ARTIFACT_ROOT #job_id
        # Validate spec object (dict -> PDESpec)
        spec_obj = PDESpec.model_validate(job.parsed_spec)

        # This is the general dispatch call (heat steady implemented first)
        result, artifacts = solve(spec_obj, out_dir=out_dir)

    except Exception as e:
        job.status = "failed"
        job.finished_at = time.time()
        job.error = str(e)
        job.message = "Solver failed."
        job.progress = 1.0
        return

    # -----------------------
    # 4) Postprocess + package response
    # -----------------------
    job.message = "Packaging results..."
    job.progress = 0.95
    await asyncio.sleep(0.05)
    check_cancel(job)
    job.result = {
        **result,
        "artifacts": [a.to_public(job_id) for a in artifacts],
        "warnings": spec.warnings,
        "clarifying_questions": spec.clarifying_questions,
    }

    job.status = "completed"
    job.finished_at = time.time()
    job.progress = 1.0
    job.message = "Completed."

async def worker_loop(worker_name: str):
    while True:
        job_id = await JOB_QUEUE.get()
        try:
            if job_id not in JOBS:
                continue
            job = JOBS[job_id]
            # If job was deleted or already processed, skip
            if job.status not in ("queued",):
                continue
            await simulate_job(job_id)
        except Exception as e:
            if job_id in JOBS:
                job = JOBS[job_id]
                job.status = "failed"
                job.finished_at = time.time()
                job.error = repr(e)
                job.message = "Failed."
        finally:
            JOB_QUEUE.task_done()


@app.on_event("startup")
async def startup():
    global WORKERS_STARTED
    if WORKERS_STARTED:
        return
    # Start 1-2 workers for the prototype
    asyncio.create_task(worker_loop("worker-1"))
    asyncio.create_task(worker_loop("worker-2"))
    WORKERS_STARTED = True


# ----------------------------
# API
# ----------------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/parse", response_model=PDESpec)
def parse_only(req: SimRequest):
    return parse_prompt_to_spec(req.prompt, domain_hint=req.domain, parameters=req.parameters)

@app.post("/api/simulate", response_model=CreateJobResponse)
async def create_job(req: SimRequest):
    job_id = str(uuid.uuid4())
    now = time.time()

    JOBS[job_id] = JobRecord(
        job_id=job_id,
        status="queued",
        domain=req.domain,
        created_at=now,
        progress=0.0,
        message="Queued.",
        request=req.model_dump(),
    )

    trim_jobs_if_needed()

    await JOB_QUEUE.put(job_id)

    return CreateJobResponse(
        job_id=job_id,
        status="queued",
        message="Job accepted and queued."
    )

@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # If already finished, nothing to cancel
    if job.status in ("completed", "failed", "cancelled"):
        return {"ok": True, "status": job.status}

    job.cancel_requested = True

    # If it is still queued, we can cancel "for real"
    if job.status == "queued":
        job.status = "cancelled"
        job.finished_at = time.time()
        job.progress = 1.0
        job.message = "Cancelled (was queued)."
        return {"ok": True, "status": "cancelled"}

    # If running: this only marks it; actual stop needs process isolation (see below)
    job.message = "Cancellation requested..."
    return {"ok": True, "status": "cancelling"}

@app.get("/api/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobDetail(
        job_id=job.job_id,
        status=job.status,  # type: ignore
        domain=job.domain,  # type: ignore
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress=job.progress,
        message=job.message,
        parsed_spec=job.parsed_spec,
        result=job.result,
        error=job.error,
    )

@app.get("/api/jobs", response_model=ListJobsResponse)
def list_jobs(limit: int = 20):
    items: List[JobRecord] = sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)[:limit]
    jobs = [
        JobMeta(
            job_id=j.job_id,
            status=j.status,  # type: ignore
            domain=j.domain,  # type: ignore
            created_at=j.created_at,
            started_at=j.started_at,
            finished_at=j.finished_at,
            progress=j.progress,
            message=j.message,
        )
        for j in items
    ]
    return ListJobsResponse(jobs=jobs)

@app.get("/")
def root():
    return {"message": "API up."}

@app.get("/api/jobs/{job_id}/artifacts/{name}")
def get_artifact(job_id: str, name: str):
    path = Path("artifacts") / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type, _ = mimetypes.guess_type(str(path))
    media_type = media_type or "application/octet-stream"

    return FileResponse(path, media_type=media_type, filename=name)

def load_spec_from_artifacts(spec_dir: Path, key: str) -> Optional[Dict[str, Any]]:
    """
    Loads the last spec dict from artifacts/spec/<key>.json if it exists.
    """
    path = spec_dir / f"{key}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_spec_to_artifacts(spec_dir: Path, key: str, spec_dict: Dict[str, Any]) -> Path:
    """
    Saves spec dict to artifacts/spec/<key>.json using atomic replace.
    """
    spec_dir.mkdir(parents=True, exist_ok=True)
    path = spec_dir / f"{key}.json"
    tmp = spec_dir / f".{key}.json.tmp"

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(spec_dict, f, indent=2)

    os.replace(tmp, path)  # atomic on same filesystem
    return path