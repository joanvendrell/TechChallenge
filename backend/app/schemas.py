from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List

PhysicsDomain = Literal["heat_transfer", "solid_mechanics", "fluid", "other"]

JobStatus = Literal["queued", "running", "completed", "failed"]

class SimRequest(BaseModel):
    prompt: str = Field(..., min_length=3)
    domain: PhysicsDomain = "other"
    parameters: Dict[str, Any] = Field(default_factory=dict)

class JobMeta(BaseModel):
    job_id: str
    status: JobStatus
    domain: PhysicsDomain
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = 0.0  # 0..1
    message: str = ""

class JobDetail(JobMeta):
    parsed_spec: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str

class ListJobsResponse(BaseModel):
    jobs: List[JobMeta]
