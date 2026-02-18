from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ArtifactType = Literal["vtk", "xdmf", "json", "txt", "bin"]

@dataclass
class Artifact:
    name: str
    path: Path
    type: ArtifactType

    def to_public(self, job_id: str) -> dict:
        # This matches how you'll serve artifacts later:
        return {
            "name": self.name,
            "type": self.type,
            "url": f"/api/jobs/{job_id}/artifacts/{self.name}",
        }
