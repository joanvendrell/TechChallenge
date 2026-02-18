from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List

Domain = Literal["heat_transfer", "solid_mechanics", "fluid"]

class GeometrySpec(BaseModel):
    type: Literal["box", "beam", "cylinder", "thin_film", "custom", "sphere"] = "custom"
    dims: Dict[str, float] = Field(default_factory=dict)  # meters, e.g. {"Lx":1e-6,"Ly":1e-6,"Lz":30e-9}

class MaterialSpec(BaseModel):
    name: Optional[str] = None
    props: Dict[str, float] = Field(default_factory=dict)  # e.g. {"k":400, "E":200e9, "nu":0.3}

class BCSpeс(BaseModel):
    kind: Dict[str, Any] = Field(default_factory=dict)     # e.g. {"T_left":400, "T_right":300} or {"fixed_end":"x=0","force_end":"x=L","F":5}

class MeshSpec(BaseModel):
    resolution: Optional[str] = "auto"
    params: Dict[str, int] = Field(default_factory=dict)   # e.g. {"nx":40,"ny":40,"nz":2}

class PDESpec(BaseModel):
    domain: Domain
    pde: str                       # e.g. "steady_heat", "linear_elasticity"
    geometry: GeometrySpec
    material: MaterialSpec
    bc: BCSpeс
    mesh: MeshSpec
    outputs: List[str] = Field(default_factory=list)       # e.g. ["temperature"], ["displacement","von_mises"]
    units: str = "SI"
    extracted: Dict[str, Any] = Field(default_factory=dict) # raw parsed values for traceability
    warnings: List[str] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)
