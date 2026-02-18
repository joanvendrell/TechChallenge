from typing import Tuple, Dict, Any
from ..pde_spec import PDESpec

def get_box_dims(spec: PDESpec, defaults=(1.0, 0.2, 0.2)) -> Tuple[float, float, float]:
    if spec.geometry.type != "box":
        raise ValueError(f"This solver expects geometry.type='box' but got '{spec.geometry.type}'.")
    dims = spec.geometry.dims or {}
    Lx = float(dims.get("Lx", defaults[0]))
    Ly = float(dims.get("Ly", defaults[1]))
    Lz = float(dims.get("Lz", defaults[2]))
    return Lx, Ly, Lz

def get_sphere_radius(spec: PDESpec, default=1.0) -> float:
    if spec.geometry.type != "sphere":
        raise ValueError(f"Expected geometry.type='sphere' but got '{spec.geometry.type}'.")
    dims = spec.geometry.dims or {}
    if "R" not in dims:
        raise ValueError("Sphere geometry requires dims['R'] (radius).")
    return float(dims["R"])
