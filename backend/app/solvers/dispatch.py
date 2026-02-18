from __future__ import annotations
from typing import Dict, Any, Tuple, List
from pathlib import Path

# We use your PDESpec class
from ..pde_spec import PDESpec

from .heat_steady import solve_heat_steady
from .solid_linear import solve_solid_linear_elasticity
from .fluid_stokes import solve_fluid_steady_stokes
from .io import Artifact

def solve(spec: PDESpec, out_dir: Path) -> Tuple[Dict[str, Any], List[Artifact]]:
    """
    General entry point. Dispatch based on spec.domain + spec.pde.
    Returns:
      result: JSON-serializable dict (min/max, summary, etc.)
      artifacts: list of files (VTU/XDMF/JSON) written under out_dir
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    key = (spec.domain, spec.pde)

    if key == ("heat_transfer", "steady_heat"):
        return solve_heat_steady(spec, out_dir)
    if key == ("solid_mechanics", "linear_elasticity"):
        return solve_solid_linear_elasticity(spec, out_dir)
    if key == ("fluid", "steady_stokes"):
        return solve_fluid_steady_stokes(spec, out_dir)

    raise ValueError(f"Unsupported solver for domain={spec.domain}, pde={spec.pde}")
