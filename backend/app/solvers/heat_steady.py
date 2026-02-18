from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import meshio
from mpi4py import MPI
import inspect
import basix
from dolfinx import mesh, fem, io, plot
from dolfinx.fem.petsc import LinearProblem
import ufl
import pyvista as pv
from .vtu_to_ply import vtu_piece_to_ply

from ..pde_spec import PDESpec
from .io import Artifact
import gmsh
from dolfinx.io import XDMFFile

def _create_sphere_mesh(comm, R: float, h: float, out_dir: Path):
    # --- Build gmsh model ---
    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.model.add("sphere")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, R)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(3)

    msh_path = out_dir / "sphere.msh"
    gmsh.write(str(msh_path))

    # Important in server context (avoid accumulating models)
    gmsh.model.remove()

    # --- Convert msh -> XDMF (volume mesh only) ---
    msh = meshio.read(str(msh_path))

    # Keep only tetrahedra (3D volume cells)
    tetra_cells = msh.get_cells_type("tetra")
    if tetra_cells is None or len(tetra_cells) == 0:
        raise RuntimeError("Gmsh mesh has no tetra cells (volume mesh).")

    # meshio wants a Mesh with points + tetra connectivity
    tetra_mesh = meshio.Mesh(points=msh.points, cells=[("tetra", tetra_cells)])

    xdmf_path = out_dir / "sphere.xdmf"
    meshio.write(str(xdmf_path), tetra_mesh)

    # --- Load XDMF into dolfinx ---
    with XDMFFile(comm, str(xdmf_path), "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    return domain



def solve_heat_steady(spec: PDESpec, out_dir: Path) -> Tuple[Dict[str, Any], List[Artifact]]:
    """
    Solve steady heat conduction:
        -div(k grad T) = 0
    on a box domain with Dirichlet BC on x=0 (T_left) and x=Lx (T_right).

    Writes: solution.vtu
    Returns: result dict + artifacts list
    """

    # ----------------------------
    # 1) Read geometry + mesh
    # ----------------------------
    gtype = getattr(spec.geometry, "type", "box")
    dims = spec.geometry.dims or {}

    geometry_requested = {"type": gtype, "dims": dict(dims)}
    warnings_out: List[str] = []
    geometry_used_for_mesh = None

    mparams = spec.mesh.params or {}
    nx = int(mparams.get("nx", 40))
    ny = int(mparams.get("ny", 40))
    nz = int(mparams.get("nz", 1))

    domain = None
    facet_tags = None  # optional, if you later want tagged boundaries

    # Resolve geometry / build mesh
    if gtype == "box":
        Lx = float(dims.get("Lx", 1.0))
        Ly = float(dims.get("Ly", 1.0))
        Lz = float(dims.get("Lz", 0.0))  # if 0 -> 2D

        geometry_used_for_mesh = {"type": "box" if Lz > 0 else "rectangle",
                                "dims": {"Lx": Lx, "Ly": Ly, **({} if Lz <= 0 else {"Lz": Lz})}}

    elif gtype == "thin_film":
        Lx = float(dims.get("Lx", 1.0e-6))
        Ly = float(dims.get("Ly", 1.0e-6))
        Lz = float(dims.get("Lz", 1.0e-9))
        nz = max(nz, 1)

        geometry_used_for_mesh = {"type": "box", "dims": {"Lx": Lx, "Ly": Ly, "Lz": Lz}}

    elif gtype == "sphere":
        if "R" not in dims:
            raise ValueError("Sphere geometry requires dims['R'] (radius).")
        R = float(dims["R"])

        # mesh size
        h = float(mparams.get("h", R / 10.0))

        domain = _create_sphere_mesh(MPI.COMM_WORLD, R, h, out_dir)

        geometry_used_for_mesh = {"type": "sphere", "dims": {"R": R}}
        warnings_out.append("Sphere meshed with gmsh (3D).")

    else:
        raise ValueError(
            f"Heat solver supports geometry.type in {{'box','thin_film','sphere'}}. Got '{gtype}'."
        )

    # Create mesh only if we haven't already (sphere creates it above)
    if domain is None:
        if Lz <= 0.0:
            domain = mesh.create_rectangle(
                MPI.COMM_WORLD,
                [np.array([0.0, 0.0]), np.array([Lx, Ly])],
                [nx, ny],
                cell_type=mesh.CellType.triangle,
            )
        else:
            domain = mesh.create_box(
                MPI.COMM_WORLD,
                [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
                [nx, ny, nz],
                cell_type=mesh.CellType.tetrahedron,
            )


    # ----------------------------
    # 2) Function space
    # ----------------------------
    V = fem.functionspace(domain, ("Lagrange", 1))
    T = fem.Function(V)
    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)

    # ----------------------------
    # 3) Material parameter k
    # ----------------------------
    k_val = float(spec.material.props.get("k", 1.0))
    k = fem.Constant(domain, k_val)

    # ----------------------------
    # 4) Boundary conditions
    # ----------------------------
    bc_kind = spec.bc.kind or {}
    T_left = float(bc_kind.get("T_left", 400.0))
    T_right = float(bc_kind.get("T_right", 300.0))

    bcs = []

    if gtype == "sphere":
        # For a sphere, "left/right boundary" doesn't exist.
        # MVP: apply Dirichlet on the entire surface using T_left.
        fdim = domain.topology.dim - 1

        def on_boundary(x):
            return np.full(x.shape[1], True, dtype=bool)

        boundary_facets = mesh.locate_entities_boundary(domain, fdim, on_boundary)
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

        bc_all = fem.dirichletbc(fem.Constant(domain, T_left), boundary_dofs, V)
        bcs = [bc_all]

        warnings_out.append(
            "For sphere: applied T_left as a Dirichlet condition on the entire surface (MVP behavior)."
        )

    else:
        # Box / rectangle: Dirichlet on x=0 and x=Lx
        def left_boundary(x):
            return np.isclose(x[0], 0.0)

        def right_boundary(x):
            return np.isclose(x[0], Lx)

        left_dofs = fem.locate_dofs_geometrical(V, left_boundary)
        right_dofs = fem.locate_dofs_geometrical(V, right_boundary)

        bc_left = fem.dirichletbc(fem.Constant(domain, T_left), left_dofs, V)
        bc_right = fem.dirichletbc(fem.Constant(domain, T_right), right_dofs, V)
        bcs = [bc_left, bc_right]


    # ----------------------------
    # 5) Variational form
    # ----------------------------
    a = ufl.inner(k * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, 0.0) * v * ufl.dx

    problem = LinearProblem(
                            a, L,
                            bcs=bcs,
                            petsc_options_prefix="heat_",
                            petsc_options={
                                "ksp_type": "cg",
                                "pc_type": "hypre"
                            },
                        )
    Th = problem.solve()
    Th.name = "temperature"
    # ----------------------------
    # 6) Export VTU (for VTK.js later)
    # ----------------------------
    # ----------------------------
    # 7) Compute basic stats (min/max)
    # ----------------------------
    # local values
    local = Th.x.array
    print("T min/max:", float(local.min()), float(local.max()))
    print("any NaN:", bool(np.isnan(local).any()))
    local_min = float(np.min(local)) if local.size else float("nan")
    local_max = float(np.max(local)) if local.size else float("nan")

    # global reduce
    comm = domain.comm
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)

    result: Dict[str, Any] = {
        "summary": "Solved steady heat conduction (Laplace) with Dirichlet BC on x=0 and x=Lx.",
        "field": "temperature",
        "units": "K",
        "k": k_val,
        "T_left": T_left,
        "T_right": T_right,
        "min": gmin,
        "max": gmax,
        "mesh": {"nx": nx, "ny": ny, "nz": nz, "dim": domain.topology.dim},
        "geometry_requested": geometry_requested,
        "geometry_used_for_mesh": geometry_used_for_mesh,
        "warnings": warnings_out,
    }

    # Prefer PVTU if it exists (vtk.js handles it well for parallel output)
    #ply_path = vtu_piece_to_ply(out_dir, prefer_rank=1)  # <-- this forces _000001
    #artifacts = [Artifact(name="solution.ply", path=ply_path, type="mesh")]
    topology, cell_types, geometry = plot.vtk_mesh(Th.function_space)

    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    # Attach temperature as point data
    grid.point_data["temperature"] = np.asarray(Th.x.array, dtype=float)
    grid.set_active_scalars("temperature")

    # Surface for web rendering
    surf = grid.extract_surface().triangulate()
    surf.set_active_scalars("temperature") 

    # Save (single file, no parallel piece headaches)
    vtp_path = out_dir / "solution.vtp"
    surf.save(str(vtp_path))

    artifacts = [
        # Artifact(name="solution.vtu", path=vtu_collection, type="vtk"),
        Artifact(name="solution.vtp", path=vtp_path, type="vtk"),
    ]
    return result, artifacts
