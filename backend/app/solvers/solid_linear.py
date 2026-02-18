from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pyvista as pv
from mpi4py import MPI

from dolfinx import mesh, fem, plot
from dolfinx.fem.petsc import LinearProblem
import ufl

assert callable(ufl.Identity)

from ..pde_spec import PDESpec
from .io import Artifact

from dolfinx.io import XDMFFile
import meshio

def _create_sphere_mesh_xdmf(comm, R: float, h: float, out_dir: Path):
    import gmsh
    import numpy as np

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
    gmsh.model.remove()

    msh = meshio.read(str(msh_path))
    tetra = msh.get_cells_type("tetra")
    if tetra is None or len(tetra) == 0:
        raise RuntimeError("Gmsh mesh has no tetra cells.")

    tetmesh = meshio.Mesh(points=np.asarray(msh.points, dtype=np.float64),
                          cells=[("tetra", np.asarray(tetra))])

    xdmf_path = out_dir / "sphere.xdmf"
    meshio.write(str(xdmf_path), tetmesh)

    with XDMFFile(comm, str(xdmf_path), "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    return domain


def solve_solid_linear_elasticity(spec: PDESpec, out_dir: Path) -> Tuple[Dict[str, Any], List[Artifact]]:
    """
    Minimal linear elasticity:
      - geometry: beam box with dims L, W, H (meters)
      - BC: fixed x=0, traction on x=L face (tip load)
      - material: isotropic (E, nu)
      - outputs: displacement magnitude + von_mises on surface VTP
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # ----------------------------
    # 1) Read spec
    # ----------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    gtype = getattr(spec.geometry, "type", "beam")
    dims = spec.geometry.dims or {}
    geometry_requested = {"type": gtype, "dims": dict(dims)}
    warnings_out: List[str] = []
    geometry_used_for_mesh = None

    mp = spec.material.props or {}
    E = float(mp.get("E", 1e9))
    nu = float(mp.get("nu", 0.30))

    bc = spec.bc.kind or {}
    F = float(bc.get("F", 1000.0))
    direction = str(bc.get("direction", "down")).lower()

    mparams = spec.mesh.params or {}
    nx = int(mparams.get("nx", 30))
    ny = int(mparams.get("ny", 6))
    nz = int(mparams.get("nz", 6))

    domain = None

    if gtype in ("beam", "box"):
        L = float(dims.get("L", dims.get("Lx", 1.0)))
        W = float(dims.get("W", dims.get("Ly", 0.1)))
        H = float(dims.get("H", dims.get("Lz", 0.1)))

        domain = mesh.create_box(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0, 0.0]), np.array([L, W, H])],
            [nx, ny, nz],
            cell_type=mesh.CellType.tetrahedron,
        )
        geometry_used_for_mesh = {"type": "box", "dims": {"L": L, "W": W, "H": H}}

    elif gtype == "sphere":
        if "R" not in dims:
            raise ValueError("Sphere geometry requires dims['R'] (radius).")
        R = float(dims["R"])
        h = float(mparams.get("h", R / 10.0))

        domain = _create_sphere_mesh_xdmf(MPI.COMM_WORLD, R, h, out_dir)
        geometry_used_for_mesh = {"type": "sphere", "dims": {"R": R}}
        warnings_out.append("Sphere meshed via gmsh→meshio→XDMF (3D).")

    else:
        raise ValueError(f"Solid solver supports geometry.type in {{'beam','box','sphere'}}. Got '{gtype}'.")

    # ----------------------------
    # 3) Function space
    # ----------------------------
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))  # vector CG1
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Lame parameters
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lmbda * ufl.tr(eps(w)) * ufl.Identity(domain.geometry.dim)

    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    # ----------------------------
    # 4) Boundary conditions + load
    # ----------------------------
    fdim = domain.topology.dim - 1

    # Direction vector helper
    def _dir_to_vec(direction: str) -> np.ndarray:
        d = direction.lower()
        if d in ("down", "-y"):
            return np.array([0.0, -1.0, 0.0], dtype=np.float64)
        if d in ("up", "upward", "+y"):
            return np.array([0.0,  1.0, 0.0], dtype=np.float64)
        if d in ("-z", "into"):
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
        if d in ("+z", "out"):
            return np.array([0.0, 0.0,  1.0], dtype=np.float64)
        if d in ("+x", "right"):
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if d in ("-x", "left"):
            return np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)

    nvec = _dir_to_vec(direction)

    if gtype == "sphere":
        # Fix left hemisphere (x < 0)
        left_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: x[0] < 0.0)
        left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
        bc_left = fem.dirichletbc(np.array([0.0, 0.0, 0.0], dtype=np.float64), left_dofs, V)

        # Apply traction on right hemisphere (x > 0)
        right_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: x[0] > 0.0)

        facet_tags = mesh.meshtags(
            domain, fdim,
            right_facets.astype(np.int32),
            np.full(len(right_facets), 1, dtype=np.int32)
        )
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

        R = float(spec.geometry.dims["R"])
        area = 2.0 * np.pi * R * R  # hemisphere area
        traction_mag = F / area

        t = fem.Constant(domain, traction_mag * nvec)
        Lform = ufl.dot(t, v) * ds(1)
        bcs = [bc_left]

    else:
        # Cantilever beam/box: fix x=0, traction on x=L
        def left(x):  return np.isclose(x[0], 0.0)
        def right(x): return np.isclose(x[0], L)

        left_facets = mesh.locate_entities_boundary(domain, fdim, left)
        left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
        bc_left = fem.dirichletbc(np.array([0.0, 0.0, 0.0], dtype=np.float64), left_dofs, V)

        right_facets = mesh.locate_entities_boundary(domain, fdim, right)
        facet_tags = mesh.meshtags(
            domain, fdim,
            right_facets.astype(np.int32),
            np.full(len(right_facets), 1, dtype=np.int32)
        )
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

        area = W * H
        traction_mag = F / area

        t = fem.Constant(domain, traction_mag * nvec)
        Lform = ufl.dot(t, v) * ds(1)
        bcs = [bc_left]


    # ----------------------------
    # 5) Solve
    # ----------------------------
    problem = LinearProblem(
        a, Lform,
        bcs=[bc_left],
        petsc_options_prefix="elas_",
        petsc_options={"ksp_type": "cg", "pc_type": "hypre"},
    )
    uh = problem.solve()
    uh.name = "displacement"
    # ----------------------------
    # 6) Postprocess: displacement magnitude + von Mises
    # ----------------------------
    V0 = fem.functionspace(domain, ("Lagrange", 1))  # scalar CG1

    # |u|
    umag_expr = fem.Expression(ufl.sqrt(ufl.dot(uh, uh)), V0.element.interpolation_points)
    umag = fem.Function(V0)
    umag.interpolate(umag_expr)
    umag.name = "u_mag"

    # von Mises
    s = sigma(uh) - (1.0 / 3.0) * ufl.tr(sigma(uh)) * ufl.Identity(domain.geometry.dim)
    von = ufl.sqrt(3.0 / 2.0 * ufl.inner(s, s))
    von_expr = fem.Expression(von, V0.element.interpolation_points)
    vonf = fem.Function(V0)
    vonf.interpolate(von_expr)
    vonf.name = "von_mises"
    # ----------------------------
    # 7) Export surface VTP (robust)
    # ----------------------------
    topo, cell_types, geom = plot.vtk_mesh(V0)  # scalar space aligns with point dofs
    grid = pv.UnstructuredGrid(topo, cell_types, geom)

    grid.point_data["u_mag"] = np.asarray(umag.x.array, dtype=float)
    grid.point_data["von_mises"] = np.asarray(vonf.x.array, dtype=float)
    grid.set_active_scalars("u_mag")

    surf = grid.extract_surface().triangulate()
    surf.set_active_scalars("u_mag")

    vtp_path = out_dir / "solution.vtp"
    surf.save(str(vtp_path))
    # ----------------------------
    # 8) Stats
    # ----------------------------
    local_um = umag.x.array
    local_von = vonf.x.array
    comm = domain.comm

    umin = comm.allreduce(float(local_um.min()), op=MPI.MIN)
    umax = comm.allreduce(float(local_um.max()), op=MPI.MAX)
    vmin = comm.allreduce(float(local_von.min()), op=MPI.MIN)
    vmax = comm.allreduce(float(local_von.max()), op=MPI.MAX)

    result: Dict[str, Any] = {
        "summary": "Solved 3D linear elasticity with cantilever fixed at x=0 and traction at x=L.",
        "fields": ["u_mag", "von_mises"],
        "units": {"u_mag": "m", "von_mises": "Pa"},
        "E": E,
        "nu": nu,
        "F": F,
        "traction": traction_mag,
        "mesh": {"nx": nx, "ny": ny, "nz": nz, "dim": domain.topology.dim},
        "geometry_requested": geometry_requested,
        "geometry_used_for_mesh": geometry_used_for_mesh,
        "warnings": warnings_out,
        "ranges": {
            "u_mag": {"min": umin, "max": umax},
            "von_mises": {"min": vmin, "max": vmax},
        },
    }
    artifacts = [Artifact(name="solution.vtp", path=vtp_path, type="vtk")]
    return result, artifacts
