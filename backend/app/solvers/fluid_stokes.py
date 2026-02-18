from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pyvista as pv
from mpi4py import MPI

import ufl
from dolfinx import mesh, fem, plot
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element

from ..pde_spec import PDESpec
from .io import Artifact  # adjust import to your project


def solve_fluid_steady_stokes(spec: PDESpec, out_dir: Path) -> Tuple[Dict[str, Any], List[Artifact]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = spec.geometry.dims
    Lx = float(dims.get("Lx", 1.0))
    Ly = float(dims.get("Ly", 0.2))
    Lz = float(dims.get("Lz", 0.2))

    mu = float((spec.material.props or {}).get("mu", 1e-3))

    bc = spec.bc.kind or {}
    U_in = float(bc.get("U_in", 0.2))
    p_out = float(bc.get("p_out", 0.0))

    nx = int(spec.mesh.params.get("nx", 40))
    ny = int(spec.mesh.params.get("ny", 16))
    nz = int(spec.mesh.params.get("nz", 16))

    # 3D channel mesh
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )

    # Taylor-Hood: velocity P2, pressure P1 (stable Stokes)
    # velocity element (vector)
    Ve = element(
        "Lagrange",
        domain.topology.cell_name(),
        2,
        shape=(domain.geometry.dim,)
    )
    # pressure element (scalar)
    Qe = element(
        "Lagrange",
        domain.topology.cell_name(),
        1
    )
    # mixed element
    We = mixed_element([Ve, Qe])
    W = fem.functionspace(domain, We)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Facet markers
    fdim = domain.topology.dim - 1

    def inlet(x):  return np.isclose(x[0], 0.0)
    def outlet(x): return np.isclose(x[0], Lx)
    def walls(x):
        return (
            np.isclose(x[1], 0.0) | np.isclose(x[1], Ly) |
            np.isclose(x[2], 0.0) | np.isclose(x[2], Lz)
        )

    inlet_facets  = mesh.locate_entities_boundary(domain, fdim, inlet)
    outlet_facets = mesh.locate_entities_boundary(domain, fdim, outlet)
    wall_facets   = mesh.locate_entities_boundary(domain, fdim, walls)

    # BCs:
    # --- Velocity subspace (mixed) ---
    V_sub = W.sub(0)
    V, _ = V_sub.collapse()

    # inlet profile function (uniform)
    u_in_fun = fem.Function(V)
    u_in_fun.name = "u_inlet"

    def inlet_profile(x):
        vals = np.zeros((domain.geometry.dim, x.shape[1]), dtype=np.float64)
        vals[0, :] = U_in
        return vals

    u_in_fun.interpolate(inlet_profile)

    #inlet_dofs = fem.locate_dofs_topological(V_sub, fdim, inlet_facets)
    #bc_in   = fem.dirichletbc(u_in_fun, inlet_dofs)
    inlet_dofs = fem.locate_dofs_topological((V_sub, V), fdim, inlet_facets)
    bc_in = fem.dirichletbc(u_in_fun, inlet_dofs, V_sub)

    # no-slip wall function
    u0_fun = fem.Function(V)
    u0_fun.name = "u_wall"
    u0_fun.x.array[:] = 0.0

    #wall_dofs = fem.locate_dofs_topological(V_sub, fdim, wall_facets)
    #bc_wall = fem.dirichletbc(u0_fun, wall_dofs)
    wall_dofs = fem.locate_dofs_topological((V_sub, V), fdim, wall_facets)
    bc_wall = fem.dirichletbc(u0_fun, wall_dofs, V_sub)

    # --- Pressure subspace (mixed) ---
    Q_sub = W.sub(1)
    Q, _ = Q_sub.collapse()

    p_out_fun = fem.Function(Q)
    p_out_fun.name = "p_out"
    p_out_fun.x.array[:] = p_out

    #outlet_pdofs = fem.locate_dofs_topological(Q_sub, fdim, outlet_facets)
    #bc_pout = fem.dirichletbc(p_out_fun, outlet_pdofs)
    outlet_pdofs = fem.locate_dofs_topological((Q_sub, Q), fdim, outlet_facets)
    bc_pout = fem.dirichletbc(p_out_fun, outlet_pdofs, Q_sub)

    bcs = [bc_in, bc_wall, bc_pout]
    print("inlet facets:", len(inlet_facets), "inlet dofs:", len(inlet_dofs))
    print("wall facets:", len(wall_facets), "wall dofs:", len(wall_dofs))
    print("outlet facets:", len(outlet_facets), "outlet dofs:", len(outlet_pdofs))



    # Stokes weak form:
    # mu * <grad u, grad v> - <p, div v> + <q, div u> = 0
    a = (
        mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(q, ufl.div(u)) * ufl.dx
    )
    f = fem.Constant(domain, np.array([0.0, 0.0, 0.0], dtype=np.float64))
    L = ufl.inner(f, v) * ufl.dx  

    wh = fem.Function(W)
    problem = LinearProblem(
        a, L,
        bcs=bcs,
        u=wh,
        petsc_options_prefix="stokes_",
        petsc_options = {
                "ksp_type": "fgmres",
                "ksp_rtol": 1e-8,

                "pc_type": "fieldsplit",
                "pc_fieldsplit_detect_saddle_point": None,
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_factorization_type": "full",

                # KEY: use a Schur preconditioner that doesn't require diagonal of schurcomplement
                "pc_fieldsplit_schur_precondition": "selfp",

                # Velocity block solver
                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_0_pc_type": "hypre",

                # Pressure block solver (for the preconditioner / mass matrix)
                "fieldsplit_1_ksp_type": "preonly",
                "fieldsplit_1_pc_type": "hypre",

                "ksp_monitor": None,
                "ksp_converged_reason": None,
                }
    )
    print("Solving the problem.....")
    wh = problem.solve()
    print("wh (mixed) size:", wh.x.array.size)
    print("wh (mixed) min/max:", float(wh.x.array.min()), float(wh.x.array.max()))
    print("wh any NaN:", bool(np.isnan(wh.x.array).any()))

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()
    print("uh size:", uh.x.array.size)
    print("uh min/max:", float(uh.x.array.min()), float(uh.x.array.max()))
    print("uh any NaN:", bool(np.isnan(uh.x.array).any()))

    print("ph size:", ph.x.array.size)
    print("ph min/max:", float(ph.x.array.min()), float(ph.x.array.max()))

    uh.name = "velocity"
    ph.name = "pressure"

    # velocity magnitude (scalar CG1 for easy VTK mapping)
    V0 = fem.functionspace(domain, ("Lagrange", 1))
    ip_attr = V0.element.interpolation_points
    ip = ip_attr() if callable(ip_attr) else ip_attr

    umag_expr = fem.Expression(ufl.sqrt(ufl.dot(uh, uh)), ip)
    umag = fem.Function(V0)
    umag.interpolate(umag_expr)
    umag.name = "velocity_mag"

    # Build VTK mesh from scalar space (matches point dofs)
    topo, cell_types, geom = plot.vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topo, cell_types, geom)
    grid.point_data["velocity_mag"] = np.asarray(umag.x.array, dtype=float)

    # also export pressure (interpolate pressure to CG1)
    ph1 = fem.Function(V0)
    ph_expr = fem.Expression(ph, ip)
    ph1.interpolate(ph_expr)
    ph1.name = "pressure"
    grid.point_data["pressure"] = np.asarray(ph1.x.array, dtype=float)

    grid.set_active_scalars("velocity_mag")
    surf = grid.extract_surface().triangulate()
    surf.set_active_scalars("velocity_mag")

    vtp_path = out_dir / "solution.vtp"
    surf.save(str(vtp_path))

    # Stats
    comm = domain.comm
    local = umag.x.array
    print("velocity_mag DOFs:", local.size)
    print("min/max:", float(local.min()), float(local.max()))
    print("std:", float(local.std()))
    print("any NaN:", bool(np.isnan(local).any()))
    vmin = comm.allreduce(float(local.min()), op=MPI.MIN)
    vmax = comm.allreduce(float(local.max()), op=MPI.MAX)

    result: Dict[str, Any] = {
        "summary": "Solved steady Stokes flow in a 3D channel with inlet velocity, no-slip walls, and outlet pressure.",
        "default_field": "velocity_mag",
        "fields": ["velocity_mag", "pressure"],
        "units": {"velocity_mag": "m/s", "pressure": "Pa"},
        "mu": mu,
        "U_in": U_in,
        "p_out": p_out,
        "min": vmin,
        "max": vmax,
        "mesh": {"nx": nx, "ny": ny, "nz": nz, "dim": domain.topology.dim},
        "geometry": {"Lx": Lx, "Ly": Ly, "Lz": Lz},
    }

    artifacts = [Artifact(name="solution.vtp", path=vtp_path, type="vtk")]
    return result, artifacts
