from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)

V = fem.functionspace(domain, ("Lagrange", 1))

u = fem.Function(V)
u.interpolate(lambda x: x[0] + x[1])

print("FEM works ✅")
