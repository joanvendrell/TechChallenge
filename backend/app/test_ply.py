import pyvista as pv

""" 
mesh = pv.read("backend/artifacts/ece23dba-257a-44ea-a850-34669315e268/solution.ply")

print(mesh)   # always do this once — great debugging info

p = pv.Plotter()
p.add_mesh(
    mesh,
    smooth_shading=True,
    specular=0.2,
)
p.show_grid()
p.show()
"""

""" 
mesh = pv.read("backend/artifacts/solution.vtp")   # path to your .vtp
print(mesh)                       # optional: inspect

p = pv.Plotter()
p.add_mesh(mesh, show_edges=False)
p.show_axes()
p.show()
"""

m = pv.read("backend/artifacts/solution.vtp")
print(m.point_data.keys(), m.cell_data.keys())
print(m.active_scalars_name)