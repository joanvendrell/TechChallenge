import pyvista as pv
from pathlib import Path

path = Path("backend/artifacts/solution_p0_000001.vtu")

data = path.read_bytes()
print("bytes:", len(data))
print("first 200 bytes:\n", data[:200])

# also print first few lines if it's text/xml
try:
    txt = data[:2000].decode("utf-8", errors="replace")
    print("\n--- decoded head ---\n", "\n".join(txt.splitlines()[:30]))
except Exception as e:
    print("decode failed:", e)


obj = pv.read("backend/artifacts/solution_p0_000000.vtu")

# If it's a MultiBlock, grab first non-empty dataset
if isinstance(obj, pv.MultiBlock):
    mesh = None
    for i in range(len(obj)):
        if obj[i] is not None and obj[i].n_points > 0:
            mesh = obj[i]
            break
    if mesh is None:
        raise ValueError("All blocks empty or missing piece files.")
else:
    mesh = obj

p = pv.Plotter()
p.add_mesh(mesh)
p.show()