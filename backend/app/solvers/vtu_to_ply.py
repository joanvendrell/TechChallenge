from pathlib import Path
import pyvista as pv

def vtu_piece_to_ply(out_dir: Path, prefer_rank: int | None = None) -> Path:
    # 1) pick a VTU piece
    pieces = sorted(out_dir.glob("solution_p0_*.vtu"))
    if not pieces:
        raise FileNotFoundError("No solution_p0_*.vtu files found")

    if prefer_rank is not None:
        # match ..._000001.vtu style
        target = [p for p in pieces if p.name.endswith(f"{prefer_rank:06d}.vtu")]
        vtu_path = target[0] if target else pieces[0]
    else:
        # pick largest piece (often safest)
        vtu_path = max(pieces, key=lambda p: p.stat().st_size)

    # 2) read + convert
    grid = pv.read(str(vtu_path))          # UnstructuredGrid
    surf = grid.extract_surface()          # make PolyData surface
    ply_path = out_dir / "solution.ply"
    surf.save(str(ply_path))               # PLY is super easy to load in three.js

    return ply_path
