"""Door geometry test script.

Pipeline:
1. Load mesh (data/input/object.stl)
2. Clip with the hardcoded cuboid → inside (door surface) + outside (rest)
3. Extrude front-facing faces of the inside surface onto the cuboid's back face plane:
   - only faces with normal · thin_axis > FACE_THRESHOLD are used
   - each vertex is projected to the back plane (variable depth, coplanar back face)
4. Visualise (two panes: components | final door + rest)
5. Export to data/output/door_test/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from asset_articulator.geometry.clip import split_mesh_by_cuboid_clip
from asset_articulator.geometry.cuboid import OrientedCuboid
from asset_articulator.geometry.door import cut_cuboid_with_surface

# Hardcoded cuboid from cuboid_selector for cabinet mesh object.stl
# CUBOID = OrientedCuboid(
#     center=np.array([0.3333767934894833, -0.003317668775745429, 0.7440781564181818]),
#     rotation=np.array([
#         [0.7120260459909965, 0.0, 0.7021530529951624],
#         [0.7021530529951624, 0.0, -0.7120260459909965],
#         [0.0, 1.0, 0.0],
#     ]),
#     extents=np.array([0.23121737686787955, 0.6717011138123847, 0.0646907489331747]),
# )
# hardcoded cuboid from cuboid_selector for stacked washer dryer
CUBOID = OrientedCuboid(
    center=np.array([5.69383158566842, -156.44089812849145, 1562.0594939590533]),
    rotation=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    extents=np.array([273.6567554796125, 203.97563718619267, 54.119643312895924])
)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mesh_to_pv(mesh: trimesh.Trimesh) -> pv.PolyData:
    if len(mesh.faces) == 0:
        return pv.PolyData()
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
    return pv.PolyData(mesh.vertices, faces)


def cuboid_wireframe(cuboid: OrientedCuboid) -> pv.PolyData:
    e = cuboid.extents
    corners_local = np.array([
        [-e[0], -e[1], -e[2]], [ e[0], -e[1], -e[2]],
        [ e[0],  e[1], -e[2]], [-e[0],  e[1], -e[2]],
        [-e[0], -e[1],  e[2]], [ e[0], -e[1],  e[2]],
        [ e[0],  e[1],  e[2]], [-e[0],  e[1],  e[2]],
    ], dtype=float)
    corners_world = cuboid.local_to_world(corners_local)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    lines: list[int] = []
    for a, b in edges:
        lines.extend([2, a, b])
    pd = pv.PolyData()
    pd.points = corners_world
    pd.lines = np.array(lines, dtype=np.int32)
    return pd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Load mesh ---------------------------------------------------------
    candidates = [
        Path("data/input/Meshy_AI_Stacked Washer and Dryer_1775060331_texture.stl"),
        Path("data/input/faucet_modern.stl"),
    ]
    stl_path = next((p for p in candidates if p.exists()), None)
    if stl_path is None:
        raise FileNotFoundError(f"No input STL found. Tried: {[str(p) for p in candidates]}")

    mesh: trimesh.Trimesh = trimesh.load_mesh(stl_path, process=False)
    print(f"Loaded : {stl_path}")
    print(f"Verts  : {len(mesh.vertices)},  Faces: {len(mesh.faces)}")

    # --- Clip --------------------------------------------------------------
    print("\nClipping ...")
    result = split_mesh_by_cuboid_clip(mesh, CUBOID)
    inside_mesh = result.inside_mesh
    outside_mesh = result.outside_mesh
    print(f"  Inside  (door surface) : {len(inside_mesh.faces)} faces")
    print(f"  Outside (rest)         : {len(outside_mesh.faces)} faces")

    # --- Cut cuboid with surface + join -----------------------------------
    print(f"  Clip loops             : {len(result.clip_loops)}")
    door_mesh, _ = cut_cuboid_with_surface(
        inside_mesh, CUBOID,
        clip_loops=result.clip_loops or None,
    )
    print(f"\nDoor mesh  : {len(door_mesh.faces)} faces")

    # --- Export ------------------------------------------------------------
    out_dir = Path("data/output/door_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    inside_mesh.export(out_dir / "door_surface.stl")
    door_mesh.export(out_dir / "door_merged.stl")
    outside_mesh.export(out_dir / "rest.stl")
    print(f"\nExported to {out_dir}/")
    print("  door_surface.stl  — raw clip (visual surface)")
    print("  door_merged.stl   — visual surface + cut cuboid back half")
    print("  rest.stl          — remainder of object")

    # --- Visualise ---------------------------------------------------------
    plotter = pv.Plotter(shape=(1, 2), window_size=[1400, 750])

    plotter.subplot(0, 0)
    plotter.add_text("Components", position="upper_edge", font_size=12)
    plotter.add_mesh(mesh_to_pv(outside_mesh), color="lightgray", opacity=0.35, show_edges=False, label="rest")
    plotter.add_mesh(mesh_to_pv(inside_mesh), color="steelblue", opacity=0.9, show_edges=True, label="door surface (clip)")
    plotter.add_mesh(cuboid_wireframe(CUBOID), color="orange", line_width=4, label="cuboid")
    plotter.add_legend()
    plotter.add_axes()

    plotter.subplot(0, 1)
    plotter.add_text("Door + rest", position="upper_edge", font_size=12)
    plotter.add_mesh(mesh_to_pv(outside_mesh), color="lightgray", opacity=0.35, show_edges=False, label="rest")
    plotter.add_mesh(mesh_to_pv(door_mesh), color="salmon", opacity=0.85, show_edges=True, label="door (surface + cut cuboid)")
    plotter.add_legend()
    plotter.add_axes()

    plotter.show()


if __name__ == "__main__":
    main()
