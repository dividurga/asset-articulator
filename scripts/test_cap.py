"""Cap test script.

Loads an open mesh, caps the selected boundary loop, exports capped STL,
and visualizes the capped mesh with any remaining selected boundary loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from asset_articulator.geometry.cap import Cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def selected_loop_polydata(capper: Cap, mesh: trimesh.Trimesh) -> pv.PolyData | None:
    """Return selected boundary loop as PolyData lines, or None if no loop."""
    segments = capper.selected_boundary_segments(mesh)
    if len(segments) == 0:
        return None

    pts = segments.reshape(-1, 3)
    lines: list[int] = []
    for idx in range(len(segments)):
        lines.extend([2, 2 * idx, 2 * idx + 1])

    pd = pv.PolyData()
    pd.points = pts
    pd.lines = np.array(lines, dtype=np.int32)
    return pd


def build_selection_metadata() -> dict[str, np.ndarray] | None:
    """Paste current cuboid values here to target only the selection-created loop."""
    center = None
    rotation = None
    extents = None

    if center is None or rotation is None or extents is None:
        return None

    return {
        "center": np.asarray(center, dtype=float),
        "rotation": np.asarray(rotation, dtype=float),
        "extents": np.asarray(extents, dtype=float),
    }


def mesh_to_pv(mesh: trimesh.Trimesh) -> pv.PolyData:
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
    return pv.PolyData(mesh.vertices, faces)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    stl_path = Path("data/output/split_test/selection_clip.stl")
    output_stl = Path("data/output/split_test/selection_clip_capped.stl")
    if not stl_path.exists():
        raise FileNotFoundError(
            f"{stl_path} not found — run cuboid_selector.py and save split meshes first."
        )

    mesh_open: trimesh.Trimesh = trimesh.load_mesh(stl_path, process=False)
    print(f"Loaded: {len(mesh_open.vertices)} verts, {len(mesh_open.faces)} faces")
    print(f"Is watertight before cap: {mesh_open.is_watertight}")

    selection_metadata = build_selection_metadata()
    capper = Cap(stl_path, selection_metadata=selection_metadata)
    all_loops = capper.boundary_loops(mesh_open)
    selected_loop = capper.select_boundary_loop(mesh_open)
    print(f"Detected boundary loops: {len(all_loops)}")
    if selected_loop is None:
        print("Selected loop length: 0")
    else:
        print(f"Selected loop length: {len(selected_loop)}")
    if selection_metadata is None:
        print("Selection metadata: missing (fallback = longest loop)")
    else:
        print("Selection metadata: provided (loop scored against cuboid faces)")

    mesh_capped = capper.cap_mesh(mesh_open)
    mesh_capped.export(output_stl)
    print(f"Exported capped STL: {output_stl}")
    print(f"Is watertight after cap: {mesh_capped.is_watertight}")

    capper_after = Cap(output_stl, selection_metadata=selection_metadata)
    loops_after = capper_after.boundary_loops(mesh_capped)
    selected_loop_after = capper_after.select_boundary_loop(mesh_capped)
    print(f"Boundary loops after cap: {len(loops_after)}")
    if selected_loop_after is None:
        print("Selected loop after cap length: 0")
    else:
        print(f"Selected loop after cap length: {len(selected_loop_after)}")

    # --- Visualise capped mesh + selected boundary loop --------------------
    plotter: pv.Plotter = pv.Plotter(window_size=[1100, 700])
    plotter.add_text("Capped mesh + selected boundary loop", position="upper_edge", font_size=12)
    plotter.add_mesh(mesh_to_pv(mesh_capped), color="lightblue", opacity=0.85, show_edges=True)
    loop_pd = selected_loop_polydata(capper_after, mesh_capped)
    if loop_pd is not None:
        plotter.add_mesh(loop_pd, color="red", line_width=6, label="remaining selected loop")
        pv.Plotter.add_legend(plotter)
    else:
        plotter.add_text("(no boundary loop found)", position="lower_edge", font_size=9, color="green")
    pv.Plotter.add_axes(plotter)
    plotter.show()


if __name__ == "__main__":
    main()
