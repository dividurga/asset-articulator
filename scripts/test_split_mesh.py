from __future__ import annotations

from pathlib import Path
import numpy as np
import trimesh

from asset_articulator.geometry.cuboid import OrientedCuboid
from asset_articulator.geometry.mesh_ops import split_mesh_by_cuboid


def main() -> None:
    input_path = Path("data/input/object.stl")
    output_dir = Path("data/output/split_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load_mesh(input_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh, got {type(mesh)}")

    print("Bounds:")
    print(mesh.bounds)
    print("Extents:")
    print(mesh.extents)

    cuboid = OrientedCuboid(
        center=np.array([0.0, 0.47, 0.85]),
        rotation=np.eye(3),
        extents=np.array([0.30, 0.3, 0.30]),
    )

    result = split_mesh_by_cuboid(mesh, cuboid)

    inside_path = output_dir / "inside_clip.stl"
    outside_path = output_dir / "outside_clip.stl"

    result.inside_mesh.export(inside_path)
    result.outside_mesh.export(outside_path)

    print(f"Saved: {inside_path}")
    print(f"Saved: {outside_path}")
    print(f"Inside faces:  {len(result.inside_mesh.faces)}")
    print(f"Outside faces: {len(result.outside_mesh.faces)}")


if __name__ == "__main__":
    main()