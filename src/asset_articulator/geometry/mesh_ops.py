from __future__ import annotations

from asset_articulator.geometry.clip import MeshClipResult, split_mesh_by_cuboid_clip
from asset_articulator.geometry.cuboid import OrientedCuboid
import trimesh


def split_mesh_by_cuboid(
    mesh: trimesh.Trimesh,
    cuboid: OrientedCuboid,
) -> MeshClipResult:
    """Precisely split a mesh by an oriented cuboid."""
    return split_mesh_by_cuboid_clip(mesh, cuboid)


