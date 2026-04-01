"""Drawer geometry utilities."""

from __future__ import annotations

import numpy as np
import trimesh

from asset_articulator.geometry.cuboid import OrientedCuboid
from asset_articulator.geometry.door import cut_cuboid_with_surface, solid_box_from_cuboid

def solid_box_from_cuboid(cuboid: OrientedCuboid) -> trimesh.Trimesh:
    """Create a watertight solid box matching the cuboid's shape and orientation."""
    T = np.eye(4)
    T[:3, :3] = cuboid.rotation
    T[:3, 3] = cuboid.center
    return trimesh.creation.box(extents=2.0 * cuboid.extents, transform=T)


def cut_cuboid_with_surface(
    surface_mesh: trimesh.Trimesh,
    cuboid: OrientedCuboid,
) -> trimesh.Trimesh:
    """Use the surface mesh to cut the solid cuboid, then join the back half with the surface.

    Equivalent to the Blender workflow:
      create cuboid → knife-project surface onto cuboid → keep back half → join with surface.

    Steps
    -----
    1. Build solid box from cuboid.
    2. Find the thin axis (door depth direction) and auto-detect which way the surface faces.
    3. Compute the cut plane: origin = mean vertex position of the surface along the thin axis,
       normal = thin axis direction (pointing toward front).
    4. Slice the solid box with that plane — keep only the back half.
    5. Concatenate back half + visual surface.

    Parameters
    ----------
    surface_mesh
        The extracted visual surface (inside clip result).
    cuboid
        The cuboid used to clip the mesh.

    Returns
    -------
    trimesh.Trimesh
        Visual surface as front + cut-back cuboid as solid backing.
    """
    # --- Thin axis in world space ------------------------------------------
    thin_axis = int(np.argmin(cuboid.extents))
    d = cuboid.rotation[:, thin_axis].copy()
    d /= np.linalg.norm(d)

    # Auto-detect sign: surface normals tell us which way is "front"
    face_normals = surface_mesh.face_normals
    if (face_normals @ d).sum() < (face_normals @ (-d)).sum():
        d = -d

    # --- Cut plane --------------------------------------------------------
    # Origin: project each surface vertex onto the thin axis, use the mean depth.
    # This gives a plane that cuts through the "average" depth of the door surface.
    depths = surface_mesh.vertices @ d
    cut_origin = np.mean(depths) * d  # a point on the thin axis at that depth

    # --- Slice solid box --------------------------------------------------
    solid = solid_box_from_cuboid(cuboid)

    # slice_mesh_plane keeps the side where dot(pt - origin, plane_normal) >= 0.
    # We want the back half (opposite to d), so plane_normal = -d.
    back_half = trimesh.intersections.slice_mesh_plane(
        solid,
        plane_normal=-d,
        plane_origin=cut_origin,
        cap=False,
    )

    # --- Join back half + visual surface ----------------------------------
    combined = trimesh.util.concatenate([surface_mesh, back_half])
    return trimesh.Trimesh(
        vertices=combined.vertices,
        faces=combined.faces,
        process=False,
    )
