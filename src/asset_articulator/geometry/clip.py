from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt
import trimesh

from asset_articulator.geometry.cuboid import OrientedCuboid


ArrayF = npt.NDArray[np.float64]


_EPS = 1e-9


@dataclass
class MeshClipResult:
    inside_mesh: trimesh.Trimesh
    outside_mesh: trimesh.Trimesh


def split_mesh_by_cuboid_clip(
    mesh: trimesh.Trimesh,
    cuboid: OrientedCuboid,
) -> MeshClipResult:
    """Precisely split a triangle mesh by an oriented cuboid using plane clipping.

    The split is geometric, not centroid-based:
    - triangles are clipped against the 6 cuboid planes
    - both inside and outside outputs are generated

    Parameters
    ----------
    mesh
        Input triangular mesh.
    cuboid
        Oriented cuboid in world coordinates.

    Returns
    -------
    MeshClipResult
        inside_mesh: geometry inside the cuboid
        outside_mesh: complementary geometry outside the cuboid
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")

    if len(mesh.faces) == 0:
        raise ValueError("Input mesh has no faces.")

    triangles_world = mesh.triangles
    triangles_local = cuboid.world_to_local(
        triangles_world.reshape(-1, 3)
    ).reshape(-1, 3, 3)

    inside_triangles_local: list[ArrayF] = []
    outside_triangles_local: list[ArrayF] = []

    planes = _cuboid_planes(cuboid.extents)

    for tri_local in triangles_local:
        inside_polys = [tri_local]
        outside_polys: list[ArrayF] = []

        for axis, bound, keep_less_equal in planes:
            next_inside: list[ArrayF] = []

            for poly in inside_polys:
                kept, rejected = _split_polygon_by_plane(
                    poly=poly,
                    axis=axis,
                    bound=bound,
                    keep_less_equal=keep_less_equal,
                )
                if kept is not None and len(kept) >= 3:
                    next_inside.append(kept)
                if rejected is not None and len(rejected) >= 3:
                    outside_polys.append(rejected)

            inside_polys = next_inside
            if not inside_polys:
                break

        for poly in inside_polys:
            inside_triangles_local.extend(_triangulate_convex_polygon(poly))

        for poly in outside_polys:
            outside_triangles_local.extend(_triangulate_convex_polygon(poly))

    inside_mesh = _triangles_to_mesh(
        cuboid.local_to_world(np.asarray(inside_triangles_local).reshape(-1, 3))
        .reshape(-1, 3, 3)
        if inside_triangles_local
        else np.zeros((0, 3, 3), dtype=float)
    )
    outside_mesh = _triangles_to_mesh(
        cuboid.local_to_world(np.asarray(outside_triangles_local).reshape(-1, 3))
        .reshape(-1, 3, 3)
        if outside_triangles_local
        else np.zeros((0, 3, 3), dtype=float)
    )

    return MeshClipResult(
        inside_mesh=inside_mesh,
        outside_mesh=outside_mesh,
    )


def _cuboid_planes(extents: ArrayF) -> list[tuple[int, float, bool]]:
    """Return cuboid half-space planes in local coordinates.

    Each plane is represented as:
    (axis, bound, keep_less_equal)

    Meaning:
    - if keep_less_equal is True, keep points with coord <= bound
    - else keep points with coord >= bound
    """
    ex, ey, ez = extents
    return [
        (0, ex, True),    # x <= ex
        (0, -ex, False),  # x >= -ex
        (1, ey, True),    # y <= ey
        (1, -ey, False),  # y >= -ey
        (2, ez, True),    # z <= ez
        (2, -ez, False),  # z >= -ez
    ]


def _split_polygon_by_plane(
    poly: ArrayF,
    axis: int,
    bound: float,
    keep_less_equal: bool,
) -> tuple[ArrayF | None, ArrayF | None]:
    """Split a polygon by a plane into kept and rejected polygons.

    The polygon is assumed planar and vertices are ordered.
    """
    kept = _clip_polygon_halfspace(poly, axis, bound, keep_less_equal)
    rejected = _clip_polygon_halfspace(poly, axis, bound, not keep_less_equal)
    return kept, rejected


def _clip_polygon_halfspace(
    poly: ArrayF,
    axis: int,
    bound: float,
    keep_less_equal: bool,
) -> ArrayF | None:
    """Clip a polygon against a single half-space using Sutherland-Hodgman."""
    if len(poly) == 0:
        return None

    output: list[ArrayF] = []

    prev = poly[-1]
    prev_inside = _point_in_halfspace(prev, axis, bound, keep_less_equal)

    for curr in poly:
        curr_inside = _point_in_halfspace(curr, axis, bound, keep_less_equal)

        if curr_inside:
            if not prev_inside:
                inter = _segment_plane_intersection(prev, curr, axis, bound)
                output.append(inter)
            output.append(curr)
        elif prev_inside:
            inter = _segment_plane_intersection(prev, curr, axis, bound)
            output.append(inter)

        prev = curr
        prev_inside = curr_inside

    if len(output) < 3:
        return None

    cleaned = _deduplicate_polygon_vertices(np.asarray(output, dtype=float))
    if cleaned is None or len(cleaned) < 3:
        return None

    return cleaned


def _point_in_halfspace(
    point: ArrayF,
    axis: int,
    bound: float,
    keep_less_equal: bool,
) -> bool:
    value = point[axis]
    if keep_less_equal:
        return value <= bound + _EPS
    return value >= bound - _EPS


def _segment_plane_intersection(
    p0: ArrayF,
    p1: ArrayF,
    axis: int,
    bound: float,
) -> ArrayF:
    """Return intersection point of segment p0->p1 with plane coord[axis] = bound."""
    d0 = p0[axis] - bound
    d1 = p1[axis] - bound
    denom = d0 - d1

    if abs(denom) < _EPS:
        return p0.copy()

    t = d0 / denom
    t = np.clip(t, 0.0, 1.0)
    return p0 + t * (p1 - p0)


def _deduplicate_polygon_vertices(poly: ArrayF) -> ArrayF | None:
    """Remove consecutive duplicate/near-duplicate vertices."""
    if len(poly) == 0:
        return None

    cleaned = [poly[0]]
    for pt in poly[1:]:
        if np.linalg.norm(pt - cleaned[-1]) > _EPS:
            cleaned.append(pt)

    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= _EPS:
        cleaned.pop()

    if len(cleaned) < 3:
        return None

    return np.asarray(cleaned, dtype=float)


def _triangulate_convex_polygon(poly: ArrayF) -> list[ArrayF]:
    """Fan triangulate a convex polygon."""
    if len(poly) < 3:
        return []

    tris: list[ArrayF] = []
    for i in range(1, len(poly) - 1):
        tri = np.asarray([poly[0], poly[i], poly[i + 1]], dtype=float)
        if _triangle_area(tri) > _EPS:
            tris.append(tri)
    return tris


def _triangle_area(tri: ArrayF) -> float:
    return float(0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))


def _triangles_to_mesh(triangles: ArrayF) -> trimesh.Trimesh:
    """Build a trimesh from explicit triangles, preserving splits."""
    if len(triangles) == 0:
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3), dtype=float),
            faces=np.zeros((0, 3), dtype=int),
            process=False,
        )

    vertices = triangles.reshape(-1, 3)
    n_tri = len(triangles)
    faces = np.arange(3 * n_tri, dtype=int).reshape(n_tri, 3)

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)