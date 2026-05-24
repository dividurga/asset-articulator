"""Sutherland-Hodgman polygon clipping for mesh selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import trimesh

from asset_articulator.geometry.cuboid import OrientedCuboid
from asset_articulator.geometry.cylinder import OrientedCylinder

ArrayF = npt.NDArray[np.float64]


_EPS = 1e-9


@dataclass
class MeshClipResult:
    """Result of clipping a mesh with an oriented cuboid or cylinder."""

    inside_mesh: trimesh.Trimesh
    outside_mesh: trimesh.Trimesh
    clip_loops: list[ArrayF]  # directed boundary loops lying on the cuboid surface


def split_mesh_by_cuboid_clip(
    mesh: trimesh.Trimesh,
    cuboid: OrientedCuboid,
    existing_cuboids: list[OrientedCuboid] | None = None,
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
    existing_cuboids
        Previously used cuboids.  If any triangle centroid lies inside both
        ``cuboid`` and one of these, a ``ValueError`` is raised indicating
        the overlap so that the user can reposition the selection.

    Returns
    -------
    MeshClipResult
        inside_mesh: geometry inside the cuboid
        outside_mesh: complementary geometry outside the cuboid

    Raises
    ------
    ValueError
        If ``existing_cuboids`` is provided and the new cuboid overlaps a
        previous one (i.e. at least one triangle centroid is inside both).
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")

    if len(mesh.faces) == 0:
        raise ValueError("Input mesh has no faces.")

    if existing_cuboids:
        centroids = mesh.triangles_center  # (F, 3)
        new_inside = cuboid.contains(centroids)
        for i, other in enumerate(existing_cuboids):
            overlap = new_inside & other.contains(centroids)
            if overlap.any():
                n = int(overlap.sum())
                raise ValueError(
                    f"New cuboid overlaps existing cuboid {i}: "
                    f"{n} triangle centroid(s) lie inside both. "
                    f"Reposition the selection so the cuboids do not intersect."
                )

    triangles_world = mesh.triangles
    triangles_local = cuboid.world_to_local(triangles_world.reshape(-1, 3)).reshape(
        -1, 3, 3
    )

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
        cuboid.local_to_world(
            np.asarray(inside_triangles_local).reshape(-1, 3)
        ).reshape(-1, 3, 3)
        if inside_triangles_local
        else np.zeros((0, 3, 3), dtype=float)
    )
    outside_mesh = _triangles_to_mesh(
        cuboid.local_to_world(
            np.asarray(outside_triangles_local).reshape(-1, 3)
        ).reshape(-1, 3, 3)
        if outside_triangles_local
        else np.zeros((0, 3, 3), dtype=float)
    )

    inside_mesh = _drop_small_components(inside_mesh)
    outside_mesh = _drop_small_components(outside_mesh)

    clip_loops = _extract_clip_loops(inside_mesh, cuboid)

    return MeshClipResult(
        inside_mesh=inside_mesh,
        outside_mesh=outside_mesh,
        clip_loops=clip_loops,
    )


def _drop_small_components(
    mesh: trimesh.Trimesh,
    min_ratio: float = 0.005,
    min_faces: int = 4,
) -> trimesh.Trimesh:
    """Remove disconnected components that are much smaller than the largest one.

    A component is kept if its face count >= max(min_faces, min_ratio * largest). With
    the default min_ratio=0.5%, a component must be at least 0.5% of the largest
    component to survive — this kills slivers from near-coincident cuts while preserving
    intentionally small geometry.
    """
    if len(mesh.faces) == 0:
        return mesh

    # merge vertices so adjacency works on unshared-vertex (STL-style) meshes
    merged = mesh.copy()
    merged.merge_vertices()
    components = merged.split(only_watertight=False)

    if len(components) <= 1:
        return mesh  # nothing to drop

    face_counts = np.array([len(c.faces) for c in components], dtype=np.int64)
    threshold = max(min_faces, int(np.ceil(face_counts.max() * min_ratio)))
    keep = [c for c, n in zip(components, face_counts) if n >= threshold]

    if len(keep) == len(components):
        return mesh  # nothing was small enough to drop

    if not keep:
        return mesh  # safety: never return empty when input was non-empty

    combined = trimesh.util.concatenate(keep)
    combined.remove_unreferenced_vertices()
    return combined


def _extract_clip_loops(
    inside_mesh: trimesh.Trimesh,
    cuboid: OrientedCuboid,
    tol_factor: float = 1e-5,
) -> list[ArrayF]:
    """Extract directed boundary loops that lie on the cuboid cut planes.

    After Sutherland-Hodgman clipping the boundary edges of *inside_mesh* that
    sit on a cuboid face are exactly the edges introduced by the cut.  We merge
    vertices, identify those edges, and chain them into one or more directed
    loops.

    Parameters
    ----------
    inside_mesh
        The clipped-inside result (process=False, unshared vertices).
    cuboid
        The cuboid used to produce the clip.
    tol_factor
        Fraction of the largest cuboid extent used as plane-proximity tolerance.

    Returns
    -------
    list of (N, 3) float arrays, each an ordered directed loop in world space.
    """
    if len(inside_mesh.faces) == 0:
        return []

    # Merge vertices so that shared edges are represented as matching indices.
    merged = inside_mesh.copy()
    merged.merge_vertices()
    merged.remove_unreferenced_vertices()

    if len(merged.faces) == 0:
        return []

    # Classify each vertex: is it on a cuboid face plane?
    verts_local = cuboid.world_to_local(merged.vertices)  # (V, 3)
    ex, ey, ez = cuboid.extents
    tol = tol_factor * max(ex, ey, ez)

    on_face: ArrayF = (
        (np.abs(np.abs(verts_local[:, 0]) - ex) < tol)
        | (np.abs(np.abs(verts_local[:, 1]) - ey) < tol)
        | (np.abs(np.abs(verts_local[:, 2]) - ez) < tol)
    )  # (V,) bool

    # Directed boundary half-edges: (a→b) whose reverse (b→a) is absent.
    edges = merged.edges  # (3*F, 2) directed half-edges
    edge_set = set(map(tuple, edges.tolist()))
    clip_directed: list[tuple[int, int]] = [
        (int(a), int(b))
        for a, b in edges
        if (int(b), int(a)) not in edge_set and bool(on_face[a]) and bool(on_face[b])
    ]

    if not clip_directed:
        return []

    # Build next-vertex map (assumes locally manifold boundary).
    next_v: dict[int, int] = dict(clip_directed)

    visited: set[int] = set()
    loops: list[ArrayF] = []

    for start, _ in clip_directed:
        if start in visited:
            continue
        loop_idx: list[int] = []
        cur = start
        for _ in range(len(clip_directed) + 1):
            if cur in visited and loop_idx:
                break
            visited.add(cur)
            loop_idx.append(cur)
            cur = next_v.get(cur, -1)
            if cur in (-1, start):
                break

        if len(loop_idx) >= 3:
            loops.append(merged.vertices[loop_idx].copy())

    return loops


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
        (0, ex, True),  # x <= ex
        (0, -ex, False),  # x >= -ex
        (1, ey, True),  # y <= ey
        (1, -ey, False),  # y >= -ey
        (2, ez, True),  # z <= ez
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


# ---------------------------------------------------------------------------
# Cylinder clipping
# ---------------------------------------------------------------------------


def _clip_polygon_by_halfplane(
    poly: ArrayF,
    normal: ArrayF,
    bound: float,
    keep_inside: bool,
) -> ArrayF | None:
    """Sutherland-Hodgman clip against dot(normal, p) <= bound (keep_inside=True)."""
    output: list[ArrayF] = []
    prev = poly[-1]
    prev_d = float(np.dot(normal, prev))
    prev_ok = (prev_d <= bound + _EPS) if keep_inside else (prev_d >= bound - _EPS)

    for curr in poly:
        curr_d = float(np.dot(normal, curr))
        curr_ok = (curr_d <= bound + _EPS) if keep_inside else (curr_d >= bound - _EPS)

        if curr_ok:
            if not prev_ok:
                denom = curr_d - prev_d
                t = (
                    float(np.clip((bound - prev_d) / denom, 0.0, 1.0))
                    if abs(denom) > _EPS
                    else 0.0
                )
                output.append(prev + t * (curr - prev))
            output.append(curr.copy())
        elif prev_ok:
            denom = curr_d - prev_d
            t = (
                float(np.clip((bound - prev_d) / denom, 0.0, 1.0))
                if abs(denom) > _EPS
                else 0.0
            )
            output.append(prev + t * (curr - prev))

        prev, prev_d, prev_ok = curr, curr_d, curr_ok

    if len(output) < 3:
        return None
    return _deduplicate_polygon_vertices(np.asarray(output, dtype=float))


def split_mesh_by_cylinder_clip(
    mesh: trimesh.Trimesh,
    cylinder: OrientedCylinder,
    N: int = 64,
) -> MeshClipResult:
    """Split a triangle mesh by an oriented cylinder.

    The cylinder is approximated as an N-gon prism (default N=64) in the
    cylinder's local frame.  Two flat cap planes handle the axial bounds;
    N radial half-planes approximate the curved wall.

    Parameters
    ----------
    mesh
        Input triangular mesh.
    cylinder
        Oriented cylinder in world coordinates.
    N
        Number of sides for the radial polygon approximation (default 64).
        At N=64 the max radial error is ~0.12% of the radius.

    Returns
    -------
    MeshClipResult
        inside_mesh / outside_mesh / clip_loops (on the flat cap planes).
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    if len(mesh.faces) == 0:
        raise ValueError("Input mesh has no faces.")

    triangles_world = mesh.triangles
    triangles_local = cylinder.world_to_local(triangles_world.reshape(-1, 3)).reshape(
        -1, 3, 3
    )

    h = cylinder.half_height
    r = cylinder.radius
    angles = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)

    # Flat cap planes (z ≤ h  and  -z ≤ h) + N circumscribed radial planes
    planes: list[tuple[ArrayF, float]] = [
        (np.array([0.0, 0.0, 1.0]), h),
        (np.array([0.0, 0.0, -1.0]), h),
    ]
    for a in angles:
        planes.append((np.array([float(np.cos(a)), float(np.sin(a)), 0.0]), r))

    inside_tris: list[ArrayF] = []
    outside_tris: list[ArrayF] = []

    for tri_local in triangles_local:
        inside_polys: list[ArrayF] = [tri_local]
        outside_polys: list[ArrayF] = []

        for plane_n, bound in planes:
            next_inside: list[ArrayF] = []
            for poly in inside_polys:
                kept = _clip_polygon_by_halfplane(poly, plane_n, bound, True)
                rejected = _clip_polygon_by_halfplane(poly, plane_n, bound, False)
                if kept is not None and len(kept) >= 3:
                    next_inside.append(kept)
                if rejected is not None and len(rejected) >= 3:
                    outside_polys.append(rejected)
            inside_polys = next_inside
            if not inside_polys:
                break

        for poly in inside_polys:
            inside_tris.extend(_triangulate_convex_polygon(poly))
        for poly in outside_polys:
            outside_tris.extend(_triangulate_convex_polygon(poly))

    def _to_world(tris: list[ArrayF]) -> ArrayF:
        if not tris:
            return np.zeros((0, 3, 3), dtype=float)
        return cylinder.local_to_world(np.asarray(tris).reshape(-1, 3)).reshape(
            -1, 3, 3
        )

    inside_mesh = _triangles_to_mesh(_to_world(inside_tris))
    outside_mesh = _triangles_to_mesh(_to_world(outside_tris))
    inside_mesh = _drop_small_components(inside_mesh)
    outside_mesh = _drop_small_components(outside_mesh)

    clip_loops = _extract_cylinder_clip_loops(inside_mesh, cylinder)

    return MeshClipResult(
        inside_mesh=inside_mesh,
        outside_mesh=outside_mesh,
        clip_loops=clip_loops,
    )


def _extract_cylinder_clip_loops(
    inside_mesh: trimesh.Trimesh,
    cylinder: OrientedCylinder,
    tol_factor: float = 1e-4,
) -> list[ArrayF]:
    """Extract directed boundary loops lying on the cylinder's flat cap planes.

    Only edges where both endpoints sit at |z| ≈ half_height (in cylinder local frame)
    are considered cut edges.
    """
    if len(inside_mesh.faces) == 0:
        return []

    merged = inside_mesh.copy()
    merged.merge_vertices()
    merged.remove_unreferenced_vertices()

    if len(merged.faces) == 0:
        return []

    verts_local = cylinder.world_to_local(merged.vertices)
    tol = tol_factor * cylinder.half_height

    on_cap: npt.NDArray[np.bool_] = (
        np.abs(np.abs(verts_local[:, 2]) - cylinder.half_height) < tol
    )

    edges = merged.edges
    edge_set = set(map(tuple, edges.tolist()))
    clip_directed: list[tuple[int, int]] = [
        (int(a), int(b))
        for a, b in edges
        if (int(b), int(a)) not in edge_set and bool(on_cap[a]) and bool(on_cap[b])
    ]

    if not clip_directed:
        return []

    next_v: dict[int, int] = dict(clip_directed)
    visited: set[int] = set()
    loops: list[ArrayF] = []

    for start, _ in clip_directed:
        if start in visited:
            continue
        loop_idx: list[int] = []
        cur = start
        for _ in range(len(clip_directed) + 1):
            if cur in visited and loop_idx:
                break
            visited.add(cur)
            loop_idx.append(cur)
            cur = next_v.get(cur, -1)
            if cur in (-1, start):
                break
        if len(loop_idx) >= 3:
            loops.append(merged.vertices[loop_idx].copy())

    return loops
