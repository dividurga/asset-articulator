"""Door geometry utilities."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import trimesh

from asset_articulator.geometry.cuboid import OrientedCuboid
from asset_articulator.geometry.cylinder import OrientedCylinder

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _connected_clusters(
    mesh: trimesh.Trimesh, face_indices: np.ndarray
) -> list[np.ndarray]:
    """Return connected components of face_indices using trimesh face adjacency.

    Merges duplicate vertices first so adjacency works on STL-style meshes where each
    triangle has its own unshared vertex copies.
    """
    selected_set = set(int(fi) for fi in face_indices)

    # Merge vertices so shared edges have matching indices
    merged = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    _log.debug(
        "[door] clustering: original %d verts → merged %d verts, face_adjacency pairs: %d",
        len(mesh.vertices),
        len(merged.vertices),
        len(merged.face_adjacency),
    )

    neighbors: dict[int, list[int]] = defaultdict(list)
    for a, b in merged.face_adjacency:  # pylint: disable=not-an-iterable
        a, b = int(a), int(b)
        if a in selected_set and b in selected_set:
            neighbors[a].append(b)
            neighbors[b].append(a)

    visited: set[int] = set()
    clusters: list[np.ndarray] = []
    for start in face_indices:
        start = int(start)
        if start in visited:
            continue
        cluster: list[int] = []
        queue = [start]
        visited.add(start)
        while queue:
            fi = queue.pop()
            cluster.append(fi)
            for nb in neighbors[fi]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        clusters.append(np.array(cluster, dtype=np.int64))
    return clusters


# ---------------------------------------------------------------------------
# Boundary loop extraction
# ---------------------------------------------------------------------------


def _boundary_loops(
    cluster_face_indices: np.ndarray, mesh_vertices: np.ndarray, mesh_faces: np.ndarray
) -> list[np.ndarray]:
    """Return ordered boundary loops (as arrays of 3-D vertex positions).

    Steps
    -----
    1. Build a submesh from the cluster with vertices merged (process=True) so
       that shared-edge detection works even on STL-style unshared vertices.
    2. Find directed boundary edges: edges whose reverse (b, a) does not appear
       among the mesh's directed half-edges.
    3. Chain them into one or more closed loops.
    4. Return each loop as an (N, 3) array of world-space positions.
    """
    # Build merged submesh so vertex indices are shared across faces
    sub = trimesh.Trimesh(
        vertices=mesh_vertices,
        faces=mesh_faces[cluster_face_indices],
        process=True,
    )

    _log.debug(
        f"[door]   submesh (merged): {len(sub.vertices)} verts, {len(sub.faces)} faces"
    )

    # All directed half-edges of the submesh
    edges = sub.edges  # shape (3 * n_faces, 2)
    edge_set = set(map(tuple, edges.tolist()))

    # A half-edge (a, b) is a boundary edge if (b, a) is absent
    boundary_directed = [
        (int(a), int(b))
        for a, b in edges  # pylint: disable=not-an-iterable
        if (int(b), int(a)) not in edge_set
    ]
    _log.debug(f"[door]   boundary directed edges: {len(boundary_directed)}")

    if not boundary_directed:
        return []

    # Build next-vertex map for each directed edge
    next_v: dict[int, int] = {}
    for a, b in boundary_directed:
        next_v[a] = b

    visited_starts: set[int] = set()
    loops: list[np.ndarray] = []

    for start, _ in boundary_directed:
        if start in visited_starts:
            continue
        loop_idx: list[int] = []
        cur = start
        for _ in range(len(boundary_directed) + 1):
            if cur in visited_starts and loop_idx:
                break
            visited_starts.add(cur)
            loop_idx.append(cur)
            cur = next_v.get(cur, -1)
            if cur in (-1, start):
                break

        if len(loop_idx) >= 3:
            loops.append(sub.vertices[loop_idx])
            _log.debug(f"[door]   loop: {len(loop_idx)} vertices")

    return loops


# ---------------------------------------------------------------------------
# Extrusion
# ---------------------------------------------------------------------------


def _extrude_loops(
    loops: list[np.ndarray], back_origin: np.ndarray, d: np.ndarray
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Extrude each boundary loop to the cuboid back plane.

    For each vertex v in the loop, the projected-back position is:
        v_back = v - max(0, dot(v - back_origin, d)) * d

    Geometry built:
    - Side walls: one quad (two triangles) per consecutive edge pair
    - Back cap: centroid fan of the projected loop

    Winding: side walls wound so normals point outward;
    back cap wound so normal points in -d direction.

    Returns
    -------
    mesh
        The extruded trimesh geometry.
    back_perimeter_edges
        (n, 2, 3) array of the back-panel perimeter edges for the largest
        loop, where each row is [v_start, v_end] in world-space coordinates.
    """
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0

    # Track the back vertices of the largest loop for perimeter edges
    chosen_back: np.ndarray | None = None
    largest_n = 0

    for loop_pts in loops:
        n = len(loop_pts)
        front = loop_pts  # (n, 3)

        back = np.array(
            [v - max(0.0, float(np.dot(v - back_origin, d))) * d for v in front]
        )  # (n, 3)

        if n > largest_n:
            largest_n = n
            chosen_back = back

        back_center = back.mean(axis=0)

        # Layout: [front_0..front_{n-1}, back_0..back_{n-1}, back_center]
        #          indices: 0..n-1,       n..2n-1,             2n
        verts = np.vstack([front, back, back_center.reshape(1, 3)])
        all_verts.append(verts)

        faces: list[list[int]] = []
        o = vert_offset
        for i in range(n):
            j = (i + 1) % n
            fi, fj = o + i, o + j
            bi, bj = o + n + i, o + n + j
            bc = o + 2 * n

            # Side wall quad: (front_i → front_j → back_j → back_i)
            # Normal points outward (away from loop interior)
            faces.append([fi, fj, bj])
            faces.append([fi, bj, bi])

            # Back cap: fan from back_center, wound to face -d
            faces.append([bc, bj, bi])

        all_faces.append(np.array(faces, dtype=np.int64))
        vert_offset += 2 * n + 1

    if not all_verts:
        return trimesh.Trimesh(), np.empty((0, 2, 3), dtype=float)

    verts_all = np.vstack(all_verts)
    faces_all = np.vstack(all_faces)
    mesh = trimesh.Trimesh(vertices=verts_all, faces=faces_all, process=True)
    trimesh.repair.fix_normals(mesh)  # type: ignore[no-untyped-call]

    # Back perimeter edges of the chosen (largest) loop: back[i] → back[(i+1)%n]
    assert chosen_back is not None
    n = len(chosen_back)
    back_perimeter_edges = np.stack(
        [chosen_back, chosen_back[np.arange(1, n + 1) % n]], axis=1
    )  # (n, 2, 3)

    _log.debug(
        f"[door]   extrusion: {len(mesh.vertices)} verts, {len(mesh.faces)} faces  "
        f"back_perimeter_edges={len(back_perimeter_edges)}"
    )
    return mesh, back_perimeter_edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cut_cuboid_with_surface(
    surface_mesh: trimesh.Trimesh,
    cuboid: OrientedCuboid,
    normal_threshold: float = 0.25,
    clip_loops: list[np.ndarray] | None = None,
    extrusion_axis: int | None = None,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Build door geometry via boundary-loop extrusion.

    Steps (heuristic path, when clip_loops is None)
    -----
    1. Find thin axis; auto-detect front direction from majority face normal.
    2. Select triangles whose normal · d > normal_threshold.
    3. Find connected clusters (via face_adjacency); pick the largest.
    4. Extract the boundary loop(s) of that cluster.
    5. Extrude each loop to the cuboid back plane along -d.
    6. Return full surface_mesh concatenated with the extrusion.

    When clip_loops is provided (preferred)
    ----------------------------------------
    Steps 2-4 are skipped entirely.  The supplied loops (from
    ``MeshClipResult.clip_loops``) are the directed boundary edges that the
    clipping operation produced on the cuboid face planes, which are exactly
    the perimeter of the cut opening.

    Parameters
    ----------
    surface_mesh
        The extracted visual surface (inside clip result).
    cuboid
        The cuboid used to clip the mesh.
    normal_threshold
        Min dot product of face normal with front direction (default 0.25).
        Ignored when clip_loops is provided.
    clip_loops
        Pre-computed directed boundary loops from ``MeshClipResult.clip_loops``.
        Each element is an (N, 3) world-space vertex array.
    """
    # --- Thin axis and front direction (always needed for extrusion) ------
    thin_axis = (
        extrusion_axis if extrusion_axis is not None else int(np.argmin(cuboid.extents))
    )
    d = cuboid.rotation[:, thin_axis].copy()
    d = d / np.linalg.norm(d)

    face_normals = surface_mesh.face_normals
    dot_pos = float((face_normals @ d).sum())
    dot_neg = float((face_normals @ (-d)).sum())
    _log.debug(f"[door] thin_axis={thin_axis}  extents={cuboid.extents}")
    _log.debug(
        f"[door] d (pre-flip)={d}  sum(n·d)={dot_pos:.3f}  sum(n·-d)={dot_neg:.3f}"
    )
    if dot_pos < dot_neg:
        d = -d
        _log.debug(f"[door] flipped d → {d}")
    else:
        _log.debug("[door] d kept as-is")

    back_origin = cuboid.center - cuboid.extents[thin_axis] * d
    _log.debug(f"[door] back_origin={back_origin}")

    # --- Loop source: clip-tracked or heuristic --------------------------
    if clip_loops is not None:
        loops = clip_loops
        _log.debug(f"[door] using {len(loops)} pre-computed clip loop(s)")
    else:
        # --- Normal-based selection --------------------------------------
        dots = face_normals @ d
        _log.debug(
            f"[door] face normal·d  min={dots.min():.3f}  max={dots.max():.3f}  "
            f"mean={dots.mean():.3f}  threshold={normal_threshold}"
        )
        selected = np.where(dots > normal_threshold)[0]
        _log.debug(
            f"[door] selected faces: {len(selected)} / {len(surface_mesh.faces)}"
        )

        if len(selected) == 0:
            _log.debug(
                "[door] WARNING: no faces selected — returning surface mesh unchanged"
            )
            return trimesh.Trimesh(
                vertices=surface_mesh.vertices.copy(),
                faces=surface_mesh.faces.copy(),
                process=False,
            ), np.empty((0, 2, 3), dtype=float)

        # --- Cluster and pick --------------------------------------------
        clusters = _connected_clusters(surface_mesh, selected)
        top_sizes = sorted([len(c) for c in clusters], reverse=True)[:10]
        _log.debug("[door] clusters: %d  sizes: %s", len(clusters), top_sizes)

        best_idx = int(np.argmax([len(c) for c in clusters]))
        best = clusters[best_idx]
        _log.debug(f"[door] chosen cluster: idx={best_idx}  size={len(best)}")

        # --- Boundary loops ----------------------------------------------
        loops = _boundary_loops(best, surface_mesh.vertices, surface_mesh.faces)
        _log.debug(f"[door] boundary loops found: {len(loops)}")

    if not loops:
        _log.debug("[door] WARNING: no loops — returning surface mesh unchanged")
        return trimesh.Trimesh(
            vertices=surface_mesh.vertices.copy(),
            faces=surface_mesh.faces.copy(),
            process=False,
        ), np.empty((0, 2, 3), dtype=float)

    # --- Extrude and combine ---------------------------------------------
    extrusion, back_perimeter_edges = _extrude_loops(loops, back_origin, d)

    combined = trimesh.util.concatenate([surface_mesh, extrusion])
    result = trimesh.Trimesh(
        vertices=combined.vertices,
        faces=combined.faces,
        process=True,
    )
    trimesh.repair.fix_normals(result)  # type: ignore[no-untyped-call]
    _log.debug(
        f"[door] final: {len(result.vertices)} verts, {len(result.faces)} faces  "
        f"watertight={result.is_watertight}"
    )
    return result, back_perimeter_edges


def cut_cylinder_with_surface(
    surface_mesh: trimesh.Trimesh,
    cylinder: OrientedCylinder,
    clip_loops: list[np.ndarray] | None = None,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Build cylinder door geometry via boundary-loop extrusion.

    Identical workflow to ``cut_cuboid_with_surface`` but uses the cylinder axis
    directly as the extrusion direction instead of detecting a thin cuboid axis.

    The flat-cap clip loops (from ``split_mesh_by_cylinder_clip``) are extruded to the
    far cap, producing the closing geometry for the rotating piece.
    """
    d = cylinder.axis.copy()
    face_normals = surface_mesh.face_normals
    dot_pos = float((face_normals @ d).sum())
    dot_neg = float((face_normals @ (-d)).sum())
    if dot_pos < dot_neg:
        d = -d

    back_origin = cylinder.center - cylinder.half_height * d

    if not clip_loops:
        _log.debug(
            "[door/cyl] WARNING: no clip_loops — returning surface mesh unchanged"
        )
        return trimesh.Trimesh(
            vertices=surface_mesh.vertices.copy(),
            faces=surface_mesh.faces.copy(),
            process=False,
        ), np.empty((0, 2, 3), dtype=float)

    extrusion, back_perimeter_edges = _extrude_loops(clip_loops, back_origin, d)
    combined = trimesh.util.concatenate([surface_mesh, extrusion])
    result = trimesh.Trimesh(
        vertices=combined.vertices, faces=combined.faces, process=True
    )
    trimesh.repair.fix_normals(result)  # type: ignore[no-untyped-call]
    _log.debug(
        f"[door/cyl] final: {len(result.vertices)} verts, {len(result.faces)} faces  "
        f"watertight={result.is_watertight}"
    )
    return result, back_perimeter_edges
