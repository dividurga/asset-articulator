"""Door geometry utilities."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh

from asset_articulator.geometry.cuboid import OrientedCuboid
from asset_articulator.geometry.door import cut_cuboid_with_surface

# ---------------------------------------------------------------------------
# Clustering 
# ---------------------------------------------------------------------------

def _build_drawer(
    front_face: trimesh.Trimesh,
    perimeter: np.ndarray,
    thickness: float | None,
    cuboid: OrientedCuboid,
    thin_axis: int,
) -> trimesh.Trimesh:
    """Build a drawer geometry from the front face and boundary perimeter."""
    # perimeter is expected as (n_edges, 2, 3): [edge_start, edge_end]
    if perimeter.ndim != 3 or perimeter.shape[1:] != (2, 3):
        raise ValueError("perimeter must have shape (n_edges, 2, 3)")

    # Heuristic: maximum distance between boundary edges, measured using edge centers.
    edge_centers = perimeter.mean(axis=1)  # (n_edges, 3)
    center_deltas = edge_centers[:, None, :] - edge_centers[None, :, :]  # (n, n, 3)
    edge_center_dist = np.linalg.norm(center_deltas, axis=2)  # (n, n)
    max_edge_distance = float(edge_center_dist.max(initial=0.0))

    # If thickness is not specified, use 10% of the max edge distance.
    if thickness is None:
        thickness = 0.1 * max_edge_distance

    # start by extracting the back panel of the drawer


    raise NotImplementedError()

def _partition_perimeter_edges_into_sides(perimeter: np.ndarray) -> dict[int, list[int]]:
    """Partition perimeter edges into sides based on normals"""
    # compute edge vectors and normals
