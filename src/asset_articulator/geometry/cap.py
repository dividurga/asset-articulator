"""This file creates a cap geometry for any sliced stl"""

from pathlib import Path
from typing import Any
import trimesh
import numpy as np
import networkx as nx


class Cap:
    def __init__(self, mesh_path: Path, selection_metadata: dict[str, Any] | None = None):
        self.mesh_path = mesh_path
        self.selection_metadata = selection_metadata

    def _clean_mesh_for_boundary(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Return a copy with welded vertices so shared edges are counted correctly."""
        cleaned = mesh.copy()
        cleaned.remove_infinite_values()
        cleaned.remove_unreferenced_vertices()
        cleaned.merge_vertices()
        if hasattr(cleaned, "remove_duplicate_faces"):
            getattr(cleaned, "remove_duplicate_faces")()
        if hasattr(cleaned, "remove_degenerate_faces"):
            getattr(cleaned, "remove_degenerate_faces")()
        cleaned.remove_unreferenced_vertices()
        return cleaned

    def boundary_edges(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Return boundary unique edges (n, 2) from a vertex-welded mesh copy."""
        cleaned = self._clean_mesh_for_boundary(mesh)
        if len(cleaned.faces) == 0:
            return np.empty((0, 2), dtype=np.int64)

        edge_use_counts = np.bincount(
            cleaned.edges_unique_inverse, minlength=len(cleaned.edges_unique)
        )
        boundary_mask = edge_use_counts == 1
        return cleaned.edges_unique[boundary_mask]

    def boundary_loops(self, mesh: trimesh.Trimesh) -> list[np.ndarray]:
        """Return ordered boundary loops as vertex-index cycles on the cleaned mesh."""
        cleaned = self._clean_mesh_for_boundary(mesh)
        if len(cleaned.faces) == 0:
            return []

        edge_use_counts = np.bincount(
            cleaned.edges_unique_inverse, minlength=len(cleaned.edges_unique)
        )
        boundary_edges = cleaned.edges_unique[edge_use_counts == 1]
        if len(boundary_edges) == 0:
            return []

        graph = nx.Graph()
        graph.add_edges_from(boundary_edges.tolist())

        loops: list[np.ndarray] = []
        for component_nodes in nx.connected_components(graph):
            subgraph = graph.subgraph(component_nodes)
            if not all(deg == 2 for _, deg in subgraph.degree()):
                continue

            start = min(component_nodes)
            ordered: list[int] = [start]
            previous = -1
            current = start
            max_steps = len(component_nodes) + 2

            for _ in range(max_steps):
                neighbors = sorted(subgraph.neighbors(current))
                if previous == -1:
                    nxt = neighbors[0]
                else:
                    nxt = neighbors[0] if neighbors[1] == previous else neighbors[1]

                if nxt == start:
                    break

                ordered.append(nxt)
                previous, current = current, nxt

            if len(ordered) >= 3:
                loops.append(np.asarray(ordered, dtype=np.int64))

        return loops

    def _loop_score_from_selection_metadata(
        self, loop_vertices: np.ndarray, metadata: dict[str, Any]
    ) -> float:
        """Lower score means a loop is more likely to come from the cuboid selection cut."""
        if not {"center", "rotation", "extents"}.issubset(metadata.keys()):
            return float("inf")

        center = np.asarray(metadata["center"], dtype=float).reshape(3)
        rotation = np.asarray(metadata["rotation"], dtype=float).reshape(3, 3)
        extents = np.asarray(metadata["extents"], dtype=float).reshape(3)

        local = (loop_vertices - center) @ rotation
        face_distance = np.abs(np.abs(local) - extents)
        point_distance = np.min(face_distance, axis=1)
        return float(np.median(point_distance))

    def select_boundary_loop(self, mesh: trimesh.Trimesh) -> np.ndarray | None:
        """Select one boundary loop; uses metadata if available, otherwise the longest loop."""
        cleaned = self._clean_mesh_for_boundary(mesh)
        loops = self.boundary_loops(mesh)
        if not loops:
            return None

        if self.selection_metadata is None:
            return max(loops, key=len)

        scored = []
        for loop in loops:
            vertices = cleaned.vertices[loop]
            score = self._loop_score_from_selection_metadata(vertices, self.selection_metadata)
            scored.append((score, loop))

        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def selected_boundary_segments(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Return selected boundary loop as segments with shape (n, 2, 3)."""
        cleaned = self._clean_mesh_for_boundary(mesh)
        loop = self.select_boundary_loop(mesh)
        if loop is None or len(loop) < 3:
            return np.empty((0, 2, 3), dtype=float)

        points = cleaned.vertices[loop]
        rolled = np.roll(points, -1, axis=0)
        return np.stack([points, rolled], axis=1)

    def boundary_segments(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Return boundary edges as 3D line segments of shape (n, 2, 3)."""
        cleaned = self._clean_mesh_for_boundary(mesh)
        edges = self.boundary_edges(mesh)
        return cleaned.vertices[edges]

    def _plane_basis_from_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return centroid and orthonormal in-plane basis (u, v) from 3D loop points."""
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        covariance = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(covariance)

        # Smallest-eigenvalue direction is plane normal; the other two span the plane.
        order = np.argsort(eigvals)
        u = eigvecs[:, order[2]]
        v = eigvecs[:, order[1]]

        u = u / np.linalg.norm(u)
        v = v - np.dot(v, u) * u
        v = v / np.linalg.norm(v)
        return centroid, u, v

    def _triangulate_loop_2d(self, loop_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Triangulate polygon loop in 2D; returns vertices2d and triangle faces."""
        if len(loop_2d) < 3:
            return np.empty((0, 2), dtype=float), np.empty((0, 3), dtype=np.int64)

        try:
            import importlib

            sg = importlib.import_module("shapely.geometry")
            polygon = sg.Polygon(loop_2d)
            if polygon.is_valid and polygon.area > 0.0:
                vertices_2d, faces = trimesh.creation.triangulate_polygon(polygon)
                return np.asarray(vertices_2d, dtype=float), np.asarray(faces, dtype=np.int64)
        except Exception:
            pass

        # Fallback: simple fan triangulation from first vertex.
        vertices_2d = np.asarray(loop_2d, dtype=float)
        faces = []
        for i in range(1, len(vertices_2d) - 1):
            faces.append([0, i, i + 1])
        return vertices_2d, np.asarray(faces, dtype=np.int64)

    def _lift_2d_to_3d(
        self, vertices_2d: np.ndarray, centroid: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Lift 2D plane coordinates back to 3D points using plane basis."""
        return centroid + np.outer(vertices_2d[:, 0], u) + np.outer(vertices_2d[:, 1], v)

    def cap_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Return a new mesh with the boundary edges capped."""
        cleaned = self._clean_mesh_for_boundary(mesh)
        loop = self.select_boundary_loop(cleaned)
        if loop is None or len(loop) < 3:
            return cleaned

        loop_points = cleaned.vertices[loop]
        centroid, u, v = self._plane_basis_from_points(loop_points)
        loop_2d = np.column_stack(
            [
                np.dot(loop_points - centroid, u),
                np.dot(loop_points - centroid, v),
            ]
        )

        cap_vertices_2d, cap_faces = self._triangulate_loop_2d(loop_2d)
        if len(cap_faces) == 0:
            return cleaned

        cap_vertices_3d = self._lift_2d_to_3d(cap_vertices_2d, centroid, u, v)

        # Stitch cap triangles to existing boundary vertices whenever possible.
        # This avoids duplicate rim vertices that can keep the result non-watertight.
        scale = max(float(np.linalg.norm(cleaned.extents)), 1.0)
        tol = 1e-6 * scale

        merged_vertices = cleaned.vertices.copy()
        cap_vertex_to_mesh_index: list[int] = []
        for v2d, v3d in zip(cap_vertices_2d, cap_vertices_3d):
            distances = np.linalg.norm(loop_2d - v2d, axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= tol:
                cap_vertex_to_mesh_index.append(int(loop[nearest]))
            else:
                cap_vertex_to_mesh_index.append(len(merged_vertices))
                merged_vertices = np.vstack([merged_vertices, v3d])

        stitched_cap_faces = np.asarray(
            [[cap_vertex_to_mesh_index[int(i)] for i in face] for face in cap_faces],
            dtype=np.int64,
        )
        combined_faces = np.vstack([cleaned.faces, stitched_cap_faces])

        capped = trimesh.Trimesh(vertices=merged_vertices, faces=combined_faces, process=False)
        capped.remove_infinite_values()
        capped.merge_vertices()
        if hasattr(capped, "remove_duplicate_faces"):
            getattr(capped, "remove_duplicate_faces")()
        if hasattr(capped, "remove_degenerate_faces"):
            getattr(capped, "remove_degenerate_faces")()
        capped.remove_unreferenced_vertices()

        if not capped.is_watertight and hasattr(capped, "fill_holes"):
            try:
                getattr(capped, "fill_holes")()
            except Exception:
                pass

        capped.remove_unreferenced_vertices()
        return capped



