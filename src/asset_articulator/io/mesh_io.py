from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def load_trimesh(path: str | Path) -> trimesh.Trimesh:
	"""Load a mesh file into a single ``trimesh.Trimesh`` instance."""
	loaded = trimesh.load_mesh(Path(path), process=False)

	if isinstance(loaded, trimesh.Scene):
		if not loaded.geometry:
			raise ValueError(f"No geometry found in mesh file: {path}")
		loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))

	if not isinstance(loaded, trimesh.Trimesh):
		raise TypeError(f"Expected trimesh.Trimesh, got {type(loaded)}")

	if len(loaded.faces) == 0:
		raise ValueError(f"Mesh has no faces: {path}")

	return loaded


def trimesh_to_open3d(
	mesh: trimesh.Trimesh,
	color: tuple[float, float, float] | None = None,
) -> o3d.geometry.TriangleMesh:
	"""Convert ``trimesh`` mesh into an Open3D triangle mesh."""
	o3d_mesh = o3d.geometry.TriangleMesh(
		vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=float)),
		triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
	)
	o3d_mesh.compute_vertex_normals()

	if color is not None:
		o3d_mesh.paint_uniform_color(np.asarray(color, dtype=float))

	return o3d_mesh
