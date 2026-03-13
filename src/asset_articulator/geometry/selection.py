from __future__ import annotations

import numpy as np
import numpy.typing as npt

from asset_articulator.geometry.cuboid import OrientedCuboid


ArrayF = npt.NDArray[np.float64]


def build_axis_aligned_cuboid_from_points(
	points_world: ArrayF,
	padding_ratio: float = 0.03,
	min_extent: float = 1e-3,
) -> OrientedCuboid:
	"""Build an axis-aligned cuboid enclosing the given world points.

	Parameters
	----------
	points_world
		Selected world points with shape ``(N, 3)``, where ``N >= 2``.
	padding_ratio
		Relative padding added around selected points based on diagonal length.
	min_extent
		Minimum half-extent along each axis.
	"""
	points_world = np.asarray(points_world, dtype=float)
	if points_world.ndim != 2 or points_world.shape[1] != 3:
		raise ValueError("points_world must have shape (N, 3)")
	if len(points_world) < 2:
		raise ValueError("At least two points are required to define a cuboid.")

	mins = points_world.min(axis=0)
	maxs = points_world.max(axis=0)

	diagonal = float(np.linalg.norm(maxs - mins))
	padding = max(diagonal * float(padding_ratio), float(min_extent))
	mins -= padding
	maxs += padding

	center = 0.5 * (mins + maxs)
	extents = 0.5 * (maxs - mins)
	extents = np.maximum(extents, float(min_extent))

	return OrientedCuboid(
		center=center,
		rotation=np.eye(3, dtype=float),
		extents=extents,
	)
