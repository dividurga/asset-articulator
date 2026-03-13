from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


ArrayF = npt.NDArray[np.float64]


@dataclass
class OrientedCuboid:
    """An oriented cuboid represented by center, local-to-world rotation, and half-extents.

    Conventions
    -----------
    - center: shape (3,), cuboid center in world coordinates
    - rotation: shape (3, 3), local-to-world rotation
      Columns are the cuboid's local axes expressed in world coordinates.
    - extents: shape (3,), positive half-sizes along local x, y, z
    """

    center: ArrayF
    rotation: ArrayF
    extents: ArrayF

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(3)
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        self.extents = np.asarray(self.extents, dtype=float).reshape(3)

        if np.any(self.extents <= 0.0):
            raise ValueError("Cuboid extents must all be positive.")

    def world_to_local(self, points_world: ArrayF) -> ArrayF:
        """Transform world points of shape (N, 3) into cuboid-local coordinates."""
        points_world = np.asarray(points_world, dtype=float)
        return (points_world - self.center) @ self.rotation

    def local_to_world(self, points_local: ArrayF) -> ArrayF:
        """Transform local points of shape (N, 3) into world coordinates."""
        points_local = np.asarray(points_local, dtype=float)
        return points_local @ self.rotation.T + self.center