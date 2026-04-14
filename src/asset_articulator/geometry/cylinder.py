from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


ArrayF = npt.NDArray[np.float64]


def _build_basis(axis: ArrayF) -> ArrayF:
    """Build a right-handed orthonormal basis with z = axis.

    Returns a (3, 3) matrix whose columns are [x, y, z] in world coords.
    """
    axis = np.asarray(axis, dtype=float).ravel()
    axis = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, axis))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, axis)
    x /= np.linalg.norm(x)
    y = np.cross(axis, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, axis])  # (3, 3)


@dataclass
class OrientedCylinder:
    """Oriented cylinder defined by center, unit axis, radius, and half-height.

    Local frame convention
    ----------------------
    - z axis  → cylinder axis direction
    - x, y    → arbitrary orthogonal radial directions
    - origin  → cylinder center
    """

    center: ArrayF      # (3,) world coords of cylinder center
    axis: ArrayF        # (3,) unit vector along cylinder axis
    radius: float
    half_height: float  # half the total cylinder height

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(3)
        self.axis = np.asarray(self.axis, dtype=float).reshape(3)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self.radius = float(self.radius)
        self.half_height = float(self.half_height)
        if self.radius <= 0.0:
            raise ValueError("Cylinder radius must be positive.")
        if self.half_height <= 0.0:
            raise ValueError("Cylinder half_height must be positive.")
        self._basis: ArrayF = _build_basis(self.axis)

    def world_to_local(self, points_world: ArrayF) -> ArrayF:
        """Transform (N, 3) world points to cylinder-local frame (z = axis)."""
        pts = np.asarray(points_world, dtype=float)
        return (pts - self.center) @ self._basis

    def local_to_world(self, points_local: ArrayF) -> ArrayF:
        """Transform (N, 3) local points to world frame."""
        pts = np.asarray(points_local, dtype=float)
        return pts @ self._basis.T + self.center

    def contains(self, points_world: ArrayF, tol: float = 1e-6) -> npt.NDArray[np.bool_]:
        """Return boolean mask of points strictly inside the cylinder."""
        local = self.world_to_local(np.asarray(points_world, dtype=float))
        r2 = local[:, 0] ** 2 + local[:, 1] ** 2
        return (r2 < (self.radius - tol) ** 2) & (np.abs(local[:, 2]) < self.half_height - tol)
