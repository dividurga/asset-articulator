from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


ArrayF = npt.NDArray[np.float64]


@dataclass
class Edge:
    """An edge represented by two world points.

    p0_world: shape (3,), first edge endpoint in world coordinates
    p1_world: shape (3,), second edge endpoint in world coordinates
    """

    p0_world: ArrayF
    p1_world: ArrayF

    def __post_init__(self) -> None:
        self.p0_world = np.asarray(self.p0_world, dtype=float).reshape(3)
        self.p1_world = np.asarray(self.p1_world, dtype=float).reshape(3)
        if np.allclose(self.p0_world, self.p1_world):
            raise ValueError("Edge endpoints must not be the same.")
