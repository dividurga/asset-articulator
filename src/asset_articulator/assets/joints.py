from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointLimits:
    """Joint limits metadata for URDF export.

    Revolute joints should store values in radians.
    Prismatic joints should store values in meters.
    """

    lower: float
    upper: float
    unit: str
