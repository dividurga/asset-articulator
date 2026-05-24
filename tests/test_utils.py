"""Tests for core geometry utilities."""

import numpy as np

from asset_articulator.geometry.cuboid import OrientedCuboid


def test_oriented_cuboid_local_to_world_identity() -> None:
    """OrientedCuboid with identity rotation maps local origin to world center."""
    cuboid = OrientedCuboid(
        center=np.array([1.0, 2.0, 3.0]),
        rotation=np.eye(3),
        extents=np.array([0.5, 0.5, 0.5]),
    )
    world_pts = cuboid.local_to_world(np.array([[0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(world_pts, np.array([[1.0, 2.0, 3.0]]))


def test_oriented_cuboid_round_trip() -> None:
    """world_to_local followed by local_to_world is identity."""
    cuboid = OrientedCuboid(
        center=np.array([0.5, -1.0, 2.0]),
        rotation=np.eye(3),
        extents=np.array([0.3, 0.4, 0.5]),
    )
    pts = np.array([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.1]])
    np.testing.assert_allclose(
        cuboid.local_to_world(cuboid.world_to_local(pts)),
        pts,
        atol=1e-12,
    )


def test_oriented_cuboid_contains() -> None:
    """Points inside the unit cuboid are detected correctly."""
    cuboid = OrientedCuboid(
        center=np.zeros(3),
        rotation=np.eye(3),
        extents=np.array([1.0, 1.0, 1.0]),
    )
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mask = cuboid.contains(pts)
    assert mask[0]
    assert not mask[1]
