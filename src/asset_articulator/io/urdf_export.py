from __future__ import annotations

import os
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

import numpy as np

from asset_articulator.assets.joints import JointLimits
from asset_articulator.geometry.cuboid import OrientedCuboid
from asset_articulator.geometry.edge import Edge


def export_to_urdf(
    urdf_path: str | Path,
    parent_mesh_stl: str | Path,
    child_mesh_stl: str | Path,
    cuboid: OrientedCuboid,
    edge_of_interest: Edge,
    joint_type: str,
    joint_limits: JointLimits,
) -> None:
    """Export the selected articulation as a URDF file.

    Frame convention
    ----------------
    - Parent link frame == world frame used when exporting the parent STL.
    - Child link frame == joint origin, chosen as edge_of_interest.p0_world.
    - Child STL is expected to already be exported in this child-local frame.
      In other words, the child mesh vertices should have been shifted by
      -edge_of_interest.p0_world before writing child_mesh_stl.

    Notes
    -----
    - This exporter does not currently use `cuboid`, but it remains in the
      signature to keep the API stable.
    - Only visual geometry is exported for now. Collision and inertial tags
      can be added later once the frame conventions are settled.
    """
    del cuboid

    if joint_type not in {"revolute", "prismatic"}:
        raise ValueError(f"Unsupported joint type: {joint_type}")

    urdf_path = Path(urdf_path)
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_dir = urdf_path.parent.resolve()

    parent_mesh_stl = Path(parent_mesh_stl).resolve()
    child_mesh_stl = Path(child_mesh_stl).resolve()

    if not parent_mesh_stl.exists():
        raise FileNotFoundError(f"Parent mesh STL not found: {parent_mesh_stl}")
    if not child_mesh_stl.exists():
        raise FileNotFoundError(f"Child mesh STL not found: {child_mesh_stl}")

    parent_mesh_rel = Path(os.path.relpath(parent_mesh_stl, start=urdf_dir)).as_posix()
    child_mesh_rel = Path(os.path.relpath(child_mesh_stl, start=urdf_dir)).as_posix()

    joint_origin = np.asarray(edge_of_interest.p0_world, dtype=float)
    axis = np.asarray(
        edge_of_interest.p1_world - edge_of_interest.p0_world,
        dtype=float,
    )

    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        raise ValueError("Edge of interest is too short to define a valid joint axis.")
    axis = axis / axis_norm

    robot = ET.Element("robot", name="articulated_object")

    # ------------------------------------------------------------------
    # Parent link
    # Parent link frame == world frame.
    # Parent mesh is expected to already be expressed in this frame.
    # ------------------------------------------------------------------
    parent_link = ET.SubElement(robot, "link", name="parent")

    parent_visual = ET.SubElement(parent_link, "visual")
    ET.SubElement(parent_visual, "origin", xyz="0 0 0", rpy="0 0 0")
    parent_geometry = ET.SubElement(parent_visual, "geometry")
    ET.SubElement(parent_geometry, "mesh", filename=parent_mesh_rel)

    # ------------------------------------------------------------------
    # Child link
    # Child link frame == joint origin.
    # Child mesh is expected to already be expressed in this local frame.
    # ------------------------------------------------------------------
    child_link = ET.SubElement(robot, "link", name="child")

    child_visual = ET.SubElement(child_link, "visual")
    ET.SubElement(child_visual, "origin", xyz="0 0 0", rpy="0 0 0")
    child_geometry = ET.SubElement(child_visual, "geometry")
    ET.SubElement(child_geometry, "mesh", filename=child_mesh_rel)

    # TODO: add inertial and collision tags later

    # ------------------------------------------------------------------
    # Joint
    # Joint origin is expressed in the parent frame (world).
    # Joint axis is expressed in the joint/parent frame.
    # ------------------------------------------------------------------
    joint = ET.SubElement(robot, "joint", name="joint1", type=joint_type)
    ET.SubElement(joint, "parent", link="parent")
    ET.SubElement(joint, "child", link="child")
    ET.SubElement(joint, "origin", xyz=_format_xyz(joint_origin), rpy="0 0 0")
    ET.SubElement(joint, "axis", xyz=_format_xyz(axis))
    ET.SubElement(
        joint,
        "limit",
        lower=str(joint_limits.lower),
        upper=str(joint_limits.upper),
        effort="100.0",   # TODO: allow user to specify effort
        velocity="1.0",   # TODO: allow user to specify velocity
    )

    xml_str = _prettify_xml(robot)
    urdf_path.write_text(xml_str, encoding="utf-8")


def _format_xyz(v: np.ndarray) -> str:
    return f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}"


def _prettify_xml(elem: ET.Element) -> str:
    rough = ET.tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")